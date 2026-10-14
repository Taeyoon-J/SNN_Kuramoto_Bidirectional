"""Shared-rollout fixed-320 prediction and GT-last evaluation for SW0163."""
from __future__ import annotations
import hashlib, json, os, sys, time, uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
import h5py, numpy as np, torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143, geodesic_operator
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0161_supervised_spike_partition import objective
from collaborative_test.SW_0162_connected_signed_partition import preflight as sw162
from collaborative_test.SW_0163_relation_context_assignment import model, train

SEED=1
IMAGE_IDS=np.asarray(contract.IMAGE_IDS,dtype=np.int64)
COUNT,GRID=contract.COUNT,contract.GRID
PREDICTION_ROOT=HERE/"results_archive"/"predictions_20261014"
WORK_ROOT=HERE/"results_archive"/"prediction_work_20261014"
SCORE_PATH=HERE/"results_archive"/"evaluation_seed1_20261014.json"
DATASET=Path(sw130.source97.DATASET)
SLOT_ROOT=ROOT/"trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE=ROOT/"collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256="4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
ARMS=model.ARMS
BOOTSTRAP_SEED,BOOTSTRAP_REPLICATES=163,10_000

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def _json_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _validated_prediction_path(ref):
    path=ref.get("prediction_path",ref.get("npz_path",ref.get("archive_path")))
    if path is None: raise ValueError("validated prediction reference omitted archive path")
    return Path(path)

def implementation_fingerprint():
    result={}
    result.update(train.implementation_fingerprint())
    result.update(sw162.implementation_fingerprint())
    for p in (HERE/"predict_evaluate.py",HERE/"protocol.json",HERE/"DESIGN_KO.md",
              HERE/"model.py",Path(contract.__file__),Path(objective.__file__),Path(sw136.__file__)):
        if not p.is_file(): raise FileNotFoundError(p)
        result[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    return result

def _out(arm,root=PREDICTION_ROOT):
    if arm not in (*ARMS,"source_T_control"): raise ValueError("unregistered SW0163 arm")
    return Path(root)/f"{arm}_seed1"

def _work(root=WORK_ROOT): return Path(root)/"shared_source_seed1"

def _references():
    qcc=sw137.validate_prediction(sw137.OUTPUT_ROOT/"seed1",seed=SEED)
    t=sw143.validate_arm_prediction(sw143._arm_dir("T",SEED),seed=SEED)
    # The frozen validators expose different path field names (SW0137 uses
    # archive_path; SW0143 uses npz_path). Normalize only after validation.
    for ref in (qcc,t):
        ref["prediction_path"]=_validated_prediction_path(ref)
        ref["archive_path"]=_validated_prediction_path(ref)
    for name,ref in (("source QCC",qcc),("source T",t)):
        if not np.array_equal(ref["ids"],IMAGE_IDS): raise ValueError(f"{name} fixed IDs mismatch")
    if t.get("arm")!="T": raise ValueError("wrong registered source-T arm")
    sd=SLOT_ROOT/"seed1_epoch10"
    snp,sp=sd/"predictions.npz",sd/"protocol.json"
    bsha=sha256_file(SLOT_BASELINE)
    if bsha!=SLOT_BASELINE_SHA256: raise ValueError("Slot baseline contract changed")
    proto=json.loads(sp.read_text(encoding="utf-8"))
    psha=contract.validate_slot_protocol(proto,seed=SEED,prediction_path=snp)
    with np.load(snp,allow_pickle=False) as z:
        sids=contract.require_fixed_image_ids(z["image_ids"],name="Slot IDs")
        slabels=contract.slot_native_pixels_to32(z["labels"])
    if not np.array_equal(sids,IMAGE_IDS): raise ValueError("Slot fixed IDs mismatch")
    slot={"ids":sids,"labels":slabels,"prediction_sha256":psha,"protocol_sha256":sha256_file(sp),
          "prediction_path":snp,"protocol_path":sp,"baseline_sha256":bsha}
    return {"source_qcc":qcc,"source_T":t,"slot":slot}

def prepare_prediction_inputs_cpu(*,dataset=DATASET):
    refs=_references()
    rgb_path=Path(sw130.VAL_RGB); meta_path=rgb_path.with_suffix(rgb_path.suffix+".complete.json")
    meta=json.loads(meta_path.read_text(encoding="utf-8-sig"))
    idsha=hashlib.sha256(np.arange(1320,1640,dtype="<i8").tobytes()).hexdigest()
    if meta.get("status")!="complete" or meta.get("cache_shape")!=[320,128,128,3] or meta.get("ids_mapping_sha256")!=idsha:
        raise ValueError("registered fixed320 RGB cache metadata mismatch")
    rgb=np.load(rgb_path,mmap_mode="r",allow_pickle=False)
    if tuple(rgb.shape)!=(320,128,128,3) or rgb.dtype!=np.uint8: raise ValueError("RGB cache schema mismatch")
    rgb_hashes=[hashlib.sha256(np.asarray(rgb[i]).tobytes()).hexdigest() for i in range(COUNT)]
    dataset=Path(dataset)
    if dataset.resolve()!=DATASET.resolve() or not dataset.is_file(): raise ValueError("noncanonical HDF5 path")
    st=dataset.stat()
    return {"references":refs,"rgb":rgb,"rgb_path":rgb_path,"rgb_manifest_sha256":sha256_file(meta_path),
      "rgb_hashes":rgb_hashes,"dataset":dataset,"dataset_path":str(dataset.resolve()),
      "dataset_size_bytes":st.st_size,"dataset_mtime_ns":st.st_mtime_ns}

def prepare_prediction_cpu(*,dataset=DATASET):
    p=prepare_prediction_inputs_cpu(dataset=dataset)
    p["training"]={a:train.validate_training_artifact(a) for a in ARMS}
    p["training_manifest_sha256"]={a:sha256_file(train.TRAIN_ROOT/f"{a}_seed1"/"manifest.json") for a in ARMS}
    return p

def _bindings(prepared):
    src=prepared["training"][ARMS[0]]["bindings"]
    return {"experiment":"SW0163_relation_context_assignment","seed":SEED,
      "implementation_fingerprint":implementation_fingerprint(),
      "arms":{a:{"manifest_sha256":prepared["training_manifest_sha256"][a],
                   "checkpoint_sha256":prepared["training"][a]["checkpoint_sha256"],
                   "model_state_sha256":prepared["training"][a]["final_model_state_sha256"]} for a in ARMS},
      "source_checkpoint_sha256":src["source_checkpoint_sha256"],"source_manifest_sha256":src["source_manifest_sha256"],
      "source_model_state_sha256":src["source_model_state_sha256"],
      "source_trace_manifest_sha256":src["source_trace_manifest_sha256"],
      "context_cache_manifest_sha256":src["context_cache_manifest_sha256"],
      "preflight_report_sha256":src["preflight_report_sha256"],
      "source_T_prediction_sha256":prepared["references"]["source_T"]["prediction_sha256"],
      "source_T_protocol_sha256":prepared["references"]["source_T"]["protocol_sha256"],
      "source_qcc_prediction_sha256":prepared["references"]["source_qcc"]["prediction_sha256"],
      "source_qcc_protocol_sha256":prepared["references"]["source_qcc"]["protocol_sha256"],
      "slot_prediction_sha256":prepared["references"]["slot"]["prediction_sha256"],
      "slot_protocol_sha256":prepared["references"]["slot"]["protocol_sha256"],
      "slot_baseline_sha256":prepared["references"]["slot"]["baseline_sha256"],
      "rgb_manifest_sha256":prepared["rgb_manifest_sha256"],
      "rgb_size_bytes":prepared["rgb_path"].stat().st_size,"rgb_mtime_ns":prepared["rgb_path"].stat().st_mtime_ns,
      "image_ids_sha256":hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),
      "dataset_path":prepared["dataset_path"],"dataset_size_bytes":prepared["dataset_size_bytes"],
      "dataset_mtime_ns":prepared["dataset_mtime_ns"],"steps":1024,"settle":512,"live_tail":0,
      "ground_truth_used_for_prediction":False}

def _write_once(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())

def _rowfolder(work,i): return Path(work)/"rows"/f"{i:04d}"

def _source_state_path(work): return Path(work)/"source_state.json"

def _source_state_record(work,bind_sha,expected_sha):
    path=_source_state_path(work)
    if not path.is_file(): raise ValueError("immutable source-state receipt is absent")
    record=json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment")!="SW0163_relation_context_assignment"
            or record.get("bindings_sha256")!=bind_sha
            or record.get("expected_source_state_sha256")!=expected_sha
            or record.get("source_state_sha256_before")!=expected_sha
            or record.get("source_state_sha256_after")!=expected_sha
            or record.get("source_unchanged") is not True
            or record.get("row_count")!=COUNT
            or record.get("row_commits_sha256")!=_ledger_sha(work)
            or record.get("implementation_fingerprint")!=implementation_fingerprint()):
        raise ValueError("source-state receipt does not bind the complete prediction ledger")
    return record

def _write_row(work,i,image_id,rgb_sha,bind_sha,labels_by_arm,control,trace_shas):
    if set(labels_by_arm)!=set(ARMS): raise ValueError("every committed source row requires exactly four assignment heads")
    target=_rowfolder(work,i)
    if target.exists(): raise FileExistsError(f"preserving existing row {i}")
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=target.with_name(target.name+f".staging-{uuid.uuid4().hex}"); stage.mkdir()
    labels=np.stack([np.asarray(labels_by_arm[a],dtype=np.int32) for a in ARMS]); control=np.asarray(control,dtype=np.int32)
    with (stage/"row.npz").open("xb") as f:
        np.savez_compressed(f,index=np.asarray(i,np.int64),image_id=np.asarray(image_id,np.int64),labels=labels,control=control)
        f.flush(); os.fsync(f.fileno())
    trace_shas=dict(trace_shas); q_sha=trace_shas.pop("q_sha256",None); a_sha=trace_shas.pop("actual_a_sha256",None)
    if set(trace_shas)!=set(sw162.TRACE_FIELDS) or any(not isinstance(x,str) or len(x)!=64 for x in (*trace_shas.values(),q_sha,a_sha)):
        raise ValueError("row trace/Q/A digest schema invalid")
    record={"index":i,"image_id":image_id,"rgb_sha256":rgb_sha,"bindings_sha256":bind_sha,
      "labels_sha256":hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest(),
      "control_sha256":hashlib.sha256(control.astype("<i4").tobytes()).hexdigest(),
      "trace_sha256":trace_shas,"q_sha256":q_sha,"actual_a_sha256":a_sha,
      "npz_sha256":sha256_file(stage/"row.npz")}
    _write_once(stage/"row.json",record)
    os.rename(stage,target)

def _read_row(work,i,image_id,rgb_sha,bind_sha):
    d=_rowfolder(work,i); r=json.loads((d/"row.json").read_text()); zpath=d/"row.npz"
    if any(r.get(k)!=v for k,v in {"index":i,"image_id":image_id,"rgb_sha256":rgb_sha,"bindings_sha256":bind_sha}.items()) or r.get("npz_sha256")!=sha256_file(zpath):
        raise ValueError(f"row {i} provenance/hash mismatch")
    with np.load(zpath,allow_pickle=False) as z:
        if int(z["index"])!=i or int(z["image_id"])!=image_id: raise ValueError("row archive identity mismatch")
        labs=np.asarray(z["labels"]); ctrl=np.asarray(z["control"])
    if labs.shape!=(len(ARMS),32,32) or labs.dtype!=np.int32 or labs.min()<0 or labs.max()>10 or r.get("labels_sha256")!=hashlib.sha256(labs.astype("<i4").tobytes()).hexdigest():
        raise ValueError("bad arm label row")
    if ctrl.shape!=(32,32) or ctrl.dtype!=np.int32 or ctrl.min()<0 or ctrl.max()>128 or r.get("control_sha256")!=hashlib.sha256(ctrl.astype("<i4").tobytes()).hexdigest(): raise ValueError("bad control row")
    if set(r.get("trace_sha256",{}))!=set(sw162.TRACE_FIELDS) or any(not isinstance(x,str) or len(x)!=64 or any(c not in "0123456789abcdef" for c in x) for x in r["trace_sha256"].values()): raise ValueError("trace hashes missing")
    for key in ("q_sha256","actual_a_sha256"):
        value=r.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value): raise ValueError("Q/A hashes missing")
    return {a:labs[j] for j,a in enumerate(ARMS)},ctrl,r

def _load_heads(prepared,device):
    heads={}
    for arm in ARMS:
        d=train.TRAIN_ROOT/f"{arm}_seed1"; ck=torch.load(d/"checkpoint.pt",map_location="cpu",weights_only=False)
        if sha256_file(d/"checkpoint.pt")!=prepared["training"][arm]["checkpoint_sha256"]: raise ValueError("trained endpoint changed after CPU validation")
        head=model.RelationContextAssignment(arm,seed=train.MESSAGE_SEED,binder_seed=train.HEAD_SEED).to(device)
        if train._model_state_sha(ck["assignment"])!=prepared["training"][arm]["final_model_state_sha256"]: raise ValueError("head state hash changed")
        head.load_state_dict(ck["assignment"],strict=True); head.eval(); heads[arm]=head
    return heads

@torch.no_grad()
def predict_source_image(found, heads, image, references, row_index):
    """One actual source rollout, reused by every registered head."""
    gamma=found.encode(image)
    trace,actual_a=sw162.capture_source_coupling(found.wrapped.core.kuramoto,
        lambda:late_rollout(found.wrapped,gamma,total_steps=1024,live_tail_steps=0))
    sw162._validate_actual_A(actual_a)
    q=sw162._actual_q(trace); qcc=sw136._qcc_labels(trace)[0]
    if not np.array_equal(qcc,references["source_T"]["labels"][row_index]):
        raise ValueError("fresh QCC does not replay archived SW0143 T")
    spikes=trace["component_spikes"][:,:,:,512:].detach()
    if spikes.shape!=(1,4,1024,512) or q.shape!=(1,1024,1024) or actual_a.shape!=(1,1024,1024):
        raise ValueError("source trace geometry invalid")
    output={}
    for arm,head in heads.items():
        p,_,_=head(spikes,q,actual_a)
        output[arm]=objective.canonical_hard_labels(p)[0].cpu().numpy().astype(np.int32)
    hashes={k:sw162._tensor_sha(trace[k]) for k in sw162.TRACE_FIELDS}
    hashes.update(q_sha256=sw162._tensor_sha(q),actual_a_sha256=sw162._tensor_sha(actual_a))
    return output,qcc,hashes

@torch.no_grad()
def predict_all(*,device="cuda:0",prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
    prepared=prepared or prepare_prediction_cpu(dataset=dataset)
    binds=_bindings(prepared); bsha=_json_sha(binds); work=_work(work_root)
    binding_record={"bindings":binds,"bindings_sha256":bsha}
    if work.exists() and (work/"bindings.json").is_file() and json.loads((work/"bindings.json").read_text())!=binding_record: raise ValueError("existing rows bind different inputs")
    work.mkdir(parents=True,exist_ok=True)
    if not (work/"bindings.json").exists(): _write_once(work/"bindings.json",binding_record)
    committed=all(_rowfolder(work,i).is_dir() for i in range(COUNT))
    existing=[_out(a,prediction_root).exists() for a in (*ARMS,"source_T_control")]
    if any(existing) and not committed: raise ValueError("existing prediction artifact without a complete row ledger; preserving partial")
    if committed:
        for i,image_id in enumerate(IMAGE_IDS.tolist()): _read_row(work,i,int(image_id),prepared["rgb_hashes"][i],bsha)
        try:
            _source_state_record(work,bsha,prepared["training"][ARMS[0]]["bindings"]["source_model_state_sha256"])
        except ValueError:
            # Rows alone do not prove the source remained immutable. Re-load
            # the frozen source and re-check its state before final publication.
            pass
        else:
            return publish_predictions(prepared=prepared,prediction_root=prediction_root,work_root=work_root)
    dev=torch.device(device)
    if dev.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("SW0163 fixed320 prediction requires owner-held CUDA")
    torch.cuda.set_device(dev)
    found=sw155._load_foundation(dev); found.encoder.eval(); found.wrapped.eval()
    geodesic_operator.install_operator(found.wrapped.core.graph_generator,"T")
    before=sw162._source_state_digest(found)
    if before!=prepared["training"][ARMS[0]]["bindings"]["source_model_state_sha256"]: raise ValueError("source foundation lineage changed")
    heads=_load_heads(prepared,dev)
    for i,image_id in enumerate(IMAGE_IDS.tolist()):
        if _rowfolder(work,i).exists(): _read_row(work,i,image_id,prepared["rgb_hashes"][i],bsha); continue
        rgb=torch.from_numpy(np.asarray(prepared["rgb"][i:i+1]).copy()).to(dev)
        outputs,qcc,tsh=predict_source_image(found,heads,rgb,prepared["references"],i)
        _write_row(work,i,int(image_id),prepared["rgb_hashes"][i],bsha,outputs,qcc,tsh)
        if i%40==39: print(json.dumps({"stage":"SW0163_predict","done":i+1,"count":COUNT}),flush=True)
    after=sw162._source_state_digest(found)
    if before!=after: raise ValueError("frozen source changed during prediction")
    receipt={"experiment":"SW0163_relation_context_assignment","bindings_sha256":bsha,
      "expected_source_state_sha256":prepared["training"][ARMS[0]]["bindings"]["source_model_state_sha256"],
      "source_state_sha256_before":before,"source_state_sha256_after":after,"source_unchanged":True,
      "row_count":COUNT,"row_commits_sha256":_ledger_sha(work),"implementation_fingerprint":implementation_fingerprint()}
    if _source_state_path(work).exists():
        if json.loads(_source_state_path(work).read_text())!=receipt: raise ValueError("existing source-state receipt changed")
    else: _write_once(_source_state_path(work),receipt)
    return publish_predictions(prepared=prepared,prediction_root=prediction_root,work_root=work_root)

def _ledger_sha(work):
    hs=[]
    for i in range(COUNT):
        p=_rowfolder(work,i)/"row.json"
        if not p.is_file(): raise ValueError("prediction row ledger incomplete")
        hs.append(sha256_file(p))
    return hashlib.sha256(json.dumps(hs,separators=(",",":")).encode()).hexdigest()

def publish_predictions(*,prepared=None,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET):
    prepared=prepared or prepare_prediction_cpu(dataset=dataset); binds=_bindings(prepared); bsha=_json_sha(binds); work=_work(work_root)
    source_receipt=_source_state_record(work,bsha,prepared["training"][ARMS[0]]["bindings"]["source_model_state_sha256"])
    arrays={a:[] for a in ARMS}; control=[]
    for i,image_id in enumerate(IMAGE_IDS.tolist()):
        labs,ctrl,_=_read_row(work,i,image_id,prepared["rgb_hashes"][i],bsha); control.append(ctrl)
        for a in ARMS: arrays[a].append(labs[a])
    control=np.stack(control)
    if not np.array_equal(control,prepared["references"]["source_T"]["labels"]): raise ValueError("fresh control fails exact archived T replay")
    arrays={a:np.stack(v) for a,v in arrays.items()}; ledger=_ledger_sha(work)
    for arm in (*ARMS,"source_T_control"):
        out=_out(arm,prediction_root)
        if out.exists(): validate_prediction(arm,prediction_root=prediction_root,work_root=work_root,prepared=prepared); continue
        out.parent.mkdir(parents=True,exist_ok=True); stage=out.with_name(out.name+f".staging-{uuid.uuid4().hex}"); stage.mkdir(parents=True,exist_ok=False)
        labels=control if arm=="source_T_control" else arrays[arm]
        with (stage/"predictions.npz").open("xb") as f:
            np.savez_compressed(f,image_ids=IMAGE_IDS,labels=labels.astype(np.int32)); f.flush(); os.fsync(f.fileno())
        proto={"experiment":"SW0163_relation_context_assignment","status":"complete","arm":arm,"seed":SEED,
          "image_ids":IMAGE_IDS.tolist(),"count":COUNT,"grid_size":[32,32],"prediction_shape":[COUNT,32,32],
          "implementation_fingerprint":implementation_fingerprint(),"bindings":binds,"bindings_sha256":bsha,
          "prediction_sha256":sha256_file(stage/"predictions.npz"),"row_commits_sha256":ledger,
          "ground_truth_used_for_prediction":False,"optimizer_updates":0,
          "source_state_sha256_before":source_receipt["source_state_sha256_before"],
          "source_state_sha256_after":source_receipt["source_state_sha256_after"],
          "source_state_receipt_sha256":sha256_file(_source_state_path(work)),
          "fresh_same_rollout_qcc_replay":True,"steps":1024,"settle":512,"live_tail":0,
          "readout":"SW0161 canonical hard labels on trained source-spike relation-context head"}
        _write_once(stage/"protocol.json",proto)
        if out.exists(): raise FileExistsError(out)
        for attempt in range(8):
            try:
                os.rename(stage,out); break
            except PermissionError:
                if out.exists() or attempt==7: raise
                time.sleep(.03*(attempt+1))
    return {"status":"prediction_complete","arms":list((*ARMS,"source_T_control"))}

def validate_prediction(arm,*,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
    if arm not in (*ARMS,"source_T_control"): raise ValueError("unregistered arm")
    prepared=prepared or prepare_prediction_cpu(dataset=dataset); binds=_bindings(prepared); bsha=_json_sha(binds); out=_out(arm,prediction_root)
    source_receipt=_source_state_record(_work(work_root),bsha,binds["source_model_state_sha256"])
    bp=_work(work_root)/"bindings.json"
    if not bp.is_file() or json.loads(bp.read_text(encoding="utf-8"))!={"bindings":binds,"bindings_sha256":bsha}:
        raise ValueError("shared prediction row bindings file is absent or changed")
    proto=json.loads((out/"protocol.json").read_text()); npz=out/"predictions.npz"
    if any(proto.get(k)!=v for k,v in {"experiment":"SW0163_relation_context_assignment","status":"complete","arm":arm,"seed":SEED,"image_ids":IMAGE_IDS.tolist(),"count":COUNT,"grid_size":[32,32],"prediction_shape":[COUNT,32,32],"implementation_fingerprint":implementation_fingerprint(),"bindings":binds,"bindings_sha256":bsha,"ground_truth_used_for_prediction":False,"optimizer_updates":0,"row_commits_sha256":_ledger_sha(_work(work_root))}.items()): raise ValueError("SW0163 prediction provenance mismatch")
    if (proto.get("prediction_sha256")!=sha256_file(npz)
            or proto.get("source_state_sha256_before")!=binds["source_model_state_sha256"]
            or proto.get("source_state_sha256_after")!=binds["source_model_state_sha256"]
            or proto.get("source_state_sha256_before")!=source_receipt["source_state_sha256_before"]
            or proto.get("source_state_sha256_after")!=source_receipt["source_state_sha256_after"]
            or proto.get("source_state_receipt_sha256")!=sha256_file(_source_state_path(_work(work_root)))
            or proto.get("fresh_same_rollout_qcc_replay") is not True
            or any(proto.get(k)!=v for k,v in {"steps":1024,"settle":512,"live_tail":0}.items())): raise ValueError("prediction hashes/control contract invalid")
    with np.load(npz,allow_pickle=False) as z: ids=contract.require_fixed_image_ids(z["image_ids"]); labels=np.asarray(z["labels"])
    label_limit=128 if arm=="source_T_control" else 10
    if labels.shape!=(COUNT,32,32) or labels.dtype.kind not in "iu" or labels.min()<0 or labels.max()>label_limit: raise ValueError("SW0163 label contract invalid")
    if arm=="source_T_control" and not np.array_equal(labels,prepared["references"]["source_T"]["labels"]): raise ValueError("same-rollout control does not replay T")
    expected_t=prepared["references"]["source_T"]["labels"]
    for i,image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels,row_control,_=_read_row(_work(work_root),i,image_id,prepared["rgb_hashes"][i],bsha)
        if not np.array_equal(row_control,expected_t[i]): raise ValueError(f"fresh same-rollout control row {i} changed")
        if arm in ARMS and not np.array_equal(row_labels[arm],labels[i]): raise ValueError(f"published {arm} differs from shared source row {i}")
    return {"arm":arm,"ids":ids,"labels":labels.astype(np.int64),"prediction_sha256":proto["prediction_sha256"],"protocol_sha256":sha256_file(out/"protocol.json"),"protocol":proto,"directory":out}

def _bootstrap(a,b):
    delta=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    if delta.shape!=(COUNT,) or not np.isfinite(delta).all(): raise ValueError("bad paired metric vectors")
    rng=np.random.RandomState(BOOTSTRAP_SEED); ix=rng.randint(0,COUNT,size=(BOOTSTRAP_REPLICATES,COUNT)); draws=delta[ix].mean(1)
    return {"mean":float(delta.mean()),"ci95":[float(x) for x in np.percentile(draws,[2.5,97.5])],"seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES}

def _metrics(frozen,gt): return {n:contract.production_metrics32(v["labels"],gt) for n,v in frozen.items()}

def score_all(*,output=SCORE_PATH,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
    output=Path(output)
    if output.exists(): raise FileExistsError(f"preserving existing score {output}")
    prepared=prepared or prepare_prediction_cpu(dataset=dataset)
    frozen={n:validate_prediction(n,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=prepared) for n in (*ARMS,"source_T_control")}
    refs=prepared["references"]
    frozen.update({n:{"ids":refs[k]["ids"],"labels":refs[k]["labels"],"prediction_sha256":refs[k]["prediction_sha256"],"protocol_sha256":refs[k]["protocol_sha256"]} for n,k in (("source_T","source_T"),("source_qcc","source_qcc"),("Slot70k","slot"))})
    archive_paths={n:(frozen[n]["directory"]/"predictions.npz" if "directory" in frozen[n] else _validated_prediction_path(refs[{"source_T":"source_T","source_qcc":"source_qcc","Slot70k":"slot"}[n]])) for n in frozen}
    protocols={n:(frozen[n]["directory"]/"protocol.json" if "directory" in frozen[n] else refs[{"source_T":"source_T","source_qcc":"source_qcc","Slot70k":"slot"}[n]]["protocol_path"]) for n in frozen}
    hashes={n:{"prediction_sha256":frozen[n]["prediction_sha256"],"protocol_sha256":frozen[n]["protocol_sha256"]} for n in frozen}
    for n in frozen:
        if not np.array_equal(frozen[n]["ids"],IMAGE_IDS) or sha256_file(archive_paths[n])!=hashes[n]["prediction_sha256"] or sha256_file(protocols[n])!=hashes[n]["protocol_sha256"]: raise ValueError("prediction/reference changed before GT")
    with h5py.File(prepared["dataset"],"r") as h:
        if "mask" not in h or h["mask"].shape[0]<=int(IMAGE_IDS[-1]): raise ValueError("missing canonical validation masks")
        gt=contract.modal_native32(np.asarray(h["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1])+1]))
    scores=_metrics(frozen,gt); means={n:{m:float(scores[n][m]["mean"]) for m in contract.METRICS} for n in scores}
    paired={}
    for arm in ARMS:
        for base in ("source_T","source_qcc","Slot70k",*ARMS):
            if arm==base: continue
            for metric in contract.METRICS: paired[f"{arm}_minus_{base}_{metric}"]=_bootstrap(scores[arm][metric]["per_image"],scores[base][metric]["per_image"])
    gate={"Q_A_fg_above_source_T":means["Q_A"]["fg_ari"]>means["source_T"]["fg_ari"],
      "Q_A_fg_above_raw":means["Q_A"]["fg_ari"]>means["raw_only"]["fg_ari"],
      "Q_A_ci_lower_vs_source_T_positive":paired["Q_A_minus_source_T_fg_ari"]["ci95"][0]>0,
      "Q_A_ci_lower_vs_raw_positive":paired["Q_A_minus_raw_only_fg_ari"]["ci95"][0]>0,
      "foreground_iou_at_least_T":means["Q_A"]["foreground_iou"]>=means["source_T"]["foreground_iou"],
      "object_iou_at_least_T":means["Q_A"]["matched_object_iou"]>=means["source_T"]["matched_object_iou"]}
    gate["registered_gate_passed"]=all(gate.values())
    for n in frozen:
        if sha256_file(archive_paths[n])!=hashes[n]["prediction_sha256"] or sha256_file(protocols[n])!=hashes[n]["protocol_sha256"]: raise ValueError("prediction changed during GT scoring")
    st=prepared["dataset"].stat()
    if st.st_size!=prepared["dataset_size_bytes"] or st.st_mtime_ns!=prepared["dataset_mtime_ns"]: raise ValueError("HDF5 changed during scoring")
    mean_comparisons={}
    for arm in ARMS:
        mean_comparisons[arm]={base:{metric:{"delta":means[arm][metric]-means[base][metric],"strictly_above":means[arm][metric]>means[base][metric]} for metric in contract.METRICS} for base in ("source_T","source_qcc","Slot70k")}
    report={"experiment":"SW0163_relation_context_assignment","status":"complete","seed_scope":[SEED],"implementation_fingerprint":implementation_fingerprint(),"scores":scores,"means":means,"mean_comparisons":mean_comparisons,"paired_bootstrap":paired,"registered_gate":gate,"prediction_hashes_before_gt":hashes,"ground_truth_opened_after_all_predictions_validated":True,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,"dataset":{"path":prepared["dataset_path"],"size_bytes":st.st_size,"mtime_ns":st.st_mtime_ns},"bootstrap":{"rng":"numpy RandomState","seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES,"common_paired_image_indices":True},"diagnostic_deltas":{"Q_A_minus_Q_only_fg_ari":paired["Q_A_minus_Q_only_fg_ari"],"Q_A_minus_A_only_fg_ari":paired["Q_A_minus_A_only_fg_ari"]},"scope":"supervised fixed320 seed1 pilot; no automatic expansion or unsupervised-success claim"}
    _write_once(output,report); return report

def validate_score(path=SCORE_PATH,*,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
    path=Path(path); report=json.loads(path.read_text()); prepared=prepared or prepare_prediction_cpu(dataset=dataset)
    names=(*ARMS,"source_T_control","source_T","source_qcc","Slot70k")
    frozen={n:validate_prediction(n,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=prepared) for n in (*ARMS,"source_T_control")}
    refs=prepared["references"]
    frozen.update({n:{"ids":refs[k]["ids"],"labels":refs[k]["labels"],"prediction_sha256":refs[k]["prediction_sha256"],"protocol_sha256":refs[k]["protocol_sha256"]} for n,k in (("source_T","source_T"),("source_qcc","source_qcc"),("Slot70k","slot"))})
    expected_hashes={n:{"prediction_sha256":frozen[n]["prediction_sha256"],"protocol_sha256":frozen[n]["protocol_sha256"]} for n in names}
    if report.get("experiment")!="SW0163_relation_context_assignment" or report.get("status")!="complete" or report.get("seed_scope")!=[SEED] or report.get("implementation_fingerprint")!=implementation_fingerprint() or report.get("ground_truth_used_for_prediction") is not False or report.get("ground_truth_used_for_scoring") is not True or report.get("prediction_hashes_before_gt")!=expected_hashes: raise ValueError("score provenance/GT-last contract mismatch")
    # Independent recomputation from immutable labels catches internally consistent report edits.
    scratch=path.with_name(path.name+f".verify-{uuid.uuid4().hex}.json")
    try:
        fresh=score_all(output=scratch,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=prepared)
        for key in ("scores","means","mean_comparisons","paired_bootstrap","registered_gate","prediction_hashes_before_gt","ground_truth_opened_after_all_predictions_validated","ground_truth_used_for_prediction","ground_truth_used_for_scoring","dataset","bootstrap","diagnostic_deltas"):
            if report.get(key)!=fresh.get(key): raise ValueError(f"score recomputation mismatch: {key}")
    finally:
        if scratch.exists(): scratch.unlink()
    return report

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("predict","score"),required=True); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(argv)
    if a.stage=="predict": print(json.dumps(predict_all(device=a.device),allow_nan=False))
    else: print(json.dumps(score_all(),allow_nan=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
