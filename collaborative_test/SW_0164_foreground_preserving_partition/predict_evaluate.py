"""Fixed-320 Source-T foreground-preserving readout and GT-last evaluation."""
from __future__ import annotations
import hashlib,json,os,sys,time,uuid
try: import resource as _resource
except ImportError: _resource=None
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
import h5py,numpy as np,torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143,geodesic_operator
from collaborative_test.SW_0162_connected_signed_partition import preflight as source,preflight_queue as source_queue
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver,preflight

SEED=1; IMAGE_IDS=np.asarray(contract.IMAGE_IDS,dtype=np.int64); COUNT,GRID=contract.COUNT,contract.GRID
DATASET=Path(sw130.source97.DATASET); SLOT_ROOT=ROOT/"trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE=ROOT/"collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256="4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
PREDICTION_ROOT=HERE/"results_archive"/"predictions_20261014"; WORK_ROOT=HERE/"results_archive"/"prediction_work_20261014"
SCORE_PATH=HERE/"results_archive"/"evaluation_seed1_20261014.json"
BOOTSTRAP_SEED,BOOTSTRAP_REPLICATES=164,10_000
ARMS=("foreground_induced","source_T_control")

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def _json_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def implementation_fingerprint():
 d={}; d.update(preflight.implementation_fingerprint()); d.update(sw143.implementation_fingerprint()); d.update(sw137.implementation_fingerprint())
 for p in (HERE/"predict_evaluate.py",HERE/"foreground_signed.py",HERE/"preflight.py",HERE/"DESIGN_KO.md",HERE/"protocol.json",Path(contract.__file__)):
  if not p.is_file(): raise FileNotFoundError(p)
  d[p.relative_to(ROOT).as_posix()]=sha256_file(p)
 return d
def _out(arm,root=PREDICTION_ROOT):
 if arm not in ARMS: raise ValueError("unregistered SW0164 artifact")
 return Path(root)/f"{arm}_seed1"
def _work(root=WORK_ROOT): return Path(root)/"foreground_induced_seed1"
def _rowdir(work,i): return Path(work)/"rows"/f"{i:04d}"
def _rss_max_bytes():
 if _resource is None: return None
 value=int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
 return value if sys.platform=="darwin" else value*1024

def _normalize_ref(ref):
 path=ref.get("prediction_path",ref.get("npz_path",ref.get("archive_path")))
 if path is None: raise ValueError("validated reference missing prediction archive path")
 ref["prediction_path"]=Path(path); ref["archive_path"]=Path(path); return ref

def _references():
 qcc=_normalize_ref(sw137.validate_prediction(sw137.OUTPUT_ROOT/"seed1",seed=SEED))
 t=_normalize_ref(sw143.validate_arm_prediction(sw143._arm_dir("T",SEED),seed=SEED))
 if not np.array_equal(qcc["ids"],IMAGE_IDS) or not np.array_equal(t["ids"],IMAGE_IDS) or t.get("arm")!="T": raise ValueError("fixed320 source prediction contract mismatch")
 slotdir=SLOT_ROOT/"seed1_epoch10"; snp=slotdir/"predictions.npz"; spp=slotdir/"protocol.json"
 baseline_sha=sha256_file(SLOT_BASELINE)
 if baseline_sha!=SLOT_BASELINE_SHA256: raise ValueError("registered Slot baseline changed")
 slotproto=json.loads(spp.read_text(encoding="utf-8")); slotsha=contract.validate_slot_protocol(slotproto,seed=SEED,prediction_path=snp)
 with np.load(snp,allow_pickle=False) as z:
  slotids=contract.require_fixed_image_ids(z["image_ids"],name="Slot fixed320 IDs"); slotlabels=contract.slot_native_pixels_to32(z["labels"])
 if not np.array_equal(slotids,IMAGE_IDS): raise ValueError("Slot IDs mismatch")
 slot={"ids":slotids,"labels":slotlabels,"prediction_sha256":slotsha,"protocol_sha256":sha256_file(spp),"prediction_path":snp,"archive_path":snp,"protocol_path":spp,"baseline_sha256":baseline_sha}
 return {"source_qcc":qcc,"source_T":t,"slot":slot}

def prepare_prediction_inputs_cpu(*,dataset=DATASET):
 refs=_references(); rgbpath=Path(sw130.VAL_RGB); metapath=rgbpath.with_suffix(rgbpath.suffix+".complete.json")
 meta=json.loads(metapath.read_text(encoding="utf-8-sig")); expected=hashlib.sha256(np.arange(1320,1640,dtype="<i8").tobytes()).hexdigest()
 if meta.get("status")!="complete" or meta.get("cache_shape")!=[320,128,128,3] or meta.get("ids_mapping_sha256")!=expected: raise ValueError("fixed320 RGB cache registration mismatch")
 rgb=np.load(rgbpath,mmap_mode="r",allow_pickle=False)
 if tuple(rgb.shape)!=(320,128,128,3) or rgb.dtype!=np.uint8: raise ValueError("fixed320 RGB schema invalid")
 hashes=[hashlib.sha256(np.asarray(rgb[i]).tobytes()).hexdigest() for i in range(COUNT)]
 dataset=Path(dataset)
 if dataset.resolve()!=DATASET.resolve() or not dataset.is_file(): raise ValueError("HDF5 scoring path is not canonical")
 st=dataset.stat()
 return {"references":refs,"rgb":rgb,"rgb_path":rgbpath,"rgb_manifest_sha256":sha256_file(metapath),"rgb_hashes":hashes,
  "dataset":dataset,"dataset_path":str(dataset.resolve()),"dataset_size_bytes":st.st_size,"dataset_mtime_ns":st.st_mtime_ns}

def _expected_source(): return source_queue.source_identity()
def validate_preflight(path=preflight.OUTPUT,*,expected_fingerprint=None,expected_source=None):
 return preflight.validate_report(path,expected_fingerprint=expected_fingerprint,expected_source=expected_source)

def prepare_prediction_cpu(*,dataset=DATASET):
 p=prepare_prediction_inputs_cpu(dataset=dataset)
 report=json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
 if not validate_preflight(preflight.OUTPUT,expected_fingerprint=preflight.implementation_fingerprint(),expected_source=_expected_source()): raise ValueError("SW0164 source8 preflight is missing/stale")
 p["preflight"]=report; p["preflight_sha256"]=sha256_file(preflight.OUTPUT); return p

def _bindings(p):
 s=p["preflight"]["source"]; refs=p["references"]; st=p["rgb_path"].stat()
 return {"experiment":"SW0164_foreground_preserving_partition","seed":1,"implementation_fingerprint":implementation_fingerprint(),
  "preflight_sha256":p["preflight_sha256"],"source_checkpoint_sha256":s["checkpoint_sha256"],"source_manifest_sha256":s["manifest_sha256"],
  "source_model_state_sha256":s["source_state_sha256_before"],"source_training_ids_sha256":s["ordered_train_ids_sha256"],
  "source_T_prediction_sha256":refs["source_T"]["prediction_sha256"],"source_T_protocol_sha256":refs["source_T"]["protocol_sha256"],
  "source_qcc_prediction_sha256":refs["source_qcc"]["prediction_sha256"],"source_qcc_protocol_sha256":refs["source_qcc"]["protocol_sha256"],
  "slot_prediction_sha256":refs["slot"]["prediction_sha256"],"slot_protocol_sha256":refs["slot"]["protocol_sha256"],"slot_baseline_sha256":refs["slot"]["baseline_sha256"],
  "rgb_manifest_sha256":p["rgb_manifest_sha256"],"rgb_size_bytes":st.st_size,"rgb_mtime_ns":st.st_mtime_ns,
  "image_ids_sha256":hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),"dataset_path":p["dataset_path"],
  "dataset_size_bytes":p["dataset_size_bytes"],"dataset_mtime_ns":p["dataset_mtime_ns"],"steps":1024,"settle":512,"live_tail":0,
  "optimizer_updates":0,"ground_truth_used_for_prediction":False}

def _atomic_json(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
 with tmp.open("x",encoding="utf-8",newline="\n") as f: json.dump(obj,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 if path.exists(): raise FileExistsError(path)
 os.replace(tmp,path)
def _write_row(work,i,image_id,rgb_sha,bsha,labels,control,q,a,gain,merge_count,traces,diagnostics=None):
 target=_rowdir(work,i)
 if target.exists(): raise FileExistsError(target)
 target.parent.mkdir(parents=True,exist_ok=True); stage=target.with_name(target.name+f".stage-{uuid.uuid4().hex}"); stage.mkdir()
 labels=np.asarray(labels,dtype=np.int32); control=np.asarray(control,dtype=np.int32)
 with (stage/"row.npz").open("xb") as f: np.savez_compressed(f,index=np.int64(i),image_id=np.int64(image_id),labels=labels,control=control); f.flush(); os.fsync(f.fileno())
 if diagnostics is None:
  diagnostics={"source_encode_seconds":0.,"rollout_seconds":1.,"cuda_peak_allocated_bytes":0,"cpu_maxrss_bytes":_rss_max_bytes(),
   "discarded_foreground_mass":0,"candidate_foreground_fraction":float(np.mean(labels>0)),"control_foreground_fraction":float(np.mean(control>0))}
 rec={"index":i,"image_id":image_id,"rgb_sha256":rgb_sha,"bindings_sha256":bsha,"labels_sha256":hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest(),
  "control_sha256":hashlib.sha256(control.astype("<i4").tobytes()).hexdigest(),"q_sha256":q,"actual_a_sha256":a,"gain":float(gain),"merge_count":int(merge_count),
  "trace_sha256":traces,"diagnostics":diagnostics,"npz_sha256":sha256_file(stage/"row.npz"),"foreground_support_exact":True}
 with (stage/"row.json").open("x",encoding="utf-8",newline="\n") as f: json.dump(rec,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.rename(stage,target)
def _read_row(work,i,image_id,rgb_sha,bsha):
 folder=_rowdir(work,i); p=folder/"row.npz"; rec=json.loads((folder/"row.json").read_text())
 if any(rec.get(k)!=v for k,v in {"index":i,"image_id":image_id,"rgb_sha256":rgb_sha,"bindings_sha256":bsha}.items()) or rec.get("npz_sha256")!=sha256_file(p): raise ValueError("row provenance mismatch")
 with np.load(p,allow_pickle=False) as z: labels=np.asarray(z["labels"]); control=np.asarray(z["control"])
 if labels.shape!=(32,32) or labels.dtype!=np.int32 or labels.min()<0 or labels.max()>1024: raise ValueError("candidate variable-K label contract invalid")
 if control.shape!=(32,32) or control.dtype!=np.int32 or control.min()<0 or control.max()>128: raise ValueError("Source-T QCC control label contract invalid")
 if rec.get("labels_sha256")!=hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest() or rec.get("control_sha256")!=hashlib.sha256(control.astype("<i4").tobytes()).hexdigest(): raise ValueError("row label hashes invalid")
 if rec.get("foreground_support_exact") is not True or not np.array_equal(labels>0,control>0) or not isinstance(rec.get("trace_sha256"),dict) or set(rec["trace_sha256"])!=set(source.TRACE_FIELDS): raise ValueError("row support/trace provenance invalid")
 for x in (rec.get("q_sha256"),rec.get("actual_a_sha256"),*rec["trace_sha256"].values()):
  if not isinstance(x,str) or len(x)!=64 or any(c not in "0123456789abcdef" for c in x): raise ValueError("row trace digest missing")
 if not np.isfinite(rec.get("gain",np.nan)) or rec["gain"]<=0 or type(rec.get("merge_count")) is not int or rec["merge_count"]<0: raise ValueError("row solver metadata invalid")
 d=rec.get("diagnostics"); required=("source_encode_seconds","rollout_seconds","cuda_peak_allocated_bytes","discarded_foreground_mass","candidate_foreground_fraction","control_foreground_fraction")
 if not isinstance(d,dict) or any(k not in d for k in required): raise ValueError("row runtime/readout diagnostics missing")
 if (not np.isfinite(d["source_encode_seconds"]) or d["source_encode_seconds"]<0 or not np.isfinite(d["rollout_seconds"]) or d["rollout_seconds"]<=0 or
     type(d["cuda_peak_allocated_bytes"]) is not int or d["cuda_peak_allocated_bytes"]<0 or d["discarded_foreground_mass"]!=0 or
     not np.isfinite(d["candidate_foreground_fraction"]) or not np.isfinite(d["control_foreground_fraction"]) or
     d["candidate_foreground_fraction"]!=float(np.mean(labels>0)) or d["control_foreground_fraction"]!=float(np.mean(control>0)) or
     d["candidate_foreground_fraction"]!=d["control_foreground_fraction"]): raise ValueError("row diagnostic values violate actual readout")
 if d.get("cpu_maxrss_bytes") is not None and (type(d["cpu_maxrss_bytes"]) is not int or d["cpu_maxrss_bytes"]<0): raise ValueError("invalid process RSS observation")
 return labels,control,rec

def _ledger_sha(work):
 hashes=[]
 for i in range(COUNT):
  p=_rowdir(work,i)/"row.json"
  if not p.is_file(): raise ValueError("candidate row ledger incomplete")
  hashes.append(sha256_file(p))
 return hashlib.sha256(json.dumps(hashes,separators=(",",":")).encode()).hexdigest()

@torch.no_grad()
def _predict_one(found,image,refs,i):
 encode_start=time.perf_counter()
 gamma=found.encode(image)
 encode_seconds=time.perf_counter()-encode_start
 rollout_start=time.perf_counter()
 trace,a=source.capture_source_coupling(found.wrapped.core.kuramoto,lambda:late_rollout(found.wrapped,gamma,total_steps=1024,live_tail_steps=0))
 rollout_seconds=time.perf_counter()-rollout_start
 source._trace_check(trace); source._validate_actual_A(a); q=source._actual_q(trace); qcc=sw136._qcc_labels(trace)[0]
 if not np.array_equal(qcc,refs["source_T"]["labels"][i]): raise ValueError("fresh Source-T control differs from archived SW0143 T")
 qnp=q[0].detach().cpu().numpy().astype(np.float64,copy=True); anp=a[0].detach().cpu().numpy().astype(np.float64,copy=True)
 gain=float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
 result=solver.foreground_induced_merge(qnp,anp,gain,qcc,grid_shape=(32,32))
 if not np.array_equal(result["labels"]>0,qcc>0): raise ValueError("candidate changed Source-T foreground membership")
 traces={k:source._tensor_sha(trace[k]) for k in source.TRACE_FIELDS}
 device=image.device; peak=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0
 diagnostics={"source_encode_seconds":float(encode_seconds),"rollout_seconds":float(rollout_seconds),"cuda_peak_allocated_bytes":peak,
  "cpu_maxrss_bytes":_rss_max_bytes(),"discarded_foreground_mass":int(result["discarded_mass"]),
  "candidate_foreground_fraction":float(np.mean(result["labels"]>0)),"control_foreground_fraction":float(np.mean(qcc>0))}
 return result["labels"],qcc.astype(np.int32),source._tensor_sha(q),source._tensor_sha(a),gain,result["merge_count"],traces,diagnostics

@torch.no_grad()
def predict_candidate(*,device="cuda:0",prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
 p=prepared or prepare_prediction_cpu(dataset=dataset); binds=_bindings(p); bsha=_json_sha(binds); work=_work(work_root); work.mkdir(parents=True,exist_ok=True)
 bp=work/"bindings.json"; binding={"bindings":binds,"bindings_sha256":bsha}
 if bp.exists() and json.loads(bp.read_text())!=binding: raise ValueError("existing rows bind different SW0164 inputs")
 if not bp.exists(): _atomic_json(bp,binding)
 complete=all(_rowdir(work,i).is_dir() for i in range(COUNT))
 if complete and (work/"source_state.json").is_file():
  receipt=json.loads((work/"source_state.json").read_text())
  if (receipt.get("source_unchanged") is True and receipt.get("bindings_sha256")==bsha and
      receipt.get("source_state_sha256_before")==binds["source_model_state_sha256"] and
      receipt.get("source_state_sha256_after")==binds["source_model_state_sha256"] and
      receipt.get("row_commits_sha256")==_ledger_sha(work)):
   return publish_predictions(prepared=p,prediction_root=prediction_root,work_root=work_root)
  raise ValueError("existing complete rows have an invalid source immutability receipt")
 if complete and not (work/"source_state.json").exists():
  raise ValueError("complete row ledger has no source immutability receipt; preserving for explicit recovery")
 if not complete:
  dev=torch.device(device)
  if dev.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("SW0164 prediction requires owner-held CUDA")
  torch.cuda.set_device(dev)
  from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
  found=load_native32_foundation(SEED,dev,verify_rgb_assets=False); found.encoder.eval(); found.wrapped.eval(); geodesic_operator.install_operator(found.wrapped.core.graph_generator,"T")
  if getattr(found.wrapped.core,"kuramoto_backend",None)!="factorized": raise ValueError("SW0164 prediction requires the registered factorized backend")
  before=source._source_state_digest(found)
  if before!=binds["source_model_state_sha256"]: raise ValueError("source model state differs from preflight")
  for i,image_id in enumerate(IMAGE_IDS.tolist()):
   if _rowdir(work,i).exists(): _read_row(work,i,image_id,p["rgb_hashes"][i],bsha); continue
   image=torch.from_numpy(np.asarray(p["rgb"][i:i+1]).copy()).to(dev)
   if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(dev)
   labels,control,qsha,asha,gain,merges,traces,diagnostics=_predict_one(found,image,p["references"],i)
   _write_row(work,i,image_id,p["rgb_hashes"][i],bsha,labels,control,qsha,asha,gain,merges,traces,diagnostics)
   if i%40==39: print(json.dumps({"stage":"SW0164_predict","done":i+1,"count":COUNT}),flush=True)
  after=source._source_state_digest(found)
  if before!=after: raise ValueError("source model changed during prediction")
  receipt={"source_state_sha256_before":before,"source_state_sha256_after":after,"source_unchanged":True,"bindings_sha256":bsha,"row_commits_sha256":_ledger_sha(work)}
  _atomic_json(work/"source_state.json",receipt)
 return publish_predictions(prepared=p,prediction_root=prediction_root,work_root=work_root)

def publish_predictions(*,prepared=None,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET):
 p=prepared or prepare_prediction_cpu(dataset=dataset); binds=_bindings(p); bsha=_json_sha(binds); work=_work(work_root)
 receipt=json.loads((work/"source_state.json").read_text())
 if receipt.get("source_unchanged") is not True or receipt.get("source_state_sha256_before")!=binds["source_model_state_sha256"] or receipt.get("source_state_sha256_after")!=binds["source_model_state_sha256"] or receipt.get("bindings_sha256")!=bsha or receipt.get("row_commits_sha256")!=_ledger_sha(work): raise ValueError("source immutability receipt invalid")
 candidate=[]; control=[]
 for i,image_id in enumerate(IMAGE_IDS.tolist()):
  l,c,_=_read_row(work,i,image_id,p["rgb_hashes"][i],bsha); candidate.append(l); control.append(c)
 candidate=np.stack(candidate); control=np.stack(control)
 if not np.array_equal(control,p["references"]["source_T"]["labels"]): raise ValueError("fresh Source-T QCC replay mismatch")
 ledger=_ledger_sha(work)
 for arm,labels in (("foreground_induced",candidate),("source_T_control",control)):
  out=_out(arm,prediction_root)
  if out.exists(): validate_prediction(arm,prediction_root=prediction_root,work_root=work_root,prepared=p); continue
  out.parent.mkdir(parents=True,exist_ok=True); stage=out.with_name(out.name+f".stage-{uuid.uuid4().hex}"); stage.mkdir()
  with (stage/"predictions.npz").open("xb") as f: np.savez_compressed(f,image_ids=IMAGE_IDS,labels=labels.astype(np.int32)); f.flush(); os.fsync(f.fileno())
  proto={"experiment":"SW0164_foreground_preserving_partition","status":"complete","arm":arm,"seed":SEED,
   "image_ids":IMAGE_IDS.tolist(),"count":COUNT,"grid_size":[32,32],"prediction_shape":[COUNT,32,32],"implementation_fingerprint":implementation_fingerprint(),
   "bindings":binds,"bindings_sha256":bsha,"prediction_sha256":sha256_file(stage/"predictions.npz"),"row_commits_sha256":ledger,
   "optimizer_updates":0,"ground_truth_used_for_prediction":False,"source_state_sha256_before":receipt["source_state_sha256_before"],
   "source_state_sha256_after":receipt["source_state_sha256_after"],"source_state_receipt_sha256":sha256_file(work/"source_state.json"),
   "steps":1024,"settle":512,"live_tail":0,"fresh_same_rollout_qcc_replay":True,
   "readout":"foreground-induced full-native signed energy; all Source-T foreground patches retained; no min8 refilter"}
  with (stage/"protocol.json").open("x",encoding="utf-8",newline="\n") as f: json.dump(proto,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
  os.rename(stage,out)
 return {"status":"prediction_complete","arms":["foreground_induced","source_T_control"]}

def validate_prediction(arm,*,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
 if arm not in ARMS: raise ValueError("unregistered SW0164 arm")
 p=prepared or prepare_prediction_cpu(dataset=dataset); binds=_bindings(p); bsha=_json_sha(binds); work=_work(work_root); out=_out(arm,prediction_root)
 rec=json.loads((work/"source_state.json").read_text())
 if rec.get("source_unchanged") is not True or rec.get("source_state_sha256_before")!=binds["source_model_state_sha256"] or rec.get("source_state_sha256_after")!=binds["source_model_state_sha256"] or rec.get("bindings_sha256")!=bsha or rec.get("row_commits_sha256")!=_ledger_sha(work): raise ValueError("SW0164 source-state receipt invalid")
 proto=json.loads((out/"protocol.json").read_text()); npz=out/"predictions.npz"
 expected={"experiment":"SW0164_foreground_preserving_partition","status":"complete","arm":arm,"seed":SEED,"image_ids":IMAGE_IDS.tolist(),"count":COUNT,"grid_size":[32,32],"prediction_shape":[COUNT,32,32],"implementation_fingerprint":implementation_fingerprint(),"bindings":binds,"bindings_sha256":bsha,"row_commits_sha256":_ledger_sha(work),"optimizer_updates":0,"ground_truth_used_for_prediction":False,"source_state_receipt_sha256":sha256_file(work/"source_state.json"),"fresh_same_rollout_qcc_replay":True}
 if any(proto.get(k)!=v for k,v in expected.items()) or proto.get("prediction_sha256")!=sha256_file(npz): raise ValueError("SW0164 prediction protocol/hash mismatch")
 with np.load(npz,allow_pickle=False) as z: ids=contract.require_fixed_image_ids(z["image_ids"]); labels=np.asarray(z["labels"])
 if labels.shape!=(COUNT,32,32) or labels.dtype.kind not in "iu" or labels.min()<0 or labels.max()>1024: raise ValueError("candidate labels exceed native variable-K range")
 if arm=="source_T_control" and not np.array_equal(labels,p["references"]["source_T"]["labels"]): raise ValueError("fresh control differs from archived T")
 for i,image_id in enumerate(IMAGE_IDS.tolist()):
  candidate,control,_=_read_row(work,i,image_id,p["rgb_hashes"][i],bsha)
  if not np.array_equal(control,p["references"]["source_T"]["labels"][i]) or not np.array_equal(candidate>0,control>0): raise ValueError("foreground support/reference replay mismatch")
  if arm=="foreground_induced" and not np.array_equal(candidate,labels[i]): raise ValueError("published labels differ from row ledger")
 return {"arm":arm,"ids":ids,"labels":labels.astype(np.int64),"prediction_sha256":proto["prediction_sha256"],"protocol_sha256":sha256_file(out/"protocol.json"),"protocol":proto,"directory":out}

def _bootstrap(a,b):
 d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
 if d.shape!=(COUNT,) or not np.isfinite(d).all(): raise ValueError("bad paired score vectors")
 rng=np.random.RandomState(BOOTSTRAP_SEED); ix=rng.randint(0,COUNT,size=(BOOTSTRAP_REPLICATES,COUNT)); vals=d[ix].mean(1)
 return {"mean":float(d.mean()),"ci95":[float(v) for v in np.percentile(vals,[2.5,97.5])],"seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES}

def score_all(*,output=SCORE_PATH,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
 output=Path(output)
 if output.exists(): raise FileExistsError(output)
 p=prepared or prepare_prediction_cpu(dataset=dataset)
 frozen={a:validate_prediction(a,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=p) for a in ARMS}
 refs=p["references"]
 frozen.update({n:{"ids":refs[k]["ids"],"labels":refs[k]["labels"],"prediction_sha256":refs[k]["prediction_sha256"],"protocol_sha256":refs[k]["protocol_sha256"]} for n,k in (("source_T","source_T"),("Slot70k","slot"))})
 files={n:(frozen[n]["directory"]/"predictions.npz" if "directory" in frozen[n] else Path(refs[{"source_T":"source_T","Slot70k":"slot"}[n]].get("prediction_path",refs[{"source_T":"source_T","Slot70k":"slot"}[n]].get("npz_path",refs[{"source_T":"source_T","Slot70k":"slot"}[n]].get("archive_path"))))) for n in frozen}
 protos={n:(frozen[n]["directory"]/"protocol.json" if "directory" in frozen[n] else refs[{"source_T":"source_T","Slot70k":"slot"}[n]]["protocol_path"]) for n in frozen}
 hashes={n:{"prediction_sha256":frozen[n]["prediction_sha256"],"protocol_sha256":frozen[n]["protocol_sha256"]} for n in frozen}
 for n in frozen:
  if not np.array_equal(frozen[n]["ids"],IMAGE_IDS) or sha256_file(files[n])!=hashes[n]["prediction_sha256"] or sha256_file(protos[n])!=hashes[n]["protocol_sha256"]: raise ValueError("frozen prediction/reference invalid before GT")
 with h5py.File(p["dataset"],"r") as h:
  if "mask" not in h or h["mask"].shape[0]<=int(IMAGE_IDS[-1]): raise ValueError("canonical validation masks absent")
  gt=contract.modal_native32(np.asarray(h["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1])+1]))
 scores={n:contract.production_metrics32(x["labels"],gt) for n,x in frozen.items()}
 means={n:{m:float(scores[n][m]["mean"]) for m in contract.METRICS} for n in scores}
 paired={f"foreground_induced_minus_{base}_{m}":_bootstrap(scores["foreground_induced"][m]["per_image"],scores[base][m]["per_image"])
  for base in ("source_T","Slot70k") for m in contract.METRICS}
 fg_exact=all(np.array_equal(scores["foreground_induced"]["foreground_iou"]["per_image"],scores["source_T"]["foreground_iou"]["per_image"]) for _ in (0,))
 if not fg_exact: raise ValueError("foreground IoU differs from Source-T despite registered exact-support invariant")
 gate={"fg_mean_above_source_T":means["foreground_induced"]["fg_ari"]>means["source_T"]["fg_ari"],
  "fg_paired_ci_lower_above_zero":paired["foreground_induced_minus_source_T_fg_ari"]["ci95"][0]>0,
  "foreground_iou_equal_source_T":fg_exact,
  "object_iou_not_below_source_T":means["foreground_induced"]["matched_object_iou"]>=means["source_T"]["matched_object_iou"]}
 gate["seed1_expansion_gate_passed"]=all(gate.values())
 for n in frozen:
  if sha256_file(files[n])!=hashes[n]["prediction_sha256"] or sha256_file(protos[n])!=hashes[n]["protocol_sha256"]: raise ValueError("prediction changed during scoring")
 st=p["dataset"].stat()
 if st.st_size!=p["dataset_size_bytes"] or st.st_mtime_ns!=p["dataset_mtime_ns"]: raise ValueError("HDF5 identity changed during score")
 report={"experiment":"SW0164_foreground_preserving_partition","status":"complete","seed_scope":[SEED],"implementation_fingerprint":implementation_fingerprint(),
  "scores":scores,"means":means,"paired_bootstrap":paired,"seed1_expansion_gate":gate,
  "all_metric_means_vs_slot":{m:{"delta":means["foreground_induced"][m]-means["Slot70k"][m],"strictly_above":means["foreground_induced"][m]>means["Slot70k"][m]} for m in contract.METRICS},
  "prediction_hashes_before_gt":hashes,"ground_truth_opened_after_all_predictions_validated":True,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,
  "dataset":{"path":p["dataset_path"],"size_bytes":st.st_size,"mtime_ns":st.st_mtime_ns},"bootstrap":{"rng":"numpy RandomState","seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES,"common_paired_image_indices":True},
  "source_T_foreground_iou_exact":fg_exact,"interpretation":"fixed seed1 conditional foreground partition pilot; no automatic seed expansion"}
 _atomic_json(output,report); return report

def validate_score(path=SCORE_PATH,*,prediction_root=PREDICTION_ROOT,work_root=WORK_ROOT,dataset=DATASET,prepared=None):
 path=Path(path); r=json.loads(path.read_text()); p=prepared or prepare_prediction_cpu(dataset=dataset)
 names=(*ARMS,"source_T","Slot70k"); preds={a:validate_prediction(a,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=p) for a in ARMS}; refs=p["references"]
 frozen={**preds,"source_T":{"ids":refs["source_T"]["ids"],"labels":refs["source_T"]["labels"],"prediction_sha256":refs["source_T"]["prediction_sha256"],"protocol_sha256":refs["source_T"]["protocol_sha256"]},"Slot70k":{"ids":refs["slot"]["ids"],"labels":refs["slot"]["labels"],"prediction_sha256":refs["slot"]["prediction_sha256"],"protocol_sha256":refs["slot"]["protocol_sha256"]}}
 expected_hashes={n:{"prediction_sha256":frozen[n]["prediction_sha256"],"protocol_sha256":frozen[n]["protocol_sha256"]} for n in names}
 if r.get("experiment")!="SW0164_foreground_preserving_partition" or r.get("status")!="complete" or r.get("seed_scope")!=[SEED] or r.get("implementation_fingerprint")!=implementation_fingerprint() or r.get("ground_truth_used_for_prediction") is not False or r.get("ground_truth_used_for_scoring") is not True or r.get("prediction_hashes_before_gt")!=expected_hashes: raise ValueError("score report provenance mismatch")
 # Recompute all production metrics independently to catch coherent score edits.
 scratch=path.with_name(path.name+f".verify-{uuid.uuid4().hex}.json")
 try:
  fresh=score_all(output=scratch,prediction_root=prediction_root,work_root=work_root,dataset=dataset,prepared=p)
  keys=("scores","means","paired_bootstrap","seed1_expansion_gate","all_metric_means_vs_slot","prediction_hashes_before_gt","ground_truth_opened_after_all_predictions_validated","ground_truth_used_for_prediction","ground_truth_used_for_scoring","dataset","bootstrap","source_T_foreground_iou_exact")
  if any(r.get(k)!=fresh.get(k) for k in keys): raise ValueError("independent GT metric recomputation mismatch")
 finally:
  if scratch.exists(): scratch.unlink()
 return {"status":"valid","score_sha256":sha256_file(path),"seed1_expansion_gate":r["seed1_expansion_gate"]}

def main(argv=None):
 import argparse
 p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("predict","score"),required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--output",type=Path)
 a=p.parse_args(argv); result=predict_candidate(device=a.device) if a.stage=="predict" else score_all(output=a.output or SCORE_PATH); print(json.dumps(result,allow_nan=False)); return 0
if __name__=="__main__": raise SystemExit(main())
