"""Fixed320 GT-last SW0152 native32 predictions and pilot scoring."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
import h5py,numpy as np,torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0152_source_preserving_partition_equivariance import train

NAME=train.NAME;SEED=1;ARMS=train.ARMS;COUNT=contract.COUNT;GRID=contract.GRID
IMAGE_IDS=np.asarray(contract.IMAGE_IDS,dtype=np.int64)
PREDICTION_ROOT=HERE/"predictions";SUMMARY=HERE/"results_archive"/"evaluation_seed1.json"
DATASET=Path(sw130.source97.DATASET)
SLOT_ROOT=ROOT/"trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE=ROOT/"collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256="4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
PILOT_REFERENCE=HERE/"pilot_reference.json"
PILOT_REFERENCE_SHA256="91dcd028809d34d668da4b180d5fa27ab01fb28cfc97b3015733d7f423d8da0d"
PRIMARY="production native32 QCC labels from same T1024 trace"
SECONDARY="argmax native32 spatial-diffusion P; diagnostic only"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def _fingerprint():
    fp=dict(train.implementation_fingerprint())
    paths=[HERE/"evaluate.py",HERE/"pilot_reference.json",ROOT/"collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
           ROOT/"collaborative_test/SW_0137_native32_source_qcc/run.py",
           ROOT/"collaborative_test/SW_0135_native32_spike_binding/score_slot32.py"]
    fp.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    return fp

def _fingerprint_sha():
    return hashlib.sha256(json.dumps(_fingerprint(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _fresh_source(device):
    f=train._load_foundation(device);src=train.source_contract();assets=sw130.validate_rgb_assets()
    if (f.provenance.get("source_core_sha256")!=src["source_sha256"]
        or f.provenance.get("source_manifest_sha256")!=sha(src["manifest_path"])
        or f.provenance.get("rgb_asset_validation")!=assets
        or list(map(int,f.image_ids))!=src["ids"] or not np.array_equal(f.pool_indices,src["rows"])):
        raise ValueError("evaluation source foundation differs from registered lineage")
    return f,src,assets

def _install_candidate(arm,device):
    trained=train.validate_training(arm)
    f,src,assets=_fresh_source(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(train.RESIDUAL_INIT_SEED)
        train.training.install_deep_encoder(f)
    f.encoder.to(device)
    try:
        payload=torch.load(trained["checkpoint"],map_location=device,weights_only=False)
        if not isinstance(payload,dict) or set(payload)!={"wrapped","encoder"}:
            raise ValueError("unexpected SW0152 checkpoint schema")
        f.wrapped.load_state_dict(payload["wrapped"],strict=True)
        f.encoder.load_state_dict(payload["encoder"],strict=True)
    except (OSError,RuntimeError,ValueError,EOFError) as exc:
        raise ValueError("cannot strictly restore SW0152 endpoint") from exc
    f.wrapped.eval();f.encoder.eval();f.wrapped.requires_grad_(False);f.encoder.requires_grad_(False)
    return f,src,assets,trained

def _write_prediction(name,arrays,protocol,outroot):
    directory=Path(outroot)/f"{name}_seed1"
    if directory.exists():raise FileExistsError(f"preserve existing SW0152 prediction {directory}")
    directory.mkdir(parents=True,exist_ok=False);npz=directory/"predictions.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream,image_ids=IMAGE_IDS,primary_labels=arrays["qcc"],
                            qcc_labels=arrays["qcc"],secondary_p_labels=arrays["p"])
        stream.flush();os.fsync(stream.fileno())
    protocol.update(prediction_sha256=sha(npz),implementation_fingerprint=_fingerprint(),
                    implementation_sha256=_fingerprint_sha())
    p=directory/"protocol.json"
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(protocol,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return {"npz":npz,"protocol":p,"prediction_sha256":sha(npz),"protocol_sha256":sha(p)}

@torch.no_grad()
def predict(arm,*,device="cuda:0",output_root=PREDICTION_ROOT):
    if arm not in (*ARMS,"source_p"):raise ValueError("unregistered readout arm")
    dev=torch.device(device)
    if dev.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("fixed320 prediction requires assigned CUDA")
    torch.cuda.set_device(dev)
    if arm=="source_p":f,src,assets=_fresh_source(dev);trained=None
    else:f,src,assets,trained=_install_candidate(arm,dev)
    before=train.training.state_sha({**{"wrapped."+k:v for k,v in f.wrapped.state_dict().items()},
                                      **{"encoder."+k:v for k,v in f.encoder.state_dict().items()}})
    ref=sw137.validate_prediction(sw137.OUTPUT_ROOT/"seed1",seed=SEED)
    cache=np.load(sw130.VAL_RGB,mmap_mode="r",allow_pickle=False)
    if cache.dtype!=np.uint8 or cache.shape!=(COUNT,128,128,3):raise ValueError("fixed320 RGB cache schema changed")
    ps=[];qs=[]
    for i in range(COUNT):
        rgb=torch.from_numpy(np.asarray(cache[i:i+1]).copy()).to(dev)
        r=forward_assignment(f,rgb,live_tail_steps=0)
        ps.append(r["primary_labels"][0].detach().cpu().numpy().astype(np.int64))
        qs.append(r["qcc"][0].detach().cpu().numpy().astype(np.int64))
        if i%32==31:print(json.dumps({"stage":"predict","arm":arm,"images":i+1,"total":COUNT}),flush=True)
    arrays={"p":np.stack(ps),"qcc":np.stack(qs)}
    if arrays["p"].shape!=(COUNT,GRID,GRID) or arrays["qcc"].shape!=(COUNT,GRID,GRID):raise ValueError("prediction shape mismatch")
    if arm=="source_p" and not np.array_equal(arrays["qcc"],ref["labels"]):
        raise ValueError("untrained source QCC differs from immutable SW0137 prediction")
    after=train.training.state_sha({**{"wrapped."+k:v for k,v in f.wrapped.state_dict().items()},
                                     **{"encoder."+k:v for k,v in f.encoder.state_dict().items()}})
    if before!=after:raise AssertionError("prediction modified model state")
    protocol={"experiment":NAME,"status":"complete","role":"source_p" if trained is None else "trained",
        "arm":arm,"seed":SEED,"image_ids":[1320,1639],"count":COUNT,"grid_size":[GRID,GRID],
        "prediction_shape":[COUNT,GRID,GRID],"primary_readout":PRIMARY,"secondary_readout":SECONDARY,
        "steps":1024,"settle":512,"live_tail":0,"microbatch_size":1,
        "ground_truth_used_for_prediction":False,"optimizer_updates":0,
        "source_core_sha256":src["source_sha256"],"source_manifest_sha256":sha(src["manifest_path"]),
        "source_training_ids_sha256":train.ids_sha(src["ids"]),"rgb_asset_validation":assets,
        "training_manifest_sha256":trained["manifest_sha256"] if trained else None,
        "checkpoint_sha256":trained["manifest"]["checkpoint_sha256"] if trained else None,
        "preflight_sha256":trained["manifest"]["bindings"]["preflight_sha256"] if trained else None,
        "source_qcc_prediction_sha256":ref["prediction_sha256"],
        "source_qcc_protocol_sha256":ref["protocol_sha256"]}
    return _write_prediction(arm,{"p":arrays["p"],"qcc":arrays["qcc"]},protocol,output_root)

def validate_prediction(name,*,prediction_root=PREDICTION_ROOT):
    directory=Path(prediction_root)/f"{name}_seed1";npz=directory/"predictions.npz";pp=directory/"protocol.json"
    protocol=json.loads(pp.read_text(encoding="utf-8-sig"));src=train.source_contract();assets=sw130.validate_rgb_assets()
    ref=sw137.validate_prediction(sw137.OUTPUT_ROOT/"seed1",seed=SEED)
    trained=None if name=="source_p" else train.validate_training(name)
    if name not in (*ARMS,"source_p"):raise ValueError("unregistered prediction name")
    if (protocol.get("experiment")!=NAME or protocol.get("status")!="complete"
        or protocol.get("role")!=("source_p" if name=="source_p" else "trained")
        or protocol.get("arm")!=name or protocol.get("seed")!=SEED or protocol.get("count")!=COUNT
        or protocol.get("image_ids")!=[1320,1639] or protocol.get("grid_size")!=[32,32]
        or protocol.get("prediction_shape")!=[COUNT,32,32] or protocol.get("steps")!=1024
        or protocol.get("primary_readout")!=PRIMARY or protocol.get("secondary_readout")!=SECONDARY
        or protocol.get("settle")!=512 or protocol.get("live_tail")!=0 or protocol.get("microbatch_size")!=1
        or protocol.get("optimizer_updates")!=0
        or protocol.get("ground_truth_used_for_prediction") is not False
        or protocol.get("source_core_sha256")!=src["source_sha256"]
        or protocol.get("source_manifest_sha256")!=sha(src["manifest_path"])
        or protocol.get("source_training_ids_sha256")!=train.ids_sha(src["ids"])
        or protocol.get("rgb_asset_validation")!=assets or protocol.get("implementation_fingerprint")!=_fingerprint()
        or protocol.get("implementation_sha256")!=_fingerprint_sha() or protocol.get("prediction_sha256")!=sha(npz)):
        raise ValueError("prediction provenance/contract invalid")
    if trained and (protocol.get("training_manifest_sha256")!=trained["manifest_sha256"]
            or protocol.get("checkpoint_sha256")!=trained["manifest"]["checkpoint_sha256"]
            or protocol.get("preflight_sha256")!=trained["manifest"]["bindings"]["preflight_sha256"]):
        raise ValueError("candidate prediction does not bind validated training checkpoint")
    if not trained and (protocol.get("training_manifest_sha256") is not None
            or protocol.get("checkpoint_sha256") is not None
            or protocol.get("source_qcc_prediction_sha256")!=ref["prediction_sha256"]
            or protocol.get("source_qcc_protocol_sha256")!=ref["protocol_sha256"]):
        raise ValueError("source P archive lacks exact immutable SW0137 reference")
    with np.load(npz,allow_pickle=False) as a:
        ids=contract.require_fixed_image_ids(a["image_ids"])
        qcc=np.asarray(a["primary_labels"]);p=np.asarray(a["secondary_p_labels"])
    for labels,maximum in ((qcc,1024),(p,10)):
        if (labels.shape!=(COUNT,GRID,GRID) or not np.issubdtype(labels.dtype,np.integer)
                or np.issubdtype(labels.dtype,np.bool_) or np.any(labels<0) or np.any(labels>maximum)):
            raise ValueError("prediction labels violate native32 schema")
    if name=="source_p" and not np.array_equal(qcc,ref["labels"]):raise ValueError("source QCC mismatch")
    if sha(npz)!=protocol["prediction_sha256"] or sha(pp)!=hashlib.sha256(pp.read_bytes()).hexdigest():
        raise ValueError("prediction bytes changed during validation")
    return {"ids":ids,"qcc":qcc.astype(np.int64,copy=False),"p":p.astype(np.int64,copy=False),
        "protocol":protocol,"npz":npz,"protocol_path":pp,"prediction_sha256":sha(npz),
        "protocol_sha256":sha(pp),"trained":trained}

def _slot_reference():
    directory=SLOT_ROOT/"seed1_epoch10";npz=directory/"predictions.npz";pp=directory/"protocol.json"
    protocol=json.loads(pp.read_text(encoding="utf-8-sig"));digest=contract.validate_slot_protocol(protocol,seed=1,prediction_path=npz)
    with np.load(npz,allow_pickle=False) as a:
        ids=contract.require_fixed_image_ids(a["image_ids"],name="Slot image_ids");pixels=np.asarray(a["labels"])
    labels=contract.slot_native_pixels_to32(pixels)
    return {"ids":ids,"labels":labels,"npz":npz,"protocol":pp,"prediction_sha256":digest,"protocol_sha256":sha(pp)}

def score_all(*,output=SUMMARY,prediction_root=PREDICTION_ROOT,dataset=DATASET):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from collaborative_test.SW_0150_spike_spatial_mixture import score_recovery as recovery
    output=Path(output)
    if output.exists():raise FileExistsError("preserve existing SW0152 score")
    # Validate every prediction and both baselines before opening target masks.
    pred={name:validate_prediction(name,prediction_root=prediction_root) for name in (*ARMS,"source_p")}
    source_ref=sw137.validate_prediction(sw137.OUTPUT_ROOT/"seed1",seed=SEED);slot=_slot_reference()
    if any(not np.array_equal(row["ids"],IMAGE_IDS) for row in pred.values()) or not np.array_equal(slot["ids"],IMAGE_IDS):
        raise ValueError("fixed320 predictions are not aligned")
    if not np.array_equal(pred["source_p"]["qcc"],source_ref["labels"]):raise ValueError("source P/QCC lineage mismatch")
    baseline_path=Path(SLOT_BASELINE)
    if sha(baseline_path)!=SLOT_BASELINE_SHA256:raise ValueError("registered Slot baseline bytes changed")
    baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_contract={"image_ids":[1320,1639,320],"native_image_size":[128,128],"patch_grid":[32,32],
        "patch_pixels":4,"metrics":list(contract.METRICS),
        "native_prediction":"direct32 labels; no conversion from 16-grid outputs",
        "slot_prediction":"saved native128 labels modal-pooled by production4x4 converter; existing perimeter-background remap preserved"}
    if (not recovery._same_json_schema(baseline.get("evaluation_contract"),expected_contract) or baseline.get("status")!="complete" or baseline.get("comparison")!="slot32 baseline only"):
        raise ValueError("registered Slot baseline schema changed")
    slot_means=recovery._validate_baseline_contract(baseline)
    if sha(PILOT_REFERENCE)!=PILOT_REFERENCE_SHA256:
        raise ValueError("registered SW0143T seed1 reference changed")
    pilot_reference=json.loads(PILOT_REFERENCE.read_text(encoding="utf-8-sig"))
    if (pilot_reference.get("reference")!="SW0143T" or pilot_reference.get("source_seed")!=SEED
            or set(pilot_reference.get("metrics",{}))!=set(contract.METRICS)
            or pilot_reference.get("evaluation_record_sha256")!="2ed9a82e85d610be4ad772a1648652846e962e38380a2986fe26980878870a5c"
            or pilot_reference.get("fg_promotion_margin")!=.02
            or pilot_reference.get("fg_promotion_floor")!=pilot_reference.get("metrics",{}).get("fg_ari",0)+.02
            or pilot_reference.get("paired_ci_required_vs_source_and_control") is not True):
        raise ValueError("SW0143T seed1 reference schema changed")
    source_t_means={m:float(pilot_reference["metrics"][m]) for m in contract.METRICS}
    if not all(np.isfinite(v) for v in source_t_means.values()):
        raise ValueError("SW0143T seed1 reference is nonfinite")
    freezes={f"{name}:npz":sha(row["npz"]) for name,row in pred.items()}
    freezes.update({f"{name}:protocol":sha(row["protocol_path"]) for name,row in pred.items()})
    source_paths=(source_ref["archive_path"],source_ref["protocol_path"])
    slot_paths=(slot["npz"],slot["protocol"])
    freezes.update({"source_qcc_prediction":sha(source_paths[0]),"source_qcc_protocol":sha(source_paths[1]),
                    "slot_prediction":sha(slot_paths[0]),"slot_protocol":sha(slot_paths[1]),
                    "slot_baseline":sha(baseline_path),"pilot_reference":sha(PILOT_REFERENCE)})
    if any(freezes[f"{name}:npz"]!=row["prediction_sha256"] or freezes[f"{name}:protocol"]!=row["protocol_sha256"] for name,row in pred.items()):
        raise ValueError("prediction changed before GT access")
    # Fixed-size ground truth is opened only after all prediction lineages and hashes are frozen.
    if sha(baseline_path)!=SLOT_BASELINE_SHA256:raise ValueError("Slot baseline changed immediately before scoring")
    if (freezes["source_qcc_prediction"]!=source_ref["prediction_sha256"]
            or freezes["source_qcc_protocol"]!=source_ref["protocol_sha256"]
            or freezes["slot_prediction"]!=slot["prediction_sha256"]
            or freezes["slot_baseline"]!=SLOT_BASELINE_SHA256
            or freezes["pilot_reference"]!=PILOT_REFERENCE_SHA256):
        raise ValueError("reference artifacts changed before GT access")
    with h5py.File(dataset,"r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0]<=1639:raise ValueError("fixed320 GT mask range absent")
        target=contract.modal_native32(np.asarray(h5["mask"][1320:1640]))
    metrics=lambda labels:contract.production_metrics32(labels,target)
    scores={name:{"qcc":metrics(row["qcc"]),"p":metrics(row["p"])} for name,row in pred.items()}
    scores["source97_qcc"]={"qcc":metrics(source_ref["labels"])};scores["slot32_seed1"]={"slot32":metrics(slot["labels"])}
    means={name:{readout:{m:float(result[m]["mean"]) for m in contract.METRICS} for readout,result in rows.items()}
           for name,rows in scores.items()}
    draws=np.random.RandomState(139).randint(0,COUNT,size=(10000,COUNT))
    paired={}
    def delta(a,b):return recovery.sealed._bootstrap_delta(a,b,draws)
    for m in contract.METRICS:
        paired[f"equivariance_qcc_minus_source_qcc_{m}"]=delta(scores["equivariance"]["qcc"][m]["per_image"],scores["source97_qcc"]["qcc"][m]["per_image"])
        paired[f"equivariance_qcc_minus_control_qcc_{m}"]=delta(scores["equivariance"]["qcc"][m]["per_image"],scores["mass_control"]["qcc"][m]["per_image"])
        paired[f"equivariance_qcc_minus_source_p_qcc_{m}"]=delta(scores["equivariance"]["qcc"][m]["per_image"],scores["source_p"]["qcc"][m]["per_image"])
    gate={"equivariance_qcc_fg_at_least_source_T_plus_0_02":means["equivariance"]["qcc"]["fg_ari"]>=source_t_means["fg_ari"]+.02,
        "equivariance_vs_source_qcc_fg_ci_lower_positive":paired["equivariance_qcc_minus_source_qcc_fg_ari"]["ci95"][0]>0,
        "equivariance_vs_control_fg_ci_lower_positive":paired["equivariance_qcc_minus_control_qcc_fg_ari"]["ci95"][0]>0}
    report={"experiment":NAME,"status":"complete","seed_scope":[SEED],"scores":scores,"means":means,
        "registered_slot32_three_seed_means":slot_means,"paired_bootstrap":paired,
        "bootstrap":{"samples":10000,"rng":"numpy RandomState","seed":139,"common_image_indices":True},
        "promotion_gate":gate,"may_promote_seed_expansion":all(gate.values()),
        "prediction_provenance":{n:{"prediction_sha256":r["prediction_sha256"],"protocol_sha256":r["protocol_sha256"],
            "training_manifest_sha256":r["protocol"].get("training_manifest_sha256"),"checkpoint_sha256":r["protocol"].get("checkpoint_sha256")} for n,r in pred.items()},
        "source_t_seed1_reference":source_t_means,"source_t_reference_path":str(PILOT_REFERENCE),"source_t_reference_sha256":PILOT_REFERENCE_SHA256,"ground_truth_opened_after_all_predictions_validated":True,
        "ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,
        "limits":"Single seed1 pilot; no 70k fresh native32 training, scaling, gating-necessity or universal superiority claim."}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    return report

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("predict","score"),required=True)
    parser.add_argument("--arm",choices=(*ARMS,"source_p"));parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args(argv)
    result=predict(args.arm,device=args.device) if args.stage=="predict" and args.arm else score_all() if args.stage=="score" else None
    if result is None:parser.error("--arm required for predict")
    print(json.dumps({"status":result.get("status","complete"),"stage":args.stage,"arm":args.arm,
        "ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":args.stage=="score"},allow_nan=False),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
