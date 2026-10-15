"""Eight-TRAIN-image GT-free source parity and foreground-support preflight."""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
import numpy as np
import torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
from collaborative_test.SW_0162_connected_signed_partition import preflight as source
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver

SEEDS=(0,2)
POSITIONS=(0,1,2,3,2048,2049,2050,2051)
STEPS,SETTLE=1024,512
OUTPUT_DIR=HERE/"results_archive"/"preflight_20261014"
TRACE_FIELDS=source.TRACE_FIELDS

def output_for_seed(seed):
    if int(seed) not in SEEDS: raise ValueError("SW0165 preflight is only registered for seeds 0 and 2")
    return OUTPUT_DIR/f"source8_seed{int(seed)}.json"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def implementation_fingerprint():
    result={}
    result.update(source.implementation_fingerprint())
    for path in (HERE/"foreground_signed.py",Path(solver.__file__),HERE/"preflight.py",HERE/"stage_queue.py",HERE/"DESIGN_KO.md",HERE/"INPUT_COMPATIBILITY_CHECKLIST.md",HERE/"protocol.json"):
        if not path.is_file(): raise FileNotFoundError(path)
        result[path.relative_to(ROOT).as_posix()]=sha256_file(path)
    return result

def prepare_preflight_cpu(seed):
    """Seed-explicit source contract and 8 TRAIN RGB rows, before CUDA init."""
    seed=int(seed)
    if seed not in SEEDS: raise ValueError("SW0165 accepts only prospective seeds 0 and 2")
    checkpoint,manifest_path,manifest,pool_rows,ordered_ids,source_sha=sw130.source_contract(seed)
    cache_path=Path(sw130.TRAIN_RGB); complete_path=cache_path.with_suffix(cache_path.suffix+".complete.json")
    if not complete_path.is_file(): raise FileNotFoundError(complete_path)
    complete=json.loads(complete_path.read_text(encoding="utf-8-sig"))
    expected_rows=np.concatenate((np.arange(1000,dtype="<i8"),np.arange(1640,70640,dtype="<i8")))
    expected_ids_sha=hashlib.sha256(expected_rows.tobytes()).hexdigest()
    if complete.get("status")!="complete" or complete.get("cache_shape")!=[70000,128,128,3] or complete.get("ids_mapping_sha256")!=expected_ids_sha:
        raise ValueError("registered TRAIN RGB completion metadata changed")
    cache=np.load(cache_path,mmap_mode="r",allow_pickle=False)
    if tuple(cache.shape)!=(70000,128,128,3) or cache.dtype!=np.uint8: raise ValueError("TRAIN RGB schema mismatch")
    rgb8,selected_rows,selected_ids=source.select_train_rows(cache,pool_rows,ordered_ids,POSITIONS)
    cache_stat=cache_path.stat(); complete_sha=sha256_file(complete_path); del cache
    encoder_path=Path(sw130.ENCODER_SOURCE); stats_path=Path(sw130.FEATURE_STATS)
    if not encoder_path.is_file() or not stats_path.is_file(): raise FileNotFoundError("registered encoder/statistics missing")
    stats=torch.load(stats_path,map_location="cpu",weights_only=True)
    mean=torch.as_tensor(stats["mean"]); std=torch.as_tensor(stats["std"]); clip=float(stats.get("clip",3.0))
    if stats.get("mode")!="standardize" or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.all(std>0) or not np.isfinite(clip) or clip<=0:
        raise ValueError("registered feature statistics invalid")
    assets={"train_manifest_sha256":complete_sha,"train_cache_shape":[70000,128,128,3],"train_ids_mapping_sha256":expected_ids_sha,
      "selected_rgb_sha256":hashlib.sha256(rgb8.tobytes()).hexdigest(),"encoder_source_path":str(encoder_path),"encoder_source_sha256":sha256_file(encoder_path),
      "feature_stats_path":str(stats_path),"feature_stats_sha256":sha256_file(stats_path),"feature_stats_mode":"standardize","feature_clip":clip,
      "scope":"registered TRAIN RGB only; no validation images or masks opened"}
    return {"seed":seed,"checkpoint":checkpoint,"manifest_path":manifest_path,"manifest":manifest,"pool_rows":pool_rows,"ordered_ids":ordered_ids,
      "source_sha":source_sha,"assets":assets,"rgb8":rgb8,"selected_rows":selected_rows,"selected_ids":selected_ids,
      "rgb_sha":hashlib.sha256(rgb8.tobytes()).hexdigest(),"cache_size_bytes":cache_stat.st_size,"cache_mtime_ns":cache_stat.st_mtime_ns}

def source_identity(seed):
    prepared=prepare_preflight_cpu(seed)
    return {"seed":int(seed),"checkpoint_sha256":prepared["source_sha"],
      "manifest_sha256":sha256_file(prepared["manifest_path"]),
      "ordered_train_ids_sha256":hashlib.sha256(np.asarray(prepared["ordered_ids"],dtype="<i8").tobytes()).hexdigest(),
      "rgb_cache_size_bytes":prepared["cache_size_bytes"],"rgb_cache_mtime_ns":prepared["cache_mtime_ns"],
      "rgb_manifest_sha256":prepared["assets"]["train_manifest_sha256"],"source_positions":list(POSITIONS),
      "selected_pool_rows":list(prepared["selected_rows"]),"selected_image_ids":list(prepared["selected_ids"]),
      "rgb_batch_sha256":prepared["rgb_sha"],"encoder_source_sha256":prepared["assets"]["encoder_source_sha256"],
      "feature_stats_sha256":prepared["assets"]["feature_stats_sha256"],"feature_stats_mode":prepared["assets"]["feature_stats_mode"],
      "feature_clip":prepared["assets"]["feature_clip"],
      "rgb_image_sha256":[hashlib.sha256(image.tobytes()).hexdigest() for image in prepared["rgb8"]]}

def _rng(device):
    state=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state(device).clone() if device.type=="cuda" else None
    return state,cuda

def _restore_rng(state,device):
    cpu,cuda=state; torch.random.set_rng_state(cpu)
    if device.type=="cuda": torch.cuda.set_rng_state(cuda,device)

def _atomic_json(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    with tmp.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(record,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def validate_report(path=None,*,expected_seed=None,expected_fingerprint=None,expected_source=None):
    if path is None: return False
    path=Path(path)
    if not path.is_file(): return False
    try:
        r=json.loads(path.read_text(encoding="utf-8")); s=r["source"]; op=r["source_operator"]
        if not (r.get("experiment")=="SW0165_foreground_partition_replication" and r.get("status")=="source8_preflight_complete" and r.get("seed") in SEEDS and (expected_seed is None or r.get("seed")==int(expected_seed)) and r.get("ground_truth_used") is False and r.get("optimizer_updates")==0 and len(r.get("images",[]))==8): return False
        if expected_fingerprint is not None and r.get("implementation_fingerprint")!=expected_fingerprint: return False
        if expected_source is not None:
            per_image=expected_source.get("rgb_image_sha256")
            if any(s.get(k)!=v for k,v in expected_source.items() if k!="rgb_image_sha256"): return False
            if per_image is not None and (len(per_image)!=8 or [x.get("rgb_sha256") for x in r["images"]]!=per_image): return False
        if (op.get("arm")!="T" or op.get("steps")!=1024 or op.get("settle")!=512 or op.get("live_tail")!=0 or op.get("optimizer_updates")!=0 or op.get("backend")!="factorized" or s.get("source_state_sha256_before")!=s.get("source_state_sha256_after")): return False
        for x in r["images"]:
            if not all(x.get(k) is True for k in ("trace_parity","q_parity","qcc_parity","foreground_support_exact","deterministic_solver_replay")): return False
            if not isinstance(x.get("trace_sha256"),dict) or set(x["trace_sha256"])!=set(source.TRACE_FIELDS): return False
            for value in (x.get("q_sha256"),x.get("actual_a_sha256"),*x["trace_sha256"].values()):
                if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value): return False
        return True
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError): return False

validate_preflight=validate_report

@torch.no_grad()
def run_preflight(*,seed,device="cuda:0",output=None):
    seed=int(seed); output=Path(output if output is not None else output_for_seed(seed))
    if output.exists(): raise FileExistsError(f"preserving existing SW0165 source preflight: {output}")
    prepared=prepare_preflight_cpu(seed)
    dev=torch.device(device)
    if dev.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("SW0165 source preflight requires owner-held CUDA")
    torch.cuda.set_device(dev)
    from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
    found=load_native32_foundation(seed,dev,verify_rgb_assets=False)
    found.encoder.eval(); found.wrapped.eval()
    if list(map(int,found.image_ids))!=prepared["ordered_ids"] or not np.array_equal(found.pool_indices,np.asarray(prepared["pool_rows"],dtype=np.int64)):
        raise ValueError("loaded source foundation order differs from validated SW0143 training source")
    graph=found.wrapped.core.graph_generator
    if getattr(found.wrapped.core,"kuramoto_backend",None)!="factorized":
        raise ValueError("SW0165 requires the registered factorized Kuramoto backend")
    geodesic_operator.install_operator(graph,"T")
    gain_tensor=getattr(graph,"log_coupling_gain",None)
    if not torch.is_tensor(gain_tensor) or not torch.isfinite(gain_tensor).all(): raise ValueError("source graph gain invalid")
    gain=float(torch.exp(gain_tensor.detach()).cpu())
    if not np.isfinite(gain) or gain<=0: raise ValueError("source gain must be positive finite")
    before=source._source_state_digest(found)
    rows=[]; peak=0
    for pos,pool_row,image_id,rgb in zip(POSITIONS,prepared["selected_rows"],prepared["selected_ids"],prepared["rgb8"]):
        image=torch.from_numpy(np.asarray(rgb[None]).copy()).to(dev)
        gamma=found.encode(image)
        if tuple(gamma.shape)!=(1,8,1024) or not torch.isfinite(gamma).all(): raise ValueError("source encoder gamma invalid")
        rng=_rng(dev)
        plain=late_rollout(found.wrapped,gamma,total_steps=STEPS,live_tail_steps=0)
        source._trace_check(plain)
        plain_q=source._actual_q(plain)
        plain_labels=sw136._qcc_labels(plain)
        _restore_rng(rng,dev)
        started=time.perf_counter()
        captured,a=source.capture_source_coupling(found.wrapped.core.kuramoto,
          lambda:late_rollout(found.wrapped,gamma,total_steps=STEPS,live_tail_steps=0))
        rollout_seconds=time.perf_counter()-started
        source._trace_check(captured); source._validate_actual_A(a)
        q=source._actual_q(captured); qcc=sw136._qcc_labels(captured)
        if not torch.equal(q,plain_q) or not np.array_equal(qcc,plain_labels): raise AssertionError("captured A changed native Q/QCC")
        trace_hashes={}
        for name in TRACE_FIELDS:
            if not torch.equal(plain[name],captured[name]): raise AssertionError(f"capture changed source trace {name}")
            trace_hashes[name]=source._tensor_sha(captured[name])
        q_np=q[0].detach().cpu().numpy().astype(np.float64,copy=True)
        a_np=a[0].detach().cpu().numpy().astype(np.float64,copy=True)
        labels=np.asarray(qcc[0],dtype=np.int32)
        solver_start=time.perf_counter()
        result=solver.foreground_induced_merge(q_np,a_np,gain,labels,grid_shape=(32,32))
        solver_seconds=time.perf_counter()-solver_start
        again=solver.foreground_induced_merge(q_np,a_np,gain,labels,grid_shape=(32,32))
        if result["merge_count"]!=again["merge_count"] or not np.array_equal(result["labels"],again["labels"]): raise AssertionError("solver is nondeterministic")
        output_labels=result["labels"]
        if not np.array_equal(output_labels>0,labels>0): raise AssertionError("candidate changed Source-T foreground membership")
        rows.append({"source_position":pos,"pool_row":int(pool_row),"image_id":int(image_id),
          "rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),"gamma_sha256":source._tensor_sha(gamma),
          "trace_sha256":trace_hashes,"q_sha256":source._tensor_sha(q),"actual_a_sha256":source._tensor_sha(a),
          "source_qcc_sha256":hashlib.sha256(np.asarray(qcc,dtype="<i8").tobytes()).hexdigest(),
          "trace_parity":True,"q_parity":True,"qcc_parity":True,"foreground_support_exact":True,
          "gain":gain,"merge_count":int(result["merge_count"]),"foreground_group_sizes":sorted([x["size"] for x in result["clusters"]],reverse=True),
          "foreground_fraction":float(np.mean(labels>0)),"rollout_seconds":float(rollout_seconds),"solver_seconds":float(solver_seconds),
          "deterministic_solver_replay":True})
        peak=max(peak,int(torch.cuda.max_memory_allocated(dev)))
        del image,gamma,plain,captured,plain_q,q,plain_labels,qcc,a
    after=source._source_state_digest(found)
    if before!=after: raise AssertionError("frozen source state changed during preflight")
    report={"experiment":"SW0165_foreground_partition_replication","status":"source8_preflight_complete","seed":seed,
      "implementation_fingerprint":implementation_fingerprint(),"source":{**source.source_report_fields(prepared,before,after),"seed":seed,"source_model_sha256":prepared["source_sha"]},
      "source_operator":{"arm":"T","steps":STEPS,"settle":SETTLE,"live_tail":0,"optimizer_updates":0,
        "backend":found.wrapped.core.kuramoto_backend,"gain_exp_log_coupling_gain":gain,"actual_A_origin":"untouched prepare_coupling tuple[0]"},
      "images":rows,"cuda_peak_allocated_bytes":peak,"ground_truth_used":False,"optimizer_updates":0,
      "interpretation":"Eight TRAIN source parity/support-preservation solver feasibility only; no accuracy or learning claim"}
    if len(rows)!=8 or not all(x["foreground_support_exact"] and x["trace_parity"] and x["q_parity"] and x["qcc_parity"] for x in rows):
        raise AssertionError("SW0165 preflight incomplete")
    _atomic_json(output,report)
    return report

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    report=run_preflight(seed=args.seed,device=args.device,output=args.output)
    print(json.dumps({"status":report["status"],"image_count":len(report["images"]),"report":str(args.output)},allow_nan=False),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())

