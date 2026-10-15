"""CPU-only recovery scorer for SW0168 v2 prediction supervisors."""
from __future__ import annotations
import json, os, signal, sys, threading, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from collaborative_test.SW_0168_end_to_end_connectivity import predict_evaluate as evaluate, train
from collaborative_test.SW_0168_end_to_end_connectivity import recover_completed_training as recovery

HERE=Path(__file__).resolve().parent
RUNTIME=HERE/"results_archive"/"score_queue_v4_20261015"
PREDICTION_ROOT=HERE/"results_archive"/"prediction_queue_v3_20261015"
EXPERIMENT="SW0168_score_queue_v4"
PRED_MODULE="collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue_v3"
PRED_EXPERIMENT="SW0168_prediction_queue_v3"
_stop=threading.Event()


def _write(path,payload,*,create=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(payload,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if create:
            if path.exists(): raise FileExistsError(path)
            os.link(tmp,path)
        else: os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def _module_live(owner,pid,arm):
    if type(pid) is not int or pid<=0: return False
    record=owner._proc_record(pid)
    if not record or record.get("state")=="Z": return False
    argv=record.get("argv",[])
    if "-m" not in argv: return False
    i=argv.index("-m")
    return (i+1<len(argv) and argv[i+1]==PRED_MODULE and argv.count("--arm")==1
            and argv.index("--arm")+1<len(argv) and argv[argv.index("--arm")+1]==arm)


def _prediction_state(arm,owner,recovery_sha):
    path=PREDICTION_ROOT/f"{arm}_state.json"
    if not path.is_file(): raise FileNotFoundError(f"SW0168 v2 prediction state missing: {arm}")
    state=json.loads(path.read_text(encoding="utf-8"))
    if (state.get("experiment")!=PRED_EXPERIMENT or state.get("arm")!=arm
            or state.get("recovery_receipt_sha256")!=recovery_sha):
        raise ValueError(f"SW0168 v2 prediction state identity mismatch: {arm}")
    if state.get("status")=="complete":
        out=Path(state.get("output",""))
        if (not out.is_dir() or recovery._sha(out/"predictions.npz")!=state.get("prediction_sha256")
                or recovery._sha(out/"protocol.json")!=state.get("protocol_sha256")):
            raise ValueError(f"SW0168 v2 prediction files changed: {arm}")
        return state
    active={"waiting_for_training","waiting_for_stable_gpu","waiting_for_exclusive_gpu",
            "running_prediction","yielding_foreign_gpu"}
    if state.get("status") in active and _module_live(owner,state.get("supervisor_pid"),arm):
        return False
    raise RuntimeError(f"SW0168 v2 prediction failed/orphaned: {arm}/{state.get('status')}")


def run(*,timeout_seconds=8*3600):
    import fcntl
    signal.signal(signal.SIGTERM,lambda *_:_stop.set())
    signal.signal(signal.SIGINT,lambda *_:_stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("SW0168 v2 score coordinator must remain CUDA-hidden")
    receipt=recovery.validate_receipt(deep=True)
    recovery_sha=recovery._sha(recovery.RECEIPT)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    state_path=RUNTIME/"state.json"; lock_path=RUNTIME/"queue.lock"
    if state_path.exists() or (RUNTIME/"score.log").exists():
        raise FileExistsError("preserving existing SW0168 v2 score state/log")
    lock=lock_path.open("a+"); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        owner=__import__("importlib").import_module("collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("owner import changed CPU score visibility")
        state={"experiment":EXPERIMENT,"status":"waiting_for_both_predictions",
               "supervisor_pid":os.getpid(),"recovery_receipt_sha256":recovery_sha,"updated":time.time()}
        _write(state_path,state,create=True)
        deadline=time.monotonic()+timeout_seconds
        while not _stop.is_set():
            if time.monotonic()>deadline: raise TimeoutError("SW0168 v2 score wait timed out")
            done={arm:_prediction_state(arm,owner,recovery_sha) for arm in train.ARMS}
            if all(done.values()): break
            state.update({"updated":time.time(),"prediction_status":{
                arm:("complete" if done[arm] else "waiting") for arm in train.ARMS}})
            _write(state_path,state)
            time.sleep(5)
        else: raise RuntimeError("SW0168 v2 score stopped before predictions completed")
        prepared=evaluate.prepare_prediction_inputs_cpu()
        validated={arm:evaluate.validate_prediction(arm,prepared=prepared) for arm in train.ARMS}
        for arm in train.ARMS:
            if (validated[arm]["prediction_sha256"]!=done[arm]["prediction_sha256"]
                    or validated[arm]["protocol_sha256"]!=done[arm]["protocol_sha256"]):
                raise ValueError("SW0168 v2 prediction changed before GT scoring")
        state.update({"status":"scoring_gt_last","updated":time.time(),
            "prediction_sha256":{arm:validated[arm]["prediction_sha256"] for arm in train.ARMS},
            "ground_truth_used_for_prediction":False})
        _write(state_path,state)
        report=evaluate.score_all(prepared=prepared)
        score_sha=recovery._sha(evaluate.SCORE_PATH)
        final={**state,"status":"complete","score_sha256":score_sha,
            "score_path":str(evaluate.SCORE_PATH),
            "implementation_fingerprint":evaluate.implementation_fingerprint(),
            "ground_truth_used_for_scoring":True,"gate":report["gate"],"finished":time.time()}
        _write(state_path,final)
        return final
    except BaseException as exc:
        current=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        current.update({"status":"failed","error":repr(exc),"finished":time.time()})
        if state_path.exists(): _write(state_path,current)
        else: _write(state_path,current,create=True)
        raise
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--worker",action="store_true")
    if parser.parse_args().worker: raise SystemExit("SW0168 v2 score has no GPU worker mode")
    run()

if __name__=="__main__": main()
