"""CPU-only fixed320 score coordinator; GT is opened after all predictions validate."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path
if __name__=="__main__": os.environ["CUDA_VISIBLE_DEVICES"]=""
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test")]
try: import fcntl
except ImportError: fcntl=None
_MISSING=object(); _saved=os.environ.get("CUDA_VISIBLE_DEVICES",_MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved is _MISSING: os.environ.pop("CUDA_VISIBLE_DEVICES",None)
else: os.environ["CUDA_VISIBLE_DEVICES"]=_saved
from collaborative_test.SW_0163_relation_context_assignment import predict_evaluate as evaluation, prediction_queue

RUNTIME=HERE/"results_archive"/"score_queue_20261014"
STATE,LOCK,LOG=RUNTIME/"state.json",RUNTIME/"queue.lock",RUNTIME/"score.log"
POLL_SECONDS=5.

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def implementation_fingerprint():
    paths=(HERE/"score_queue.py",HERE/"prediction_queue.py",HERE/"predict_evaluate.py",HERE/"train.py",HERE/"model.py",HERE/"context_cache.py",HERE/"preflight.py",HERE/"protocol.json",HERE/"DESIGN_KO.md",Path(owner.__file__),Path(owner.OWNER_GUARD))
    if any(not p.is_file() for p in paths): raise FileNotFoundError("SW0163 score dependency missing")
    if sha256_file(owner.OWNER_GUARD)!=owner.OWNER_GUARD_SHA256: raise ValueError("SW0148 owner guard changed")
    return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths}

def _prediction_proc_live(pid):
    module="collaborative_test.SW_0163_relation_context_assignment.prediction_queue"
    if not isinstance(pid,int) or pid<=1 or not sys.platform.startswith("linux"): return False
    try:
        args=[x.decode(errors="replace") for x in (Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0") if x]
        return "-m" in args and module in args
    except OSError: return False

def prediction_dependency(*,state_path=prediction_queue.STATE,process_live=_prediction_proc_live):
    path=Path(state_path)
    if not path.is_file(): raise ValueError("SW0163 prediction queue state missing")
    state=json.loads(path.read_text(encoding="utf-8")); status=state.get("status")
    if status=="complete":
        if state.get("queue_fingerprint")!=prediction_queue.implementation_fingerprint(): raise ValueError("prediction queue fingerprint changed")
        expected=set((*evaluation.ARMS,"source_T_control"))
        if set(state.get("prediction_hashes",{}))!=expected: raise ValueError("prediction completion state has wrong arm set")
        return {"state":state,"state_sha256":sha256_file(path)}
    if status in ("failed","cancelled"):
        raise ValueError(f"prediction prerequisite terminal: {status}")
    if not process_live(state.get("supervisor_pid")):
        raise ValueError("prediction queue is nonterminal but registered supervisor is not live")
    return None

def _atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

class ScoreQueue:
    def __init__(self,*,runtime=RUNTIME,poll_seconds=POLL_SECONDS,dependency=prediction_dependency):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        if fcntl is None: raise RuntimeError("SW0163 score queue requires Linux flock")
        self.runtime=Path(runtime); self.runtime.mkdir(parents=True,exist_ok=True)
        self.state_path=self.runtime/"state.json"; self.lock_path=self.runtime/"queue.lock"; self.log_path=self.runtime/"score.log"
        self.lock=self.lock_path.open("a+")
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: self.lock.close(); raise RuntimeError("another score supervisor owns this queue")
        if self.state_path.exists() or self.log_path.exists(): self._unlock(); raise FileExistsError("preserving existing SW0163 score attempt")
        self.stop=threading.Event(); self.poll_seconds=poll_seconds; self.dependency=dependency; self.fp=implementation_fingerprint(); self._update("waiting_for_predictions")
    def _unlock(self):
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN)
        finally: self.lock.close()
    def _update(self,status,**fields):
        _atomic(self.state_path,{"experiment":"SW0163_relation_context_assignment_score_queue","status":status,"updated":time.time(),"supervisor_pid":os.getpid(),"queue_fingerprint":self.fp,**fields})
    def run(self):
        try:
            while not self.stop.is_set():
                dep=self.dependency()
                if dep is not None: break
                self._update("waiting_for_predictions")
                self.stop.wait(self.poll_seconds)
            else: return 2
            # Cheap file-presence gate avoids repeatedly hashing/decompressing while the producer runs.
            if not all((evaluation._out(a)/"predictions.npz").is_file() for a in (*evaluation.ARMS,"source_T_control")):
                raise ValueError("prediction owner published complete state without all immutable outputs")
            prepared=evaluation.prepare_prediction_cpu()
            frozen={a:evaluation.validate_prediction(a,prepared=prepared) for a in (*evaluation.ARMS,"source_T_control")}
            actual={a:{"prediction_sha256":v["prediction_sha256"],"protocol_sha256":v["protocol_sha256"]} for a,v in frozen.items()}
            if dep["state"].get("prediction_hashes")!=actual: raise ValueError("prediction queue state hashes differ from validated outputs")
            self._update("scoring_gt_last",prediction_state_sha256=dep["state_sha256"],prediction_hashes={a:{"prediction_sha256":v["prediction_sha256"],"protocol_sha256":v["protocol_sha256"]} for a,v in frozen.items()})
            if self.log_path.exists(): raise FileExistsError("preserving existing score log")
            report=evaluation.score_all(prepared=prepared)
            validated=evaluation.validate_score(prepared=prepared)
            score_sha=sha256_file(evaluation.SCORE_PATH)
            self._update("complete",score_sha256=score_sha,score_status=validated["status"],finished=time.time(),prediction_state_sha256=dep["state_sha256"])
            return 0
        except BaseException as exc:
            self._update("failed",error=repr(exc),finished=time.time(),artifacts_preserved=True); return 1
        finally:
            self.stop.set(); self._unlock()
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args(argv)
    if a.dry_run: print(json.dumps({"runtime":str(RUNTIME),"metric_scope":"fixed320 seed1 GT-last","cuda_visible_devices_parent":os.environ.get("CUDA_VISIBLE_DEVICES","<unset>")},allow_nan=False)); return 0
    return ScoreQueue().run()
if __name__=="__main__": raise SystemExit(main())
