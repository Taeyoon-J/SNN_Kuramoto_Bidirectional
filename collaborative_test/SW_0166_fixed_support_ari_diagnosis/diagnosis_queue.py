"""CPU-only owner-free waiter for the preregistered SW0166 diagnostic."""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if __name__ == "__main__": os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
if str(ROOT/"collaborative_test") not in sys.path: sys.path.insert(0,str(ROOT/"collaborative_test"))
try: import fcntl
except ImportError: fcntl=None
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
from collaborative_test.SW_0165_foreground_partition_replication import stage_queue as q165
from collaborative_test.SW_0166_fixed_support_ari_diagnosis import diagnose

RUNTIME=HERE/"results_archive"/"diagnosis_queue_20261014"
POLL_SECONDS=10.
WATCHDOG_SECONDS=86400.
ACTIVE={"waiting_for_idle_gpu","source8_preflight_pending","waiting_stable_gpu","waiting_for_exclusive_gpu","running_source8_preflight","prediction_pending","running_predict","validating_predictions_before_gt","scoring_gt_last","running"}


def _atomic_json(path, obj):
    path=Path(path); tmp=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def _module_live(state, seed):
    if state.get("seed") != seed:
        return False
    pid=state.get("supervisor_pid")
    if type(pid) is not int or pid<=0: return False
    try: rec=owner._proc_record(pid)
    except Exception: return False
    argv=rec.get("argv",[]) if rec else []
    module=f"collaborative_test.SW_0165_foreground_partition_replication.stage_queue"
    if not rec or rec.get("state")=="Z" or module not in argv:
        return False
    positions=[i for i,value in enumerate(argv) if value=="--seed"]
    return len(positions)==1 and positions[0]+1<len(argv) and argv[positions[0]+1]==str(seed)


def completed_score_gate(seed, *, wait=True, deadline=None, sleep=time.sleep, state_reader=None):
    """Wait only for a live registered 0165 owner; dead/malformed states fail closed."""
    path=q165.RUNTIME_BASE/f"stage_queue_seed{seed}_20261014"/"state.json"
    state_reader=state_reader or (lambda p: json.loads(Path(p).read_text(encoding="utf-8")))
    end=time.monotonic()+(WATCHDOG_SECONDS if deadline is None else deadline)
    while True:
        if not path.is_file(): raise RuntimeError(f"SW0165 seed{seed} state is absent")
        state=state_reader(path)
        status=state.get("status")
        if state.get("seed") != seed:
            raise ValueError(f"SW0165 seed{seed} state is bound to a different seed")
        if status=="complete":
            if state.get("seed")!=seed: raise ValueError("SW0165 terminal seed mismatch")
            return state
        if status not in ACTIVE or not _module_live(state,seed):
            raise RuntimeError(f"SW0165 seed{seed} is terminal or has no live registered owner: {status}")
        if not wait: raise RuntimeError(f"SW0165 seed{seed} is still live in {status}")
        if time.monotonic()>=end: raise TimeoutError(f"SW0165 seed{seed} did not finish before diagnostic watchdog")
        sleep(POLL_SECONDS)


class DiagnosisQueue:
    def __init__(self, *, runtime=RUNTIME, sleep=time.sleep, state_reader=None):
        if fcntl is None: raise RuntimeError("SW0166 CPU queue requires POSIX flock")
        self.runtime=Path(runtime); self.state_path=self.runtime/"state.json"; self.lock_path=self.runtime/"queue.lock"
        self.sleep=sleep; self.state_reader=state_reader
        self.runtime.mkdir(parents=True,exist_ok=True)
        self.lock=self.lock_path.open("a+")
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close(); raise RuntimeError("SW0166 diagnostic queue already owned")
        if self.state_path.exists() or (self.runtime/"diagnosis.json").exists():
            self._unlock(); raise FileExistsError("preserving existing SW0166 diagnosis artifacts")
        self._write("waiting_for_sw0165_seeds",supervisor_pid=os.getpid(),pid=os.getpid(),
                    experiment="SW0166_fixed_support_ari_diagnosis",seed_scope=[0,1,2],
                    queue_fingerprint=diagnose.implementation_fingerprint(),
                    ground_truth_used_for_prediction=False)

    def _unlock(self):
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN)
        finally: self.lock.close()

    def _write(self,status,**fields):
        current=json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        current.update(status=status,updated_utc=time.time(),**fields)
        tmp=self.state_path.with_name(self.state_path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(current,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,self.state_path)

    def run(self):
        try:
            for seed in (0,2):
                self._write("waiting_for_sw0165_seeds",waiting_seed=seed)
                completed_score_gate(seed,wait=True,sleep=self.sleep,state_reader=self.state_reader)
            if diagnose.implementation_fingerprint() != json.loads(self.state_path.read_text()).get("queue_fingerprint"):
                raise RuntimeError("SW0166 diagnostic code changed after queue registration")
            self._write("validating_predictions_and_scores_gt_last")
            report=diagnose.run_diagnostic(output=self.runtime/"diagnosis.json")
            self._write("complete",diagnosis_sha256=diagnose.sha256_file(self.runtime/"diagnosis.json"),
                        deterministic_feasibility=report["fixed320_deterministic_feasibility"],
                        ground_truth_opened_after_all_predictions_and_scores_validated=True)
            return 0
        except BaseException as exc:
            self._write("failed",error=repr(exc),finished_utc=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    if a.dry_run:
        print(json.dumps({"experiment":"SW0166_fixed_support_ari_diagnosis","status":"planned","seeds":[0,1,2],"count":320,"gpu_required":False,"ground_truth_used_for_prediction":False},allow_nan=False)); return 0
    return DiagnosisQueue().run()
if __name__=="__main__": raise SystemExit(main())
