"""Owner-reserved lineage-recovery prediction after all four SW0163 heads finish."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, threading, time
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
from collaborative_test.SW_0163_source_lineage_recovery import predict_evaluate as evaluation
from collaborative_test.SW_0163_relation_context_assignment import train, training_queue, recovery_training_queue

RUNTIME=evaluation.RECOVERY_ROOT/"prediction_queue"
STATE,LOCK,LOG=RUNTIME/"state.json",RUNTIME/"queue.lock",RUNTIME/"attempt01.log"
MODULE="collaborative_test.SW_0163_source_lineage_recovery.prediction_queue"
DEVICE="cuda:0"; MAX_MEMORY_MIB=512; POLL_SECONDS=5.; SAMPLE_SECONDS=2.; WATCHDOG_SECONDS=8*60*60

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def implementation_fingerprint():
    guard=Path(owner.OWNER_GUARD)
    if not guard.is_file() or sha256_file(guard)!=owner.OWNER_GUARD_SHA256: raise ValueError("SW0148 owner guard changed")
    files=[HERE/"prediction_queue.py",HERE/"predict_evaluate.py",HERE/"recovery_protocol.json",
           Path(training_queue.__file__),Path(recovery_training_queue.__file__),Path(train.__file__),
           Path(evaluation.contract.__file__),Path(owner.__file__),guard]
    if any(not p.is_file() for p in files): raise FileNotFoundError("SW0163 recovery prediction dependency missing")
    result={}
    result.update(evaluation.implementation_fingerprint())
    result.update(training_queue.implementation_fingerprint())
    result[recovery_training_queue.__file__]=sha256_file(recovery_training_queue.__file__)
    result.update({p.relative_to(ROOT).as_posix():sha256_file(p) for p in files})
    return result

def command():
    return [sys.executable,"-m","collaborative_test.SW_0163_source_lineage_recovery.predict_evaluate","--stage","predict","--device",DEVICE]

def _proc_live(pid):
    if not isinstance(pid,int) or pid<=1 or not sys.platform.startswith("linux"): return False
    try:
        args=[x.decode(errors="replace") for x in (Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0") if x]
        return "-m" in args and MODULE in args
    except OSError: return False

def training_dependency(*,state_path=recovery_training_queue.RUNTIME/"state.json", process_live=recovery_training_queue._pid_live):
    path=Path(state_path)
    if not path.is_file(): raise ValueError("SW0163 recovered training state missing")
    state=json.loads(path.read_text(encoding="utf-8")); status=state.get("status")
    if status=="complete" and state.get("recovery_status")=="complete_recovered":
        recovery_training_queue.verify_recovery_fingerprint(state.get("recovery_fingerprint"))
        if state.get("queue_fingerprint")!=training_queue.implementation_fingerprint(): raise ValueError("recovered training state differs from sealed base queue")
        if set(state.get("results",{}))!=set(train.ARMS) or not all(state["results"].values()): raise ValueError("recovered training completion does not include all arms")
        if set(state.get("final_arms",{}))!=set(train.ARMS): raise ValueError("recovered training arm inventory is incomplete")
        return {"state":state,"state_sha256":sha256_file(path),"recovery_fingerprint":state["recovery_fingerprint"]}
    if status in ("failed_recovery","failed_training","failed_preflight_dependency"):
        raise ValueError(f"SW0163 recovery training prerequisite terminal: {status}")
    if not process_live(state.get("supervisor_pid")):
        raise ValueError("SW0163 recovery training is nonterminal without its registered live supervisor")
    return None

def _atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def _gpu_ok(gpu,expected):
    actual,memory,owners=owner.gpu_guard._gpu_info(gpu)
    return actual==expected and not owners and int(memory)<=MAX_MEMORY_MIB,{"uuid":actual,"memory_mib":int(memory),"owners":list(owners)}

class PredictionQueue:
    def __init__(self,*,runtime=RUNTIME,poll_seconds=POLL_SECONDS,dependency=training_dependency):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        if fcntl is None: raise RuntimeError("SW0163 recovery prediction queue requires Linux flock")
        self.runtime=Path(runtime); self.runtime.mkdir(parents=True,exist_ok=True)
        self.state_path=self.runtime/"state.json"; self.lock_path=self.runtime/"queue.lock"; self.log_path=self.runtime/"attempt01.log"
        self.lock=self.lock_path.open("a+")
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: self.lock.close(); raise RuntimeError("another prediction supervisor owns this queue")
        if self.state_path.exists() or self.log_path.exists(): self._unlock(); raise FileExistsError("preserving existing SW0163 recovery prediction queue attempt")
        self.stop=threading.Event(); self.poll_seconds=poll_seconds; self.dependency=dependency; self.child=None; self.lease=None
        self.fp=implementation_fingerprint(); self._update("waiting_for_training")
    def _unlock(self):
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN)
        finally: self.lock.close()
    def _update(self,status,**fields):
        record={"experiment":"SW0163_source_lineage_recovery_prediction_queue","status":status,"updated":time.time(),"supervisor_pid":os.getpid(),"queue_fingerprint":self.fp,**fields}
        _atomic(self.state_path,record)
    def _verify(self):
        if implementation_fingerprint()!=self.fp: raise RuntimeError("prediction code changed after registration")
    def run(self):
        try:
            while not self.stop.is_set():
                self._verify(); dep=self.dependency()
                if dep is not None: break
                self._update("waiting_for_training",predecessor_pid=json.loads(training_queue.STATE.read_text()).get("supervisor_pid"))
                self.stop.wait(self.poll_seconds)
            else: return 2
            prepared=evaluation.prepare_prediction_cpu()
            # Validate every final endpoint before acquiring a GPU.
            if self.log_path.exists(): raise FileExistsError("preserving existing prediction child log")
            self._update("waiting_for_idle_gpu",training_state_sha256=dep["state_sha256"],training_manifests=prepared["training_manifest_sha256"])
            while not self.stop.is_set():
                gpu,gpu_uuid,memory,lease=owner._acquire(self.stop); self.lease=lease; proc=None
                try:
                    stable=owner._stable_idle(gpu,gpu_uuid,self.stop,lambda elapsed,used:self._update("waiting_stable_gpu",gpu=gpu,gpu_uuid=gpu_uuid,stable_seconds=elapsed,memory_used_mib=used))
                    if not stable:
                        self._release()
                        if self.stop.is_set(): return 2
                        self._update("waiting_for_idle_gpu",lease_released=True); continue
                    self._verify()
                    current=evaluation.prepare_prediction_cpu()
                    if current["training_manifest_sha256"]!=prepared["training_manifest_sha256"]: raise ValueError("training endpoints changed while waiting for GPU")
                    current_dep=self.dependency()
                    if (current_dep is None or current_dep["state_sha256"]!=dep["state_sha256"]
                            or current_dep.get("recovery_fingerprint")!=dep.get("recovery_fingerprint")):
                        raise ValueError("recovered training completion receipt changed before launch")
                    ok,sample=_gpu_ok(gpu,gpu_uuid)
                    if not ok:
                        self._update("waiting_for_idle_gpu",final_gpu_sample=sample,lease_released=True); self._release(); continue
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
                    with self.log_path.open("x",encoding="utf-8") as log:
                        proc=subprocess.Popen(command(),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                        self.child=proc; self._update("running",child_pid=proc.pid,child_pgid=proc.pid,child_argv=command(),gpu=gpu,gpu_uuid=gpu_uuid,log=str(self.log_path),inherited_lease_fd=True)
                        began=time.monotonic()
                        while proc.poll() is None and not self.stop.is_set():
                            if time.monotonic()-began>WATCHDOG_SECONDS: owner._stop_owned(proc); raise TimeoutError("prediction watchdog")
                            time.sleep(SAMPLE_SECONDS)
                            if proc.poll() is None:
                                try: foreign=owner.gpu_guard._foreign_owners(owner.gpu_guard._gpu_info(gpu)[2],owner.gpu_guard._process_tree(proc.pid))
                                except Exception: continue
                                if foreign: owner._stop_owned(proc); raise RuntimeError(f"foreign GPU owner detected: {foreign}")
                        if self.stop.is_set() and proc.poll() is None: owner._stop_owned(proc); raise RuntimeError("prediction cancelled; owned child stopped")
                        rc=proc.wait()
                    if rc!=0: raise RuntimeError(f"prediction child exited {rc}; preserving rows/log")
                    final=evaluation.prepare_prediction_cpu()
                    preds={a:evaluation.validate_prediction(a,prepared=final) for a in (*train.ARMS,"source_T_control")}
                    self._update("complete",finished=time.time(),prediction_hashes={a:{"prediction_sha256":p["prediction_sha256"],"protocol_sha256":p["protocol_sha256"]} for a,p in preds.items()},training_state_sha256=dep["state_sha256"],training_recovery_fingerprint=dep["recovery_fingerprint"])
                    return 0
                finally:
                    try:
                        if proc is not None and proc.poll() is None: owner._stop_owned(proc)
                    finally:
                        self.child=None; self._release()
            return 2
        except BaseException as exc:
            self._update("failed",error=repr(exc),finished=time.time(),artifacts_preserved=True); return 1
        finally:
            self.stop.set()
            if self.child is not None and self.child.poll() is None: owner._stop_owned(self.child)
            self._release(); self._unlock()
    def _release(self):
        lease,self.lease=self.lease,None
        if lease is not None:
            try: fcntl.flock(lease.fileno(),fcntl.LOCK_UN)
            finally: lease.close()

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args(argv)
    if a.dry_run: print(json.dumps({"command":command(),"runtime":str(RUNTIME),"arms":list(train.ARMS),"gpu_candidates":[0,1,2,3],"max_memory_mib":MAX_MEMORY_MIB},allow_nan=False)); return 0
    return PredictionQueue().run()
if __name__=="__main__": raise SystemExit(main())
