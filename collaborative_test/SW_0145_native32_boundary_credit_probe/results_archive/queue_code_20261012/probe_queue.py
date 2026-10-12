"""Owner-aware single-job launcher for the SW0145 disposable probe."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, threading, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
try: import fcntl
except ImportError: fcntl = None
HERE=Path(__file__).resolve().parent; ACMLAB=HERE.parents[1]
ROOT=Path(os.environ.get("PATCH_V2_REPO", ACMLAB/"patch_v2_sw")).resolve()
EXP=ROOT/"collaborative_test"/"SW_0145_native32_boundary_credit_probe"
OUT=EXP/"results_archive"/"source0_boundary_credit_probe.json"
RUNTIME=EXP/"results_archive"/"queue_runtime_20261012"; STATE=RUNTIME/"queue_state.json"; LOG=RUNTIME/"probe.log"
SERVER_HELPER=ROOT/"collaborative_test"/"SW_0143_native32_selective_geodesic"/"results_archive"/"recovery_code_v3_20261012"/"recover_queue.py"
HELPER=SERVER_HELPER if SERVER_HELPER.is_file() else HERE.parent/"SW0143_recovery_v2_20261012"/"recover_queue.py"
HELPER_SHA="b32c1aa5ce8417359aaab40bb9606e6fdee2c8d8314281f365616c24805dbe00"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def load_helper():
 if not HELPER.is_file() or sha(HELPER)!=HELPER_SHA: raise RuntimeError("reviewed stable-idle helper missing/changed")
 spec=importlib.util.spec_from_file_location("sw0143_stable_idle_private",HELPER); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m
helper=load_helper(); frozen=helper.frozen
def task_plan(): return [{"task_id":"source0_boundary_credit_probe","seed":0,"stage":"disposable_probe","output":str(OUT)}]
def validate_result(path=OUT,run_sha=None):
 try:
  r=json.loads(Path(path).read_text(encoding="utf-8")); run_sha=run_sha or sha(EXP/"run.py")
  return r.get("status")=="disposable_probe_complete" and r.get("experiment")=="SW0145_native32_boundary_credit_probe" and r.get("seed")==0 and r.get("resource_only") is True and r.get("training_admission") is False and r.get("ground_truth_used") is False and r.get("optimizer_updates")==1 and r.get("implementation_sha256")==run_sha and len(r.get("gradient_comparison_after_disposable_step",{}).get("per_logical_batch_gradient_norms",[]))==4
 except (OSError,ValueError,TypeError): return False
def atomic_json(path,data):
 tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
 with tmp.open("x",encoding="utf-8",newline="\n") as f: json.dump(data,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
class ProbeQueue:
 def __init__(self,sample=None,sleep=None,monotonic=None):
  if fcntl is None: raise RuntimeError("GPU queue requires Linux flock")
  if RUNTIME.exists() or OUT.exists() or LOG.exists(): raise FileExistsError("preserve existing SW0145 outputs")
  lockpath=EXP/"results_archive"/"probe_queue.lock"; lockpath.parent.mkdir(parents=True,exist_ok=True); self.lock=lockpath.open("x+")
  try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException: self.lock.close(); raise RuntimeError("another SW0145 queue owns lock")
  RUNTIME.mkdir(parents=True); self.stop=threading.Event(); self.sample=sample or (lambda g:frozen.owner._nvidia_gpu_info(g)); self.sleep=sleep or time.sleep; self.monotonic=monotonic or time.monotonic
  self.state={"experiment":"SW0145_native32_boundary_credit_probe","status":"queued","supervisor_pid":os.getpid(),"run_sha256":sha(EXP/"run.py"),"protocol_sha256":sha(EXP/"protocol.json"),"queue_sha256":sha(Path(__file__)),"owner_helper_sha256":sha(HELPER),"owner_dispatcher_sha256":sha(Path(frozen.owner.__file__)),"output":str(OUT),"tasks":{"source0_boundary_credit_probe":{"status":"queued","task":task_plan()[0]}}}; self._save()
 def _save(self): atomic_json(STATE,self.state)
 def update(self,**kw): self.state["tasks"]["source0_boundary_credit_probe"].update(kw); self._save()
 def reserve(self):
  while not self.stop.is_set():
   for gpu in range(4):
    lp=frozen.GPU_LEASES/f"gpu{gpu}.lock"; lp.parent.mkdir(parents=True,exist_ok=True); lease=lp.open("a+")
    try: fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: lease.close(); continue
    try:
     sample=lambda:self.sample(gpu); first=sample()
     if not frozen.gpu_candidate(first[1],first[2]): fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); continue
     stable,detail=helper.stable_idle_window(sample,stop_event=self.stop,monotonic=self.monotonic,sleep=self.sleep,on_sample=lambda elapsed,uid,mem:self.update(status="waiting_for_stable_idle_window",physical_gpu=gpu,gpu_uuid=uid,stable_seconds=elapsed,memory_mib=mem))
     if stable:return gpu,lease,detail["gpu_uuid"]
     fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close()
    except BaseException: fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); raise
   self.state["status"]="waiting_for_exclusive_gpu"; self._save(); self.sleep(5)
  raise RuntimeError("queue cancelled before GPU reservation")
 def run(self):
  gpu=lease=proc=None
  try:
   if OUT.exists(): raise FileExistsError(f"preserve existing result: {OUT}")
   self.state["status"]="running"; self._save(); gpu,lease,uid=self.reserve(); now,mem,owners=self.sample(gpu)
   if now!=uid or not frozen.gpu_candidate(mem,owners): raise RuntimeError("GPU changed after idle window")
   triton=Path("/tmp")/f"kevinswk_sw0145_{os.getpid()}_{int(time.time())}"; triton.mkdir(exist_ok=False)
   env=os.environ.copy(); env.update(PATCH_V2_REPO=str(ROOT),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",TRITON_CACHE_DIR=str(triton))
   argv=[sys.executable,"-m","collaborative_test.SW_0145_native32_boundary_credit_probe.run","--device","cuda:0","--output",str(OUT)]
   with LOG.open("x",encoding="utf-8",newline="\n") as log:
    self.update(status="launching",physical_gpu=gpu,gpu_uuid=uid,argv=argv,log_path=str(LOG),cuda_visible_devices=str(gpu))
    proc=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),)); self.update(status="running",child_pid=proc.pid,started=time.time())
    while proc.poll() is None:
     self.sleep(5)
     if proc.poll() is not None: break
     _,_,owners=self.sample(gpu); foreign=frozen.owner.foreign_owner_pids(owners,frozen.owner._process_tree(proc.pid))
     if foreign: frozen.SelectiveGeodesicQueue._terminate_owned(proc); self.update(status="interrupted_foreign_gpu_owner",foreign_owner_pids=foreign,returncode=proc.returncode); raise RuntimeError("foreign GPU owner appeared")
   valid=proc.returncode==0 and validate_result(run_sha=self.state["run_sha256"]); self.state["status"]="complete" if valid else "failed"; self.update(status="passed" if valid else "failed",returncode=proc.returncode,artifact_valid=bool(valid),finished=time.time()); return 0 if valid else 1
  except BaseException as exc:
   if proc is not None and proc.poll() is None:frozen.SelectiveGeodesicQueue._terminate_owned(proc)
   self.state["status"]="failed"; self.update(status="failed",error=f"{type(exc).__name__}: {exc}",finished=time.time()); return 1
  finally:
   if lease is not None: fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close()
   if not self.lock.closed: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN); self.lock.close()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
 if a.dry_run: print(json.dumps({"status":"dry_run","tasks":task_plan(),"quiet_window_seconds":60,"max_parallel":1},indent=2)); return 0
 return ProbeQueue().run()
if __name__=="__main__": raise SystemExit(main())
