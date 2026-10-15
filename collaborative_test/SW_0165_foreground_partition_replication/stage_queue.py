"""Owner-safe per-seed SW0165 preflight, prediction, and GT-last score supervisor."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
if __name__ == "__main__": os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
for p in (ROOT,ROOT/"collaborative_test"):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
try: import fcntl
except ImportError: fcntl=None
_MISSING=object(); _vis=os.environ.get("CUDA_VISIBLE_DEVICES",_MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _vis is _MISSING: os.environ.pop("CUDA_VISIBLE_DEVICES",None)
else: os.environ["CUDA_VISIBLE_DEVICES"]=_vis
from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as evaluation,preflight
RUNTIME_BASE=HERE/"results_archive"
DEVICE="cuda:0"; MAX_MEMORY_MIB=512; POLL_SECONDS=5.; WATCHDOG_SECONDS=7200.
ACTIVE_PREFLIGHT={"waiting_for_idle_gpu","waiting_stable_gpu","waiting_for_exclusive_gpu","waiting_gpu_probe_error","running"}
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def implementation_fingerprint():
 guard=Path(owner.OWNER_GUARD)
 if sha256_file(guard)!=owner.OWNER_GUARD_SHA256: raise ValueError("SW0148 owner guard checksum mismatch")
 paths=(HERE/"stage_queue.py",HERE/"predict_evaluate.py",HERE/"preflight.py",HERE/"foreground_signed.py",HERE/"protocol.json",HERE/"DESIGN_KO.md",Path(owner.__file__),guard)
 if any(not p.is_file() for p in paths): raise FileNotFoundError("SW0165 queue dependency missing")
 result={p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths}
 result.update(evaluation.implementation_fingerprint())
 return result
def _atomic_json(path,record):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
 with tmp.open("x",encoding="utf-8",newline="\n") as f: json.dump(record,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def _proc_live(pid,module):
 if type(pid) is not int or pid<=0: return False
 try: row=owner._proc_record(pid)
 except (OSError,RuntimeError,ValueError): return False
 return bool(row and row.get("state")!="Z" and module in row.get("argv",[]))
def command(stage,seed):
 if int(seed) not in evaluation.SEEDS: raise ValueError("unregistered SW0165 seed")
 if stage=="source8_preflight": return [sys.executable,"-m","collaborative_test.SW_0165_foreground_partition_replication.preflight","--seed",str(seed),"--device",DEVICE]
 if stage=="predict": return [sys.executable,"-m","collaborative_test.SW_0165_foreground_partition_replication.predict_evaluate","--stage","predict","--seed",str(seed),"--device",DEVICE]
 raise ValueError(f"unknown GPU stage {stage}")
class StageQueue:
 def __init__(self,*,seed,prepared_fn=None,owner_api=None):
  if int(seed) not in evaluation.SEEDS: raise ValueError("SW0165 accepts only seeds 0 and 2")
  self.seed=int(seed); self.runtime=RUNTIME_BASE/f"stage_queue_seed{self.seed}_20261014"; self.state_path=self.runtime/"state.json"; self.lock_path=self.runtime/"queue.lock"; self.log_path=self.runtime/"attempt01.log"
  os.environ["CUDA_VISIBLE_DEVICES"]=""; self.owner=owner if owner_api is None else owner_api
  if fcntl is None: raise RuntimeError("SW0165 stage queue requires Linux flock")
  self.runtime.mkdir(parents=True,exist_ok=True); self.lock=self.lock_path.open("a+")
  try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc: self.lock.close(); raise RuntimeError("SW0165 stage queue already owned") from exc
  self.state_lock=threading.RLock(); self.stop_event=threading.Event(); self.child=None; self.lease=None
  self.fingerprint=implementation_fingerprint(); self.eval_fingerprint=evaluation.implementation_fingerprint(); self.prepared_fn=prepared_fn or evaluation.prepare_prediction_inputs_cpu
  try:
   if self.state_path.exists() or self.log_path.exists(): raise FileExistsError("preserving existing SW0165 queue state/log")
   self._write("waiting_for_idle_gpu",supervisor_pid=os.getpid(),seed=self.seed,queue_fingerprint=self.fingerprint,evaluation_fingerprint=self.eval_fingerprint,
    task={"seed":self.seed,"arms":list(evaluation.ARMS),"count":evaluation.COUNT,"image_ids":[1320,1639],"steps":1024,"settle":512,"optimizer_updates":0,"ground_truth_used_for_prediction":False})
  except BaseException: self._unlock(); raise
 def _unlock(self):
  try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN)
  finally: self.lock.close()
 def _write(self,status,**fields):
  with self.state_lock:
   cur=json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
   cur.update(experiment="SW0165_foreground_partition_replication",status=status,updated=time.time(),**fields); _atomic_json(self.state_path,cur)
 def _verify(self):
  if implementation_fingerprint()!=self.fingerprint or evaluation.implementation_fingerprint()!=self.eval_fingerprint: raise RuntimeError("SW0165 code changed after queue registration")
 def _idle(self,gpu,gpu_uuid):
  try: actual,memory,owners=self.owner.gpu_guard._gpu_info(gpu)
  except Exception as exc: return False,{"reason":"query_error","detail":repr(exc)}
  info={"gpu_uuid":actual,"memory_used_mib":int(memory),"owner_pids":list(owners)}
  if actual!=gpu_uuid: info["reason"]="uuid_changed"
  elif owners: info["reason"]="foreign_owner"
  elif int(memory)>MAX_MEMORY_MIB: info["reason"]="memory_over_limit"
  else: info["reason"]="idle"
  return info["reason"]=="idle",info
 def _gpu_stage(self,stage,*,prepared=None):
  self._verify()
  while not self.stop_event.is_set():
   gpu,gpu_uuid,memory,lease=self.owner._acquire(self.stop_event); self.lease=lease
   self._write("waiting_stable_gpu",stage=stage,gpu=gpu,gpu_uuid=gpu_uuid,initial_memory_mib=memory)
   stable=self.owner._stable_idle(gpu,gpu_uuid,self.stop_event,lambda elapsed,used:self._write("waiting_stable_gpu",stage=stage,gpu=gpu,gpu_uuid=gpu_uuid,stable_seconds=elapsed,memory_used_mib=used))
   if not stable:
    fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); self.lease=None
    if self.stop_event.is_set(): return 1
    self._write("waiting_for_exclusive_gpu",stage=stage,reservation_released=True); continue
   # Revalidate prerequisites after idle stabilization, immediately before launch.
   if stage=="source8_preflight":
    preflight.prepare_preflight_cpu(self.seed)
   else:
    prefpath=preflight.output_for_seed(self.seed)
    if not preflight.validate_report(prefpath,expected_seed=self.seed,expected_fingerprint=preflight.implementation_fingerprint(),expected_source=evaluation._expected_source(self.seed)):
     raise ValueError("SW0165 source8 preflight invalid before prediction")
    prepared=evaluation.prepare_prediction_cpu(seed=self.seed)
   self._verify(); ok,detail=self._idle(gpu,gpu_uuid)
   if ok: break
   fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); self.lease=None
   self._write("waiting_for_exclusive_gpu",stage=stage,reservation_released=True,gpu_check=detail)
  if self.stop_event.is_set(): return 1
  env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env.setdefault("OMP_NUM_THREADS","1"); env.setdefault("MKL_NUM_THREADS","1")
  try:
   with self.log_path.open("a" if self.log_path.exists() else "x",encoding="utf-8",newline="\n") as log:
    self.child=subprocess.Popen(command(stage,self.seed),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(self.lease.fileno(),))
    self._write(f"running_{stage}",child_pid=self.child.pid,child_pgid=self.child.pid,gpu=gpu,gpu_uuid=gpu_uuid,started=time.time())
    deadline=time.monotonic()+WATCHDOG_SECONDS
    while self.child.poll() is None:
     if time.monotonic()>=deadline: self.owner._stop_owned(self.child); raise TimeoutError(f"SW0165 {stage} watchdog expired")
     try: uid,used,owners=self.owner.gpu_guard._gpu_info(gpu)
     except Exception: uid,used,owners=gpu_uuid,0,[]
     if uid!=gpu_uuid or any(pid!=self.child.pid for pid in owners): self.owner._stop_owned(self.child); raise RuntimeError("foreign GPU collision; stopped only owned child")
     self._write(f"running_{stage}",child_pid=self.child.pid,gpu=gpu,gpu_uuid=gpu_uuid,memory_used_mib=int(used)); time.sleep(1)
    rc=self.child.wait()
   if rc!=0: raise RuntimeError(f"{stage} child failed rc={rc}; see {self.log_path}")
   self._verify(); return 0
  except BaseException:
   if self.child is not None and self.child.poll() is None: self.owner._stop_owned(self.child)
   raise
  finally:
   if self.lease is not None:
    try: fcntl.flock(self.lease.fileno(),fcntl.LOCK_UN)
    finally: self.lease.close(); self.lease=None
 def run(self):
  try:
   self.prepared_fn(seed=self.seed) # CPU references/RGB/HDF identity only; no GT opened.
   self._verify()
   prefpath=preflight.output_for_seed(self.seed)
   if not prefpath.exists():
    self._write("source8_preflight_pending",ground_truth_used=False,seed=self.seed)
    if self._gpu_stage("source8_preflight"): return 1
   if not preflight.validate_report(prefpath,expected_seed=self.seed,expected_fingerprint=preflight.implementation_fingerprint(),expected_source=evaluation._expected_source(self.seed)):
    raise ValueError("SW0165 source8 preflight report invalid")
   prepared=evaluation.prepare_prediction_cpu(seed=self.seed)
   for arm in evaluation.ARMS:
    if evaluation._out(arm,self.seed,evaluation.PREDICTION_ROOT).exists(): raise FileExistsError("preserving existing prediction output")
   self._write("prediction_pending",ground_truth_used=False,seed=self.seed)
   if self._gpu_stage("predict",prepared=prepared): return 1
   self._write("validating_predictions_before_gt")
   for arm in evaluation.ARMS:
    evaluation.validate_prediction(arm,seed=self.seed,prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   self._write("scoring_gt_last",prediction_complete=True)
   report=evaluation.score_all(seed=self.seed,output=evaluation.score_path(self.seed),prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   if report.get("status")!="complete" or report.get("ground_truth_opened_after_all_predictions_validated") is not True or report.get("ground_truth_used_for_prediction") is not False or report.get("ground_truth_used_for_scoring") is not True:
    raise ValueError("GT-last score producer contract failed")
   checked=evaluation.validate_score(evaluation.score_path(self.seed),seed=self.seed,prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   self._write("complete",score_sha256=checked["score_sha256"],replication_gate=checked["replication_gate"],finished=time.time(),ground_truth_used_for_prediction=False,ground_truth_used_for_scoring=True)
   return 0
  except BaseException as exc:
   if self.child is not None and self.child.poll() is None: self.owner._stop_owned(self.child)
   self._write("failed",error=repr(exc),finished=time.time()); raise
  finally:
   if self.lease is not None:
    try: fcntl.flock(self.lease.fileno(),fcntl.LOCK_UN)
    finally: self.lease.close(); self.lease=None
   self._unlock()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--seed",type=int,choices=evaluation.SEEDS,required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
 if a.dry_run:
  print(json.dumps({"status":"planned","seed":a.seed,"arms":list(evaluation.ARMS),"count":evaluation.COUNT,"ids":[1320,1639],"queue_fingerprint":implementation_fingerprint(),"evaluation_fingerprint":evaluation.implementation_fingerprint(),"ground_truth_used_for_prediction":False,"scoring":"CPU after GPU lease release"},allow_nan=False)); return 0
 return StageQueue(seed=a.seed).run()
if __name__=="__main__": raise SystemExit(main())
