"""Owner-safe SW0164 prediction then GT-last scoring supervisor."""
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
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as evaluation,preflight
from collaborative_test.SW_0162_connected_signed_partition import preflight_queue as source_queue
RUNTIME=HERE/"results_archive"/"stage_queue_20261014"; STATE=RUNTIME/"state.json"; LOCK=RUNTIME/"queue.lock"; LOG=RUNTIME/"prediction.attempt01.log"
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
 if any(not p.is_file() for p in paths): raise FileNotFoundError("SW0164 queue dependency missing")
 result={p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths}
 result.update(source_queue.implementation_fingerprint())
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
def command(stage):
 if stage=="source8_preflight": return [sys.executable,"-m","collaborative_test.SW_0164_foreground_preserving_partition.preflight","--device",DEVICE]
 if stage=="predict": return [sys.executable,"-m","collaborative_test.SW_0164_foreground_preserving_partition.predict_evaluate","--stage","predict","--device",DEVICE]
 raise ValueError(f"unknown GPU stage {stage}")
class StageQueue:
 def __init__(self,*,prepared_fn=None,owner_api=None):
  os.environ["CUDA_VISIBLE_DEVICES"]=""; self.owner=owner if owner_api is None else owner_api
  if fcntl is None: raise RuntimeError("SW0164 stage queue requires Linux flock")
  RUNTIME.mkdir(parents=True,exist_ok=True); self.lock=LOCK.open("a+")
  try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc: self.lock.close(); raise RuntimeError("SW0164 stage queue already owned") from exc
  self.state_lock=threading.RLock(); self.stop_event=threading.Event(); self.child=None; self.lease=None
  self.fingerprint=implementation_fingerprint(); self.eval_fingerprint=evaluation.implementation_fingerprint(); self.prepared_fn=prepared_fn or evaluation.prepare_prediction_inputs_cpu
  try:
   if STATE.exists() or LOG.exists(): raise FileExistsError("preserving existing SW0164 queue state/log")
   self._write("waiting_for_source_preflight",supervisor_pid=os.getpid(),queue_fingerprint=self.fingerprint,evaluation_fingerprint=self.eval_fingerprint,
    task={"arms":list(evaluation.ARMS),"count":evaluation.COUNT,"image_ids":[1320,1639],"steps":1024,"settle":512,"optimizer_updates":0,"ground_truth_used_for_prediction":False})
  except BaseException: self._unlock(); raise
 def _unlock(self):
  try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN)
  finally: self.lock.close()
 def _write(self,status,**fields):
  with self.state_lock:
   cur=json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
   cur.update(experiment="SW0164_foreground_preserving_partition",status=status,updated=time.time(),**fields); _atomic_json(STATE,cur)
 def _verify(self):
  if implementation_fingerprint()!=self.fingerprint or evaluation.implementation_fingerprint()!=self.eval_fingerprint: raise RuntimeError("SW0164 code changed after queue registration")
 def _preflight_live(self,state):
  for key,module in (("supervisor_pid","collaborative_test.SW_0162_connected_signed_partition.preflight_queue"),("child_pid","collaborative_test.SW_0162_connected_signed_partition.preflight")):
   pid=state.get(key)
   if type(pid) is not int or pid<=0: continue
   try: row=self.owner._proc_record(pid)
   except (OSError,RuntimeError,ValueError): continue
   if row and row.get("state")!="Z" and module in row.get("argv",[]): return True
  return False
 def _await_preflight(self):
  while not self.stop_event.is_set():
   path=source_queue.STATE; st=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
   if st is None: raise RuntimeError("registered SW0162 source preflight queue is missing")
   if st.get("experiment")!="SW0162_connected_signed_partition": raise ValueError("unexpected upstream preflight experiment")
   if st.get("status")=="complete":
    if not source_queue.validate_report(preflight.source.OUTPUT,expected_fingerprint=preflight.source.implementation_fingerprint(),expected_source=source_queue.source_identity()): raise ValueError("upstream SW0162 preflight failed strict validation")
    return True
   if st.get("status") in ACTIVE_PREFLIGHT and self._preflight_live(st):
    self._write("waiting_for_source_preflight",upstream_status=st.get("status"),upstream_supervisor_pid=st.get("supervisor_pid")); self.stop_event.wait(POLL_SECONDS); continue
   raise RuntimeError(f"upstream source preflight terminal/nonlive: {st.get('status')}")
  raise RuntimeError("SW0164 queue stopped while waiting for source preflight")
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
    self._await_preflight(); preflight.prepare_preflight_cpu()
   else:
    self._await_preflight()
    if not preflight.validate_report(preflight.OUTPUT,expected_fingerprint=preflight.implementation_fingerprint(),expected_source=evaluation._expected_source()): raise ValueError("SW0164 source8 preflight invalid before prediction")
    prepared=evaluation.prepare_prediction_cpu()
   self._verify(); ok,detail=self._idle(gpu,gpu_uuid)
   if ok: break
   fcntl.flock(lease.fileno(),fcntl.LOCK_UN); lease.close(); self.lease=None
   self._write("waiting_for_exclusive_gpu",stage=stage,reservation_released=True,gpu_check=detail)
  if self.stop_event.is_set(): return 1
  env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env.setdefault("OMP_NUM_THREADS","1"); env.setdefault("MKL_NUM_THREADS","1")
  try:
   with LOG.open("a" if LOG.exists() else "x",encoding="utf-8",newline="\n") as log:
    self.child=subprocess.Popen(command(stage),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(self.lease.fileno(),))
    self._write(f"running_{stage}",child_pid=self.child.pid,child_pgid=self.child.pid,gpu=gpu,gpu_uuid=gpu_uuid,started=time.time())
    deadline=time.monotonic()+WATCHDOG_SECONDS
    while self.child.poll() is None:
     if time.monotonic()>=deadline: self.owner._stop_owned(self.child); raise TimeoutError(f"SW0164 {stage} watchdog expired")
     try: uid,used,owners=self.owner.gpu_guard._gpu_info(gpu)
     except Exception: uid,used,owners=gpu_uuid,0,[]
     if uid!=gpu_uuid or any(pid!=self.child.pid for pid in owners): self.owner._stop_owned(self.child); raise RuntimeError("foreign GPU collision; stopped only owned child")
     self._write(f"running_{stage}",child_pid=self.child.pid,gpu=gpu,gpu_uuid=gpu_uuid,memory_used_mib=int(used)); time.sleep(1)
    rc=self.child.wait()
   if rc!=0: raise RuntimeError(f"{stage} child failed rc={rc}; see {LOG}")
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
   self.prepared_fn() # CPU-only references/RGB/HDF metadata; no masks opened.
   self._await_preflight(); self._verify()
   if not preflight.OUTPUT.exists():
    self._write("source8_preflight_pending",ground_truth_used=False)
    if self._gpu_stage("source8_preflight"): return 1
   if not preflight.validate_report(preflight.OUTPUT,expected_fingerprint=preflight.implementation_fingerprint(),expected_source=evaluation._expected_source()): raise ValueError("SW0164 source8 preflight report invalid")
   prepared=evaluation.prepare_prediction_cpu()
   for arm in evaluation.ARMS:
    if evaluation._out(arm,evaluation.PREDICTION_ROOT).exists(): raise FileExistsError("preserving existing prediction output")
   self._write("prediction_pending",ground_truth_used=False)
   if self._gpu_stage("predict",prepared=prepared): return 1
   self._write("validating_predictions_before_gt")
   for arm in evaluation.ARMS: evaluation.validate_prediction(arm,prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   self._write("scoring_gt_last",prediction_complete=True)
   report=evaluation.score_all(output=evaluation.SCORE_PATH,prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   if report.get("status")!="complete" or report.get("ground_truth_opened_after_all_predictions_validated") is not True or report.get("ground_truth_used_for_prediction") is not False or report.get("ground_truth_used_for_scoring") is not True: raise ValueError("GT-last score producer contract failed")
   checked=evaluation.validate_score(evaluation.SCORE_PATH,prediction_root=evaluation.PREDICTION_ROOT,work_root=evaluation.WORK_ROOT,prepared=prepared)
   self._write("complete",score_sha256=checked["score_sha256"],seed1_expansion_gate=checked["seed1_expansion_gate"],finished=time.time(),ground_truth_used_for_prediction=False,ground_truth_used_for_scoring=True)
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
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
 if a.dry_run:
  print(json.dumps({"status":"planned","arms":list(evaluation.ARMS),"count":evaluation.COUNT,"ids":[1320,1639],"queue_fingerprint":implementation_fingerprint(),"evaluation_fingerprint":evaluation.implementation_fingerprint(),"ground_truth_used_for_prediction":False,"scoring":"CPU after GPU lease release"},allow_nan=False)); return 0
 return StageQueue().run()
if __name__=="__main__": raise SystemExit(main())
