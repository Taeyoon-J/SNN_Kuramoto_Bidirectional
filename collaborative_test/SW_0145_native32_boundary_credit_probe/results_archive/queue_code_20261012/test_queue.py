from __future__ import annotations
import importlib.util,json,os,sys,tempfile,threading,unittest,uuid,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ACMLAB=HERE.parents[1]
ROOT=Path(os.environ.get("PATCH_V2_REPO",ACMLAB/"patch_v2_sw")).resolve()
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
sys.path[:0]=[str(HERE),str(ACMLAB),str(ROOT),str(ROOT/"collaborative_test")]
spec=importlib.util.spec_from_file_location("sw0145_probe_queue_test",HERE/"probe_queue.py")
q=importlib.util.module_from_spec(spec); sys.modules[spec.name]=q; spec.loader.exec_module(q)
class Clock:
 def __init__(self):self.t=0
 def monotonic(self):return self.t
 def sleep(self,x):self.t+=x
class QueueTests(unittest.TestCase):
 def test_one_task_dry_plan_does_not_create_outputs(self):
  tasks=q.task_plan(); self.assertEqual(len(tasks),1); self.assertEqual(tasks[0]["seed"],0); self.assertEqual(tasks[0]["stage"],"disposable_probe"); self.assertFalse(q.STATE.exists())
 def test_stable_gate_uses_reviewed_sixty_second_same_uuid_helper(self):
  c=Clock(); seen=[]; ok,detail=q.helper.stable_idle_window(lambda:("same",2,[]),stop_event=threading.Event(),monotonic=c.monotonic,sleep=c.sleep,on_sample=lambda elapsed,*_:seen.append(elapsed))
  self.assertTrue(ok); self.assertEqual(detail["stable_seconds"],60); self.assertEqual(c.t,60); self.assertGreaterEqual(len(seen),12)
 def test_result_validator_binds_probe_contract_and_run_digest(self):
  path=Path(tempfile.gettempdir())/("sw0145_validation_"+uuid.uuid4().hex+".json")
  try:
   r={"status":"disposable_probe_complete","experiment":"SW0145_native32_boundary_credit_probe","seed":0,"resource_only":True,"training_admission":False,"ground_truth_used":False,"optimizer_updates":1,"implementation_sha256":"a"*64,"gradient_comparison_after_disposable_step":{"per_logical_batch_gradient_norms":[{},{},{},{}]}}
   path.write_text(json.dumps(r),encoding="utf-8"); self.assertTrue(q.validate_result(path,run_sha="a"*64))
   r["training_admission"]=True; path.write_text(json.dumps(r),encoding="utf-8"); self.assertFalse(q.validate_result(path,run_sha="a"*64)); self.assertFalse(q.validate_result(path,run_sha="b"*64))
  finally:
   if path.exists():path.unlink()
 @unittest.skipIf(q.fcntl is None,"Linux flock inheritance is exercised on server")
 def test_shared_gpu_lease_fd_can_be_inherited_and_released(self):
  path=Path(tempfile.gettempdir())/("sw0145_lease_"+uuid.uuid4().hex+".lock"); fd=None
  try:
   fd=path.open("a+"); q.fcntl.flock(fd.fileno(),q.fcntl.LOCK_EX|q.fcntl.LOCK_NB)
   child=subprocess.run([sys.executable,"-c","import os,sys; os.fstat(int(sys.argv[1]))",str(fd.fileno())],pass_fds=(fd.fileno(),),check=False)
   self.assertEqual(child.returncode,0); q.fcntl.flock(fd.fileno(),q.fcntl.LOCK_UN); fd.close(); fd=None
   other=path.open("a+")
   try:q.fcntl.flock(other.fileno(),q.fcntl.LOCK_EX|q.fcntl.LOCK_NB)
   finally:q.fcntl.flock(other.fileno(),q.fcntl.LOCK_UN); other.close()
  finally:
   if fd is not None:fd.close()
   if path.exists():path.unlink()
if __name__=="__main__":unittest.main()