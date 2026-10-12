"""CPU guards for SW0146 fresh-arm recovery."""
from __future__ import annotations
import importlib.util,json,os,sys,threading,unittest,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("sw0146_recovery_v2_test",HERE/"recover_queue.py")
recovery=importlib.util.module_from_spec(spec);sys.modules[spec.name]=recovery;spec.loader.exec_module(recovery)

class RecoveryV2Tests(unittest.TestCase):
 def state(self):
  tasks={}
  for arm in ("encoder_graph","joint"):
   tasks["preflight_"+arm]={"status":"passed","artifact_valid":True}
  tasks["train_encoder_graph"]={"status":"failed","child_pid":461418,
   "error":"RuntimeError: foreign GPU owner appeared: [464125]"}
  tasks["train_joint"]={"status":"failed","child_pid":462591,
   "error":"RuntimeError: foreign GPU owner appeared: [463556]"}
  return {"experiment":"SW0146_native32_boundary_q_pilot","status":"failed","supervisor_pid":458960,
   "run_sha256":recovery.RUN_SHA,"protocol_sha256":recovery.PROTOCOL_SHA,"queue_sha256":recovery.QUEUE_SHA,
   "tasks":tasks}
 def test_accepts_only_observed_two_arm_foreign_failures_and_dead_processes(self):
  s=self.state()
  self.assertTrue(recovery.validate_terminal(s,pid_alive=lambda p:False,process_rows=[]))
  bad=self.state();bad["tasks"]["train_encoder_graph"]["error"]="other"
  with self.assertRaisesRegex(ValueError,"interruption differs"):recovery.validate_terminal(bad,pid_alive=lambda p:False,process_rows=[])
  with self.assertRaisesRegex(ValueError,"still alive"):recovery.validate_terminal(s,pid_alive=lambda p:p==461418,process_rows=[])
 def test_preserves_healthy_process_and_rejects_other_live_experiment_argv(self):
  s=self.state();control=s["tasks"]["train_encoder_graph"]["child_pid"]
  self.assertTrue(recovery.validate_terminal(s,pid_alive=lambda p:False,process_rows=[(control,["python","run.py","--arm","encoder_graph"]) ]))
  argv=["python","-m","collaborative_test.SW_0146_native32_boundary_q_pilot.run","--stage","train","--arm","joint"]
  with self.assertRaisesRegex(ValueError,"another SW0146 run process"):
   recovery.validate_terminal(s,pid_alive=lambda p:False,process_rows=[(999,argv)])
 def test_preflights_reused_and_predictions_wait_for_both_new_arms(self):
  self.assertEqual(recovery.recovery_plan(),["train_encoder_graph","train_joint","predict_encoder_graph","predict_joint","score"])
  self.assertFalse(recovery.score_dependencies_passed({"predict_encoder_graph":"passed","predict_joint":"queued"}))
  self.assertTrue(recovery.score_dependencies_passed({"predict_encoder_graph":"passed","predict_joint":"passed"}))
 def test_only_empty_arm_directory_can_be_archived_and_missing_is_preserved(self):
  root=HERE/("test_archive_"+uuid.uuid4().hex[:8])
  try:
   root.mkdir();training=root/"training";training.mkdir();empty=training/"joint_seed1";empty.mkdir()
   dest=root/"runtime"/"empty_joint";self.assertEqual(Path(recovery.archive_empty(empty,dest,training)),dest)
   self.assertTrue(dest.is_dir());self.assertEqual(list(dest.iterdir()),[])
   missing=training/"encoder_graph_seed1";self.assertIsNone(recovery.archive_empty(missing,root/"runtime"/"missing",training))
   bad=training/"partial";bad.mkdir();f=bad/"history.json";f.write_text("preserve",encoding="utf-8")
   with self.assertRaisesRegex(FileExistsError,"nonempty"):recovery.archive_empty(bad,root/"runtime"/"partial",training)
   self.assertEqual(f.read_text(encoding="utf-8"),"preserve")
  finally:
   f.unlink();bad.rmdir();dest.rmdir();(root/"runtime").rmdir();training.rmdir();root.rmdir()
 def test_terminal_state_sha_and_both_children_are_part_of_guard(self):
  with self.assertRaisesRegex(ValueError,"terminal state SHA"):
   recovery.validate_terminal_bytes(b"{}")
  s=self.state()
  with self.assertRaisesRegex(ValueError,"still alive"):
   recovery.validate_terminal(s,pid_alive=lambda p:p==462591,process_rows=[])
  with self.assertRaisesRegex(ValueError,"supervisor is still alive"):
   recovery.validate_terminal(s,pid_alive=lambda p:p==458960,process_rows=[])
 @unittest.skipIf(os.name=="nt","OneDrive does not support concurrent atomic replace; verified on Linux deployment")
 def test_concurrent_worker_state_updates_are_atomic(self):
  base=HERE/("test_state_"+uuid.uuid4().hex[:8]);base.mkdir()
  state_path=base/"state.json";old_state=recovery.queue.STATE
  try:
   recovery.queue.STATE=state_path
   q=object.__new__(recovery.RecoveryQueue);q._state_lock=threading.RLock()
   q.state={"tasks":{"a":{"n":0},"b":{"n":0},"c":{"n":0},"d":{"n":0}}}
   def worker(k):
    for i in range(40):q._update(k,n=i)
   workers=[threading.Thread(target=worker,args=(k,)) for k in q.state["tasks"]]
   for t in workers:t.start()
   for t in workers:t.join()
   saved=json.loads(state_path.read_text(encoding="utf-8"))
   self.assertEqual({k:v["n"] for k,v in saved["tasks"].items()},{"a":39,"b":39,"c":39,"d":39})
  finally:
   recovery.queue.STATE=old_state
   if state_path.exists():state_path.unlink()
   base.rmdir()

if __name__=="__main__":unittest.main(verbosity=2)
