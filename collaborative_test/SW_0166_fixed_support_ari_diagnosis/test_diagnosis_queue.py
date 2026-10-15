import os, shutil, unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0166_fixed_support_ari_diagnosis import diagnosis_queue as queue


class QueueGateTests(unittest.TestCase):
    def setUp(self):
        self.base=queue.HERE/f"queue_test_{os.getpid()}"
        if self.base.exists(): raise RuntimeError(f"refusing to reuse {self.base}")
        self.base.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.base))

    def test_waits_for_exact_registered_owner_then_admits_complete(self):
        runtime=self.base/"stage_queue_seed0_20261014"; runtime.mkdir()
        state_path=runtime/"state.json"
        state_path.write_text('{"status":"running_predict","seed":0,"supervisor_pid":54321}',encoding="utf-8")
        sequence=[{"status":"running_predict","seed":0,"supervisor_pid":54321},
                  {"status":"complete","seed":0,"supervisor_pid":54321,"score_sha256":"b"*64}]
        reads=[]
        def reader(_):
            reads.append(1); return sequence.pop(0)
        with mock.patch.object(queue.q165,"RUNTIME_BASE",self.base), mock.patch.object(queue,"_module_live",return_value=True):
            got=queue.completed_score_gate(0,wait=True,deadline=10,sleep=lambda _:None,state_reader=reader)
        self.assertEqual(got["status"],"complete"); self.assertEqual(len(reads),2)

    def test_dead_owner_or_terminal_failure_is_not_a_wait_state(self):
        with mock.patch.object(queue.q165,"RUNTIME_BASE",self.base), mock.patch.object(queue,"_module_live",return_value=False):
            d=self.base/"stage_queue_seed2_20261014"; d.mkdir()
            (d/"state.json").write_text('{"status":"running_predict","seed":2,"supervisor_pid":123}',encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"no live registered owner"):
                queue.completed_score_gate(2,wait=True,sleep=lambda _:None)
            (d/"state.json").write_text('{"status":"failed","seed":2,"supervisor_pid":123}',encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"terminal or has no live"):
                queue.completed_score_gate(2,wait=True,sleep=lambda _:None)

    def test_live_owner_must_match_exact_seed_in_state_and_argv(self):
        with mock.patch.object(queue.owner,"_proc_record",return_value={"state":"S","argv":[
                "python","-m","collaborative_test.SW_0165_foreground_partition_replication.stage_queue","--seed","2"]}):
            self.assertTrue(queue._module_live({"supervisor_pid":123,"seed":2},2))
            self.assertFalse(queue._module_live({"supervisor_pid":123,"seed":0},0))
        d=self.base/"stage_queue_seed0_20261014"; d.mkdir()
        (d/"state.json").write_text('{"status":"running_predict","seed":2,"supervisor_pid":123}',encoding="utf-8")
        with mock.patch.object(queue.q165,"RUNTIME_BASE",self.base):
            with self.assertRaisesRegex(ValueError,"different seed"):
                queue.completed_score_gate(0,wait=True,sleep=lambda _:None)


if __name__=="__main__": unittest.main()
