"""CPU recovery and v2 handoff contracts for completed SW0168 training."""
from __future__ import annotations
import json, os, sys, unittest
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0168_end_to_end_connectivity import (
    recover_completed_training as recovery, prediction_queue_v2 as prediction,
    score_queue_v2 as scoring, train,
)

def _new_fixture(prefix):
    parent=Path(__file__).resolve().parents[1]/"temporary"
    parent.mkdir(parents=True,exist_ok=True)
    path=parent/f"{prefix}-{os.getpid()}-{time.time_ns()}"
    path.mkdir(exist_ok=False)
    return path

def _remove_fixture(path):
    path=Path(path)
    parent=Path(__file__).resolve().parents[1]/"temporary"
    if path.parent.resolve()!=parent.resolve(): raise RuntimeError("fixture cleanup escaped its owned temp root")
    for item in sorted(path.rglob("*"),reverse=True):
        if item.is_file(): item.unlink()
        elif item.is_dir(): item.rmdir()
    path.rmdir()

class CompletedTrainingRecoveryTests(unittest.TestCase):
    def test_actual_producer_shaped_failed_and_complete_states_adopt_only_validated_artifacts(self):
        temp = _new_fixture("sw168-recovery")
        queue = temp / "queue"; artifacts = temp / "artifacts"; runtime = temp / "recovery"
        queue.mkdir(parents=True, exist_ok=False); artifacts.mkdir(); runtime.mkdir()
        old = (recovery.QUEUE_ROOT, recovery.ARTIFACT_ROOT, recovery.RUNTIME, recovery.RECEIPT)
        recovery.QUEUE_ROOT, recovery.ARTIFACT_ROOT = queue, artifacts
        recovery.RUNTIME, recovery.RECEIPT = runtime, runtime / "receipt.json"
        try:
            base_bindings = {"experiment": "SW0168_actual_spike_connectivity_training", "seed": 1,
                "arms": list(train.ARMS), "source_cache_manifest_sha256": "a"*64,
                "context_cache_manifest_sha256": "b"*64, "ordered_train_ids": [10, 11]}
            for arm in train.ARMS:
                (queue / f"{arm}.lock").write_text("", encoding="utf-8")
                arm_dir = artifacts / f"{arm}_seed1"; arm_dir.mkdir()
                bindings = {**base_bindings, "arm": arm}
                manifest = {"experiment": base_bindings["experiment"], "status": "training_complete",
                    "arm": arm, "seed": 1, "completed_updates": 64, "bindings": bindings,
                    "optimizer_step_summary": {"native_core": {"max_step": 64}}}
                (arm_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                for name in ("history.json", "optimizer.pt", "checkpoint.pt", "boundary.ptx"):
                    (arm_dir / name).write_bytes((arm + name).encode())
                supervisor = 700001 + train.ARMS.index(arm)
                state = {"experiment": "SW0168_training_queue", "arm": arm,
                    "supervisor_pid": supervisor, "status": "failed",
                    "error": recovery.IDENTITY_ERROR}
                if arm == train.ARMS[1]:
                    state = {"experiment": "SW0168_training_queue", "arm": arm,
                        "supervisor_pid": supervisor, "status": "complete",
                        "manifest_sha256": recovery._sha(arm_dir / "manifest.json")}
                (queue / f"{arm}_state.json").write_text(json.dumps(state), encoding="utf-8")
                (queue / f"{arm}_worker_attempt01.log").write_text(
                    json.dumps({"arm": arm, "completed_updates": 64, "status": "training_complete"}) + "\n",
                    encoding="utf-8")
                (queue / f"{arm}_supervisor.log").write_text("captured supervisor output\n", encoding="utf-8")
            with (mock.patch.object(recovery, "_assert_old_lock_free"),
                  mock.patch.object(recovery, "_active_training_processes", return_value=[]),
                  mock.patch.object(recovery, "_proc_argv", return_value=None),
                  mock.patch.object(train, "validate_training_artifact", side_effect=lambda arm, path: json.loads((Path(path)/"manifest.json").read_text()))):
                receipt = recovery.publish_receipt()
                self.assertEqual(receipt["status"], "both_arms_validated_complete")
                self.assertEqual(set(receipt["arms"]), set(train.ARMS))
                self.assertTrue(all(x["validated_completed_updates"] == 64 for x in receipt["arms"].values()))
                self.assertEqual(recovery.validate_receipt(deep=False)["experiment"], recovery.EXPERIMENT)
                with self.assertRaises(FileExistsError): recovery.publish_receipt()
                changed = artifacts / f"{train.ARMS[0]}_seed1" / "manifest.json"
                changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaises(ValueError): recovery.validate_receipt(deep=False)
        finally:
            recovery.QUEUE_ROOT, recovery.ARTIFACT_ROOT, recovery.RUNTIME, recovery.RECEIPT = old
            _remove_fixture(temp)

    def test_v2_prediction_state_is_mutable_and_receipt_bound(self):
        temp = _new_fixture("sw168-v2-state")
        receipt = temp / "receipt.json"; receipt.write_text("{}", encoding="utf-8")
        old_runtime, old_receipt = prediction.RUNTIME, recovery.RECEIPT
        prediction.RUNTIME, recovery.RECEIPT = temp, receipt
        try:
            arm = train.ARMS[0]
            prediction._write_state(arm, "waiting_for_training")
            prediction._write_state(arm, "complete", training_manifest_sha256="a"*64)
            state = json.loads((temp / f"{arm}_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["experiment"], prediction.EXPERIMENT)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["recovery_receipt_sha256"], recovery._sha(receipt))
        finally:
            prediction.RUNTIME, recovery.RECEIPT = old_runtime, old_receipt
            _remove_fixture(temp)

    def test_normal_exit_proc_snapshot_race_waits_for_owned_popen(self):
        class Child:
            pid = 1234; args = ["python", "-m", "worker"]
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout=None): self.returncode = 0; return 0
        child = Child()
        owner = SimpleNamespace(_proc_record=lambda pid: None)
        record = prediction._owned_proc_record(owner, child, child.args)
        self.assertIsNone(record)
        self.assertEqual(child.poll(), 0)

    def test_active_gpu_memory_is_not_a_foreign_owner_signal(self):
        self.assertFalse(prediction._foreign_owner_changed("GPU-A", "GPU-A", [1234], 1234))
        self.assertTrue(prediction._foreign_owner_changed("GPU-A", "GPU-A", [1234, 4321], 1234))
        self.assertTrue(prediction._foreign_owner_changed("GPU-A", "GPU-B", [1234], 1234))

    def test_v2_supervisor_finishes_owned_prediction_above_idle_memory_threshold(self):
        class Lease:
            def fileno(self): return 88
            def close(self): pass
        class Child:
            pid=45678; returncode=None
            def __init__(self,args): self.args=args
            def poll(self): return self.returncode
            def wait(self):
                if self.returncode is None: self.returncode=0
                return self.returncode
        temp=_new_fixture("sw168-v2-run"); runtime=temp/"runtime"; runtime.mkdir()
        receipt_path=temp/"receipt.json"; receipt_path.write_text("{}",encoding="utf-8")
        arm=train.ARMS[0]; manifest_sha="a"*64
        receipt={"arms":{name:{"manifest_sha256":manifest_sha} for name in train.ARMS}}
        gpu_calls=[]; child_holder=[]; stopped=[]
        argv=[sys.executable,"-m","collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue",
              "--worker","--arm",arm,"--physical-gpu","2"]
        owner=SimpleNamespace(
            _acquire=lambda stop:(2,"GPU-X",0,Lease()),
            _stable_idle=lambda gpu,uuid,stop:True,
            _proc_record=lambda pid:{"state":"S","argv":argv},
            _stop_owned=lambda child:stopped.append(child),
            gpu_guard=SimpleNamespace(_gpu_info=lambda gpu:(gpu_calls.append(gpu) or
                (("GPU-X",0,[]) if len(gpu_calls)==1 else ("GPU-X",4096,[45678])))),
            MAX_MEMORY_MIB=512)
        old=(prediction.RUNTIME,recovery.RECEIPT)
        prediction.RUNTIME=runtime; recovery.RECEIPT=receipt_path
        def popen(args,**kwargs):
            child=Child(args); child_holder.append(child); return child
        def sleep(_): child_holder[0].returncode=0
        fake_fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=8,
            flock=lambda *args:None)
        try:
            with (mock.patch.dict("sys.modules",{"fcntl":fake_fcntl}),
                  mock.patch.object(recovery,"validate_receipt",return_value=receipt),
                  mock.patch.object(prediction.base,"_owner_preserve_visibility",return_value=owner),
                  mock.patch.object(prediction,"_training_ready",return_value={"status":"training_complete"}),
                  mock.patch.object(prediction.base,"_worker_command",return_value=argv),
                  mock.patch.object(prediction.subprocess,"Popen",side_effect=popen),
                  mock.patch.object(prediction.time,"sleep",side_effect=sleep),
                  mock.patch.object(prediction.base.evaluate,"prepare_prediction_inputs_cpu",return_value={}),
                  mock.patch.object(prediction.base.evaluate,"_bindings",return_value={"binding":"fixed"}),
                  mock.patch.object(prediction.base.evaluate,"implementation_fingerprint",return_value={"fp":"x"}),
                  mock.patch.object(prediction.base.evaluate,"validate_prediction",return_value={
                      "directory":str(temp/"pred"),"prediction_sha256":"b"*64,"protocol_sha256":"c"*64})):
                result=prediction.run(arm,timeout_seconds=30)
            state=json.loads((runtime/f"{arm}_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"],"complete")
            self.assertEqual(result["prediction_sha256"],"b"*64)
            self.assertEqual(gpu_calls,[2,2])
            self.assertEqual(stopped,[])
            self.assertEqual(child_holder[0].returncode,0)
        finally:
            prediction.RUNTIME,recovery.RECEIPT=old
            _remove_fixture(temp)

    def test_score_recognizes_only_exact_v2_prediction_module_and_arm(self):
        arm = train.ARMS[0]
        argv = ["python", "-m", scoring.PRED_MODULE, "--arm", arm]
        owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": argv})
        self.assertTrue(scoring._module_live(owner, 321, arm))
        self.assertFalse(scoring._module_live(owner, 321, train.ARMS[1]))
        bad = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": ["python", "-m", "other", "--arm", arm]})
        self.assertFalse(scoring._module_live(bad, 321, arm))

if __name__ == "__main__": unittest.main()
