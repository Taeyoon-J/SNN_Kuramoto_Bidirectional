"""Focused recovery invariants; no GPU and no mutation of the old queue receipts."""
import hashlib
import json
import os
import shutil
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0163_relation_context_assignment import recovery_training_queue as recovery
from collaborative_test.SW_0163_relation_context_assignment import recovery_prediction_queue, recovery_score_queue


class RecoveryQueueTests(unittest.TestCase):
    def test_terminal_snapshot_is_exact_and_includes_both_failed_descendants(self):
        evidence = recovery.validate_failure_snapshot(process_live=lambda _pid: False,
                                                       require_live_states=False)
        self.assertEqual(evidence["snapshot_sha256"], recovery.EXPECTED_FAILURE_RECEIPT_SHA256)
        self.assertEqual(evidence["captured_failure_statuses"], {
            "training_queue_20261014": "failed_training",
            "prediction_queue_20261014": "failed",
            "score_queue_20261014": "failed"})
        with self.assertRaises(RuntimeError):
            recovery.validate_failure_snapshot(process_live=lambda _pid: True,
                                               require_live_states=False)

    def test_recovery_prediction_requires_recovered_four_arm_terminal(self):
        temp = Path(recovery.HERE) / f".recovery-dep-test-{os.getpid()}"
        if temp.exists(): raise AssertionError("preserving pre-existing dependency test directory")
        temp.mkdir()
        try:
            path = temp / "state.json"
            state = {"status": "complete", "recovery_status": "complete_recovered",
                     "recovery_fingerprint": recovery.recovery_fingerprint(),
                     "queue_fingerprint": recovery.base.implementation_fingerprint(),
                     "results": {arm: True for arm in recovery.train.ARMS},
                     "final_arms": {arm: {"manifest_sha256": "a" * 64} for arm in recovery.train.ARMS}}
            path.write_text(json.dumps(state), encoding="utf-8")
            dependency = recovery_prediction_queue.training_dependency(state_path=path)
            self.assertEqual(dependency["state"]["recovery_status"], "complete_recovered")
            state["results"].pop("Q_A")
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaises(ValueError):
                recovery_prediction_queue.training_dependency(state_path=path)
        finally:
            shutil.rmtree(temp)

    def test_live_recovery_module_handoffs_are_recognized_by_exact_cmdline(self):
        temp = Path(recovery.HERE) / f".recovery-proc-test-{os.getpid()}"
        if temp.exists(): raise AssertionError("preserving pre-existing proc fixture")
        temp.mkdir()
        try:
            pid = 424242
            proc = temp / str(pid); proc.mkdir()
            training_module = recovery_prediction_queue.MODULE
            (proc / "cmdline").write_bytes(b"python\0-m\0" + training_module.encode() + b"\0")
            self.assertTrue(recovery.process_module_live(pid, training_module, proc_root=temp))
            self.assertFalse(recovery.process_module_live(pid, "collaborative_test.unrelated", proc_root=temp))
            training_state = temp / "recovery-training.json"
            training_state.write_text(json.dumps({"status": "validating_recovery", "supervisor_pid": pid,
                "recovery_fingerprint": recovery.recovery_fingerprint()}), encoding="utf-8")
            self.assertIsNone(recovery_prediction_queue.training_dependency(
                state_path=training_state,
                process_live=lambda value: recovery._pid_live(value, proc_root=temp)))
            pred_state = temp / "recovery-prediction.json"
            pred_state.write_text(json.dumps({"status": "waiting_for_training", "supervisor_pid": pid}), encoding="utf-8")
            # This pending stage must be owned by its registered recovery prediction module.
            (proc / "cmdline").write_bytes(b"python\0-m\0" + recovery_prediction_queue.MODULE.encode() + b"\0")
            self.assertIsNone(recovery_score_queue.prediction_dependency(
                state_path=pred_state,
                process_live=lambda value: recovery._pid_live(value, proc_root=temp)))
        finally:
            shutil.rmtree(temp)

    def test_recovery_prediction_fingerprint_retains_original_owner_dependencies(self):
        fp = recovery_prediction_queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py", fp)
        self.assertIn("collaborative_test/SW_0163_relation_context_assignment/prediction_queue.py", fp)
        self.assertEqual(fp["recovery_packet_sha256"], recovery.EXPECTED_FAILURE_RECEIPT_SHA256)

    def test_recovery_score_rejects_prediction_without_recovery_lineage(self):
        temp = Path(recovery.HERE) / f".recovery-score-test-{os.getpid()}"
        if temp.exists(): raise AssertionError("preserving pre-existing score test directory")
        temp.mkdir()
        try:
            path = temp / "state.json"
            path.write_text(json.dumps({"status": "complete", "queue_fingerprint": {},
                                        "prediction_hashes": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                recovery_score_queue.prediction_dependency(state_path=path)
        finally:
            shutil.rmtree(temp)

    def test_parent_torch_validation_is_serialized_across_workers(self):
        active = 0
        peak = 0
        guard = threading.Lock()
        def unsafe_validation(*_args, **_kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.015)
            with guard:
                active -= 1
            return {"completed_updates": 512}
        original, original_prepare = recovery.train.validate_training_artifact, recovery.train.prepare_training_inputs_cpu
        recovery.train.validate_training_artifact = unsafe_validation
        recovery.train.prepare_training_inputs_cpu = lambda **_kwargs: {}
        try:
            old_validate, old_prepare = recovery._serialize_parent_validation()
            try:
                threads = [threading.Thread(target=recovery.train.validate_training_artifact) for _ in range(12)]
                for thread in threads: thread.start()
                for thread in threads: thread.join()
            finally:
                recovery.train.validate_training_artifact = old_validate
                recovery.train.prepare_training_inputs_cpu = old_prepare
        finally:
            recovery.train.validate_training_artifact = original
            recovery.train.prepare_training_inputs_cpu = original_prepare
        self.assertEqual(peak, 1)

    def test_full_recovery_adopts_two_and_runs_only_missing_arms(self):
        temp = Path(recovery.HERE) / f".recovery-test-{os.getpid()}"
        if temp.exists():
            raise AssertionError("preserving pre-existing recovery test directory")
        temp.mkdir()
        try:
            root = temp / "training"
            root.mkdir()
            for arm in recovery.ADOPTED:
                output = root / f"{arm}_seed1"
                output.mkdir()
                (output / "manifest.json").write_text("adopt-" + arm, encoding="utf-8")
            original_pins = recovery.ADOPTED_MANIFESTS
            recovery.ADOPTED_MANIFESTS = {arm: recovery.sha256_file(root / f"{arm}_seed1" / "manifest.json")
                                          for arm in recovery.ADOPTED}
            calls = []
            shared_init = recovery.EXPECTED_INITIAL_MODEL_SHA256
            def valid_artifact(arm, *, directory=None, prepared=None):
                self.assertIsNotNone(prepared)
                calls.append(arm)
                return {"completed_updates": 512, "initial_model_state_sha256": shared_init,
                        "checkpoint_sha256": hashlib.sha256(arm.encode()).hexdigest()}
            instances = []
            class FakeQueue:
                def __init__(self, *, runtime):
                    self.runtime = Path(runtime)
                    self.state_path = self.runtime / "state.json"
                    self.stop_event = threading.Event(); self.children = {}; self.leases = {}
                    self.records = []
                    instances.append(self)
                def _verify_code(self): pass
                def _update(self, status, **fields): self.records.append((status, fields))
                def _arm_update(self, arm, status, **fields): self.records.append((arm, status, fields))
                def _run_arm(self, arm, prepared):
                    self.records.append(("run", arm))
                    output = root / f"{arm}_seed1"; output.mkdir()
                    (output / "manifest.json").write_text("new-" + arm, encoding="utf-8")
                    return True
                def _release_lease(self, _arm): pass
                def _unlock(self): pass
            original_root = recovery.train.TRAIN_ROOT
            original_validate = recovery.train.validate_training_artifact
            try:
                recovery.train.TRAIN_ROOT = root
                recovery.train.validate_training_artifact = valid_artifact
                with mock.patch.object(recovery.base, "preflight_dependency", return_value={
                        "prepared": {"preflight_report_sha256": "b" * 64}, "report_sha256": "b" * 64}), \
                     mock.patch.object(recovery, "validate_failure_snapshot", return_value={"snapshot_sha256": "c" * 64}), \
                     mock.patch.object(recovery, "_serialize_parent_validation", return_value=(valid_artifact, lambda **_kw: {})):
                    status = recovery.run_recovery(runtime=Path(temp) / "recovery", acquire_old_locks=False,
                                                   process_live=lambda _pid: False, queue_factory=FakeQueue)
            finally:
                recovery.train.TRAIN_ROOT = original_root
                recovery.ADOPTED_MANIFESTS = original_pins
                recovery.train.validate_training_artifact = original_validate
            self.assertEqual(status, 0)
            ran_arms = [record[1] for record in instances[0].records
                        if isinstance(record, tuple) and len(record) > 1 and record[0] == "run"]
            self.assertCountEqual(ran_arms, list(recovery.TRAIN_ONLY))
            self.assertTrue(any(isinstance(record, tuple) and len(record) == 2 and record[0] == "complete"
                                and record[1].get("recovery_status") == "complete_recovered"
                                for record in instances[0].records))
            adopted_records = [record[0] for record in instances[0].records
                               if isinstance(record, tuple) and len(record) > 1 and record[1] == "adopted_complete"]
            self.assertCountEqual(adopted_records, list(recovery.ADOPTED))
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    unittest.main()
