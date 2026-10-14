import unittest
import json
import os
import uuid
from pathlib import Path

import numpy as np

from collaborative_test.SW_0160_end_to_end_source_encoder import recovery


class ProtocolRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "experiment": "SW0160_end_to_end_source_encoder",
            "status": "complete", "arm": "full_encoder_joint", "seed": 1,
            "image_ids": [1320, 1639], "count": 320,
            "grid_size": [32, 32], "prediction_shape": [320, 32, 32],
            "steps": 1024, "settle": 512, "live_tail": 0,
            "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
            "prediction_bindings": {"dataset_path": "/dataset", "dataset_size_bytes": 10},
            "prediction_bindings_sha256": "a" * 64,
            "prediction_row_commits_sha256": "b" * 64,
            "implementation_fingerprint": "c" * 64,
            "source_checkpoint_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
            "source_training_ids_sha256": "f" * 64,
            "source_qcc_prediction_sha256": "1" * 64,
            "source_T_prediction_sha256": "2" * 64,
            "training_manifest_sha256": "3" * 64,
            "training_checkpoint_sha256": "4" * 64,
            "training_model_state_sha256": "5" * 64,
        }
        self.manifest = {
            "source_base_before_sha256": "6" * 64,
            "source_base_after_sha256": "7" * 64,
        }
        self.protocol = dict(self.expected)
        self.protocol.update({
            "prediction_sha256": "8" * 64,
            "model_state_sha256_before": "5" * 64,
            "model_state_sha256_after": "5" * 64,
            "readout": "original SW0134 actual-spike native32 QCC from source-T T1024 rollout",
            "actual_thresholds": [0.1, 0.2, 0.3, 0.4],
        })

    def test_producer_protocol_migrates_only_two_missing_mirrors(self):
        original = dict(self.protocol)
        recovery.validate_legacy_protocol_shape(
            self.protocol, self.expected, "full_encoder_joint", self.manifest)
        migrated = recovery.validate_and_migrate_protocol_fields(
            self.protocol, self.expected, "full_encoder_joint", self.manifest)
        self.assertEqual(migrated["source_base_before_sha256"], "6" * 64)
        self.assertEqual(migrated["source_base_after_sha256"], "7" * 64)
        self.assertEqual({k: v for k, v in migrated.items() if k not in {
            "source_base_before_sha256", "source_base_after_sha256"}}, original)
        self.assertEqual(self.protocol, original)  # no in-place mutation

    def test_partial_or_mismatching_mirror_is_rejected(self):
        partial = dict(self.protocol, source_base_before_sha256="6" * 64)
        with self.assertRaisesRegex(ValueError, "partial"):
            recovery.validate_and_migrate_protocol_fields(
                partial, self.expected, "full_encoder_joint", self.manifest)
        mismatch = dict(self.protocol, source_base_before_sha256="9" * 64,
                        source_base_after_sha256="7" * 64)
        with self.assertRaisesRegex(ValueError, "does not match"):
            recovery.validate_and_migrate_protocol_fields(
                mismatch, self.expected, "full_encoder_joint", self.manifest)

    def test_numeric_contract_and_unexpected_fields_fail_closed(self):
        changed = dict(self.protocol, count=319)
        with self.assertRaisesRegex(ValueError, "count"):
            recovery.validate_and_migrate_protocol_fields(
                changed, self.expected, "full_encoder_joint", self.manifest)
        extra = dict(self.protocol, unregistered_field="x")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            recovery.validate_legacy_protocol_shape(
                extra, self.expected, "full_encoder_joint", self.manifest)
        changed_threshold = dict(self.protocol, actual_thresholds=[0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "threshold"):
            recovery.validate_legacy_protocol_shape(
                changed_threshold, self.expected, "full_encoder_joint", self.manifest)

    def test_checkpoint_digest_must_match_protocol_model_fields(self):
        bad = dict(self.protocol, model_state_sha256_after="9" * 64)
        recovery.validate_legacy_protocol_shape(
            bad, self.expected, "full_encoder_joint", self.manifest)
        with self.assertRaisesRegex(ValueError, "model-state"):
            recovery.validate_prediction_artifact_mirrors(
                bad, prediction_sha256="8" * 64, model_state_sha256="5" * 64)
        with self.assertRaisesRegex(ValueError, "NPZ"):
            recovery.validate_prediction_artifact_mirrors(
                self.protocol, prediction_sha256="9" * 64, model_state_sha256="5" * 64)

    def test_candidate_and_control_source_base_semantics(self):
        recovery.source_base_mirrors("full_encoder_joint", self.manifest)
        control = {"source_base_before_sha256": "6" * 64,
                   "source_base_after_sha256": "6" * 64}
        recovery.source_base_mirrors("frozen_base_joint", control)
        with self.assertRaisesRegex(ValueError, "did not change"):
            recovery.source_base_mirrors("full_encoder_joint", control)
        with self.assertRaisesRegex(ValueError, "changed its source base"):
            recovery.source_base_mirrors("frozen_base_joint", self.manifest)

    def test_actual_terminal_handoff_requires_exact_failed_cause_and_dead_owners(self):
        fixture_dir = (Path(__file__).parent / "results_archive" / "cpu_test_scratch"
                       / f"recovery-{os.getpid()}-{uuid.uuid4().hex}")
        fixture_dir.mkdir(parents=True)
        try:
            state_path = fixture_dir / "state.json"
            error = "ValueError('SW0160 prediction provenance mismatch: source_base_before_sha256')"
            state_path.write_text(json.dumps({
                "experiment": recovery.train.EXPERIMENT,
                "arm": "full_encoder_joint", "status": "failed",
                "error": error, "supervisor_pid": 12345, "child_pid": 12346,
            }), encoding="utf-8")
            specs = [{"kind": "prediction", "arm": "full_encoder_joint",
                "state": state_path, "module": "prediction_queue",
                "child_module": "predict_evaluate", "terminal_status": "failed",
                "terminal_error": error}]
            dead = lambda _pid: None
            states = recovery.assert_old_queues_terminal(specs, proc_reader=dead)
            self.assertEqual(states[str(state_path)]["record"]["error"], error)
            record = json.loads(state_path.read_text(encoding="utf-8"))
            record["error"] = "ValueError('different failure')"
            state_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "registered mirror mismatch"):
                recovery.assert_old_queues_terminal(specs, proc_reader=dead)
            record["error"] = error
            record["child_pid"] = 12347
            state_path.write_text(json.dumps(record), encoding="utf-8")
            live_wrong_identity = lambda _pid: {"state": "S", "argv": ["unexpected"]}
            with self.assertRaisesRegex(RuntimeError, "identity"):
                recovery.assert_old_queues_terminal(specs, proc_reader=live_wrong_identity)
        finally:
            state_path.unlink(missing_ok=True)
            fixture_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
