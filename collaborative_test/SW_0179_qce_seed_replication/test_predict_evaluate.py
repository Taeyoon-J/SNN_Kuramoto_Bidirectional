"""CPU contracts for seed-isolated SW0179 fixed320 prediction records."""
import json
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as pe
from collaborative_test.SW_0179_qce_seed_replication import train


@contextmanager
def _json_fixture(value):
    root = pe.HERE / "results_archive"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"test_prediction_fixture_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        yield path
    finally:
        if not path.parent.samefile(root):
            raise RuntimeError("test fixture cleanup escaped the designated results_archive")
        if path.exists():
            path.unlink()


class SW0179PredictionContracts(unittest.TestCase):
    def test_seed_roots_and_prepared_seed_cannot_collide(self):
        root0 = pe.configure_seed(0)
        out0, work0 = pe._out(pe.ARMS[0]), pe._work(pe.ARMS[0])
        root2 = pe.configure_seed(2)
        out2, work2 = pe._out(pe.ARMS[0]), pe._work(pe.ARMS[0])
        self.assertEqual((root0, root2), (0, 2))
        self.assertNotEqual(out0, out2)
        self.assertNotEqual(work0, work2)
        self.assertIn("seed0", str(out0.parent))
        self.assertIn("seed2", str(out2.parent))
        with self.assertRaisesRegex(ValueError, "different source seed"):
            pe._assert_prepared_seed({"seed": 0}, 2)

    def test_independent_foreground_contracts_allow_standalone_to_differ(self):
        shape = (pe.COUNT, pe.GRID, pe.GRID)
        conditional = np.zeros(shape, dtype=np.int32)
        conditional[:, 0, 0] = 1
        source = conditional > 0
        support = np.zeros(shape, dtype=np.int32)
        support[:, 1, 1] = 5
        standalone = support.copy()
        labels, checked_support = pe.validate_prediction_arrays(
            {"conditional": conditional, "standalone": standalone}, support, source)
        self.assertTrue(np.array_equal(labels["conditional"] > 0, source))
        self.assertTrue(np.array_equal(labels["standalone"] > 0, checked_support > 0))
        self.assertFalse(np.array_equal(labels["standalone"] > 0, source))

    def test_training_receipt_consumes_actual_seeded_producer_schema(self):
        manifest_sha = "a" * 64
        state = {"experiment": pe.TRAIN_QUEUE_EXPERIMENT, "arm": pe.ARMS[0], "seed": 2,
            "status": "complete", "completed_updates": train.UPDATES,
            "validation_data_read": False, "queue_fingerprint_sha256": "f" * 64,
            "manifest_sha256": manifest_sha}
        with _json_fixture(state) as state_path:
            output_root = pe.HERE / "results_archive"
            artifact = output_root / f"{pe.ARMS[0]}_seed2"
            bindings = {"experiment": train.EXPERIMENT, "arm": pe.ARMS[0], "seed": 2,
                "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
                "source_binding": {"seed": 2}, "dataset_producer_proof": {"scope": "shared TRAIN data"}}
            manifest = {"experiment": train.EXPERIMENT, "status": "training_complete",
                "arm": pe.ARMS[0], "seed": 2, "completed_updates": train.UPDATES,
                "validation_data_read": False, "bindings": bindings}
            queue = SimpleNamespace(_state_path=lambda arm, seed: state_path)
            real_sha = pe.sha256_file
            state_sha = real_sha(state_path)
            def fixture_sha(path):
                path = Path(path)
                if path == artifact / "manifest.json": return manifest_sha
                if path == state_path: return state_sha
                return "9" * 64
            with mock.patch.object(train, "OUTPUT_ROOT", output_root), \
                 mock.patch.object(train, "_module", return_value=queue), \
                 mock.patch.object(pe, "sha256_file", side_effect=fixture_sha), \
                 mock.patch.object(train, "validate_training_artifact", return_value=manifest) as validator:
                result = pe._training_records(2, expected_queue_fingerprint="f" * 64)
                validator.assert_called_once_with(pe.ARMS[0], artifact, seed=2)
                self.assertEqual(result["seed"], 2)
                self.assertEqual(result["manifests"][pe.ARMS[0]]["seed"], 2)
            with mock.patch.object(train, "OUTPUT_ROOT", output_root), \
                 mock.patch.object(train, "_module", return_value=queue), \
                 mock.patch.object(pe, "sha256_file", side_effect=fixture_sha), \
                 mock.patch.object(train, "validate_training_artifact", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    pe._training_records(2, expected_queue_fingerprint="0" * 64)

    def test_prediction_bindings_label_data_proof_separately_from_seed_model(self):
        artifact = pe.HERE / "results_archive" / "not_created_seed0_artifact"
        manifest = {"bindings": {"source_binding": {"seed": 0, "source_checkpoint_sha256": "a" * 64},
                    "dataset_producer_proof": {"producer_seed": 1, "scope": "shared data only"},
                    "dataset_manifest_sha256": "b" * 64, "dataset_arrays_sha256": "c" * 64},
            "final_model_state_sha256": "d" * 64, "training_fingerprint": "e" * 64,
            "source_model_state_before_sha256": "7" * 64}
        state = {"queue_fingerprint_sha256": "f" * 64}
        p = {"seed": 0, "manifests": {pe.ARMS[0]: manifest},
            "artifacts": {pe.ARMS[0]: artifact}, "states": {pe.ARMS[0]: state},
            "state_sha256": {pe.ARMS[0]: "1" * 64}, "references": {"source_T": {
                "prediction_sha256": "2" * 64, "protocol_sha256": "3" * 64}},
            "rgb_meta_sha256": "4" * 64, "rgb_hashes": ["5" * 64] * pe.COUNT,
            "dataset_path": "dataset.h5", "dataset_size_bytes": 123, "dataset_mtime_ns": 456}
        with mock.patch.object(pe, "sha256_file", return_value="6" * 64):
            binding = pe._bindings(p, pe.ARMS[0])
        self.assertEqual(binding["source_model_binding"]["seed"], 0)
        self.assertEqual(binding["dataset_producer_proof"]["producer_seed"], 1)
        self.assertEqual(binding["dataset_producer_proof"]["scope"], "shared data only")
        self.assertNotIn("source_cache_manifest_sha256", binding)



if __name__ == "__main__":
    unittest.main()
