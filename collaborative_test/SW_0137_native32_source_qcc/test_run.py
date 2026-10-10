from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import uuid
from contextlib import contextmanager
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0137_native32_source_qcc import run
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file


@contextmanager
def workspace_fixture():
    """Use a short workspace path; Windows AppData temp ACLs can be restricted."""
    root = ROOT / f"sw137_tmp_{uuid.uuid4().hex[:8]}"
    root.mkdir()
    try:
        yield root
    finally:
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


class SourceQCCContracts(unittest.TestCase):
    def test_sw136_train_preflight_normalizes_actual_schema_and_binds_row(self):
        actual = {"image_id": 1000, "pool_row": 0, "gamma_shape": [1, 8, 1024],
                  "qcc_shape": [1, 32, 32], "finite": True, "ground_truth_used": False}
        normalized = run._normalize_sw136_train_preflight(actual, image_id=1000, pool_row=0)
        self.assertEqual(normalized["split"], "TRAIN")
        self.assertNotIn("split", actual)
        bad = {**actual, "image_id": 1001}
        with self.assertRaises(ValueError):
            run._normalize_sw136_train_preflight(bad, image_id=1000, pool_row=0)

    def test_sw136_numpy_qcc_train_preflight_is_supported(self):
        import torch
        gamma = torch.zeros((1, 8, 1024))
        qcc = np.zeros((1, 32, 32), dtype=np.int64)
        run._validate_train_preflight_output(gamma, qcc)
        qcc[0, 0, 0] = 1
        run._validate_train_preflight_output(gamma, qcc)

    def test_saved_qcc_prediction_roundtrips_under_source_asset_and_id_contract(self):
        with workspace_fixture() as root:
            manifest = root / "source_manifest.json"
            manifest.write_text('{"registered":true}\n', encoding="utf-8")
            expected_sha = run.sw130.source97.EXPECTED_SOURCE_SHAS[0]
            ids = list(range(1320, 1640))
            asset_hashes = {"train_cache_sha256": "train", "train_manifest_sha256": "trainmeta",
                            "validation_cache_sha256": "val", "validation_manifest_sha256": "valmeta",
                            "encoder_source_sha256": "encoder", "feature_preprocessing_sha256": "stats"}
            preflight = {"split": "TRAIN", "image_id": 1320, "pool_row": 0,
                         "qcc_shape": [1, 32, 32], "finite": True,
                         "ground_truth_used": False}
            record = {"source_core_sha256": expected_sha,
                      "source_manifest_sha256": sha256_file(manifest),
                      "source_native32_state_sha256": "native-state",
                      "source_wrapped_state_sha256": "wrapped-state",
                      "encoder_sha256": "encoder", "feature_preprocessing_sha256": "stats",
                      "rgb_assets": asset_hashes, "preflight": preflight}
            predictions = np.zeros((320, 32, 32), dtype=np.int64)
            implementation = {"runner": "frozen-fixture"}
            with mock.patch.object(run, "implementation_fingerprint", return_value=implementation), \
                 mock.patch.object(run.sw130, "source_contract",
                                   return_value=(Path("core"), manifest, {}, np.arange(4096), ids, expected_sha)), \
                 mock.patch.object(run, "_validated_rgb_assets", return_value=asset_hashes):
                path = root / "seed0"
                run._write_prediction(path, 0, predictions, record)
                loaded = run.validate_prediction(path, seed=0)
                self.assertEqual(loaded["labels"].shape, (320, 32, 32))
                self.assertTrue(np.array_equal(loaded["ids"], np.asarray(ids)))
                with self.assertRaises(FileExistsError):
                    run._write_prediction(path, 0, predictions, record)

    def test_label_reader_rejects_wrong_grid_and_noninteger_dtype(self):
        with workspace_fixture() as root:
            path = root / "bad.npz"
            with path.open("xb") as stream:
                np.savez(stream, image_ids=np.arange(1320, 1640),
                         qcc_labels=np.zeros((320, 16, 16), dtype=np.int64))
            with self.assertRaises(ValueError):
                run._read_npz_labels(path)
            with path.open("wb") as stream:
                np.savez(stream, image_ids=np.arange(1320, 1640),
                         qcc_labels=np.zeros((320, 32, 32), dtype=np.float32))
            with self.assertRaises(ValueError):
                run._read_npz_labels(path)

    def test_seed1_wait_reuses_only_when_source_artifact_and_live_process_exist(self):
        with workspace_fixture() as root:
            source = root / "source97_qcc_seed1"
            source.mkdir()
            (source / "predictions.npz").write_bytes(b"pred")
            (source / "protocol.json").write_bytes(b"protocol")
            destination = root / "seed1"
            with mock.patch.object(run, "_sw136_source_path", return_value=source), \
                 mock.patch.object(run, "_sw136_source_process_live", return_value=False), \
                 mock.patch.object(run, "_reuse_sw136_seed1", return_value={"status": "prediction_complete"}) as reuse:
                result = run.wait_and_reuse_sw136_seed1(destination)
            self.assertEqual(result["status"], "prediction_complete")
            reuse.assert_called_once_with(destination)

    def test_seed1_does_not_infer_a_live_producer_from_stale_state_only(self):
        with workspace_fixture() as root:
            state_path = root / "transfer_queue_state.json"
            state_path.write_text(json.dumps({"supervisor_pid": 123456789,
                "tasks": {"sw0136_actual_joint_seed1": {"status": "running", "child_pid": 123456788}}}),
                encoding="utf-8")
            with mock.patch.object(run.sw136_eval, "ARCHIVE", root):
                self.assertFalse(run._sw136_source_process_live())


if __name__ == "__main__":
    unittest.main()
