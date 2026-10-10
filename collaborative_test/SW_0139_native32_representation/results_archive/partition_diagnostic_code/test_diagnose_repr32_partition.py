from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
import uuid
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnose_repr32_partition as diag


class PartitionDiagnosticTests(unittest.TestCase):
    def test_merge_split_and_predicted_background_column_are_separated(self):
        target = [0] * (32 * 32)
        prediction = [0] * (32 * 32)
        for index in (0, 1):
            target[index] = 1
        for index in (2, 3):
            target[index] = 2
        for index in (4, 5):
            target[index] = 3
        prediction[0], prediction[2] = 7, 7
        prediction[1], prediction[3] = 0, 0
        prediction[4], prediction[5] = 8, 9

        result = diag.partition_error(np.asarray(target, dtype=np.int64).reshape(32, 32),
                                      np.asarray(prediction, dtype=np.int64).reshape(32, 32))
        self.assertEqual(result["true_fg_pair_count"], 15)
        self.assertEqual(result["same_true_object_pairs_T"], 3)
        self.assertEqual(result["same_predicted_label_pairs_P"], 2)
        self.assertEqual(result["same_true_and_predicted_pairs_S"], 0)
        self.assertEqual(result["false_merge_pairs_P_minus_S"], 2)
        self.assertEqual(result["false_split_pairs_T_minus_S"], 3)
        self.assertEqual(result["predicted_background_column_merge_pairs"], 1)
        self.assertAlmostEqual(result["foreground_missed_fraction"], 2 / 6)
        self.assertAlmostEqual(result["false_merge_fraction_of_P"], 1.0)
        self.assertAlmostEqual(result["false_split_fraction_of_T"], 1.0)
        self.assertAlmostEqual(result["false_merge_fraction_of_all_true_fg_pairs"], 2 / 15)
        self.assertAlmostEqual(result["false_split_fraction_of_all_true_fg_pairs"], 3 / 15)

    def test_nonzero_label_permutation_does_not_change_partition_counts(self):
        target = [0] * (32 * 32)
        prediction = [0] * (32 * 32)
        target[:6] = [1, 1, 2, 2, 3, 3]
        prediction[:6] = [7, 7, 7, 8, 0, 0]
        permuted = [0 if value == 0 else 100 - value for value in prediction]
        first = diag.partition_error(np.asarray(target, dtype=np.int64).reshape(32, 32),
                                     np.asarray(prediction, dtype=np.int64).reshape(32, 32))
        second = diag.partition_error(np.asarray(target, dtype=np.int64).reshape(32, 32),
                                      np.asarray(permuted, dtype=np.int64).reshape(32, 32))
        for key in ("true_fg_pair_count", "same_true_object_pairs_T",
                    "same_predicted_label_pairs_P", "same_true_and_predicted_pairs_S",
                    "false_merge_pairs_P_minus_S", "false_split_pairs_T_minus_S",
                    "predicted_background_column_merge_pairs", "foreground_missed_patches"):
            self.assertEqual(first[key], second[key])

    def test_rejects_invalid_label_arrays(self):
        with self.assertRaises(ValueError):
            diag.partition_error([[1]], [[1]])
        bad = [0] * (32 * 32)
        bad[0] = -1
        with self.assertRaises(ValueError):
            diag.partition_error(np.asarray(bad, dtype=np.int64),
                                 np.zeros(32 * 32, dtype=np.int64))

    def test_repository_override_is_validated_before_import_roots_are_used(self):
        self.assertEqual(diag.resolve_repo_root({"PATCH_V2_REPO": str(diag.REPO)}),
                         diag.REPO.resolve())
        with self.assertRaises(FileNotFoundError):
            diag.resolve_repo_root({"PATCH_V2_REPO": str(HERE)})

    def test_training_root_matches_sw0139_runner_default(self):
        self.assertEqual(diag.TRAINING_ROOT, diag.sw139.run.OUT)

    def test_actual_partial_report_schema_and_hashes_are_checked_before_gt(self):
        summary_bytes = diag.SUMMARY.read_bytes()
        summary = json.loads(summary_bytes.decode("utf-8-sig"))
        self.assertEqual(summary["status"], "partial_complete")
        temp = HERE / f"_repr32_contract_{uuid.uuid4().hex}"
        temp.mkdir()
        try:
            sha_by_path = {}
            rows = {}
            labels = np.zeros((diag.COUNT, diag.GRID, diag.GRID), dtype=np.int64)
            for arm in diag.ARMS:
                protocol_path = diag.EVAL_DIR / arm / "protocol.json"
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                pred_path = temp / f"{arm}.npz"
                fake_protocol_path = temp / f"{arm}.json"
                pred_path.write_bytes(b"fixed fake prediction bytes " + arm.encode())
                fake_protocol_path.write_bytes(b"fixed fake protocol bytes " + arm.encode())
                pred_sha = protocol["prediction_sha256"]
                protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                sha_by_path[str(pred_path)] = pred_sha
                sha_by_path[str(fake_protocol_path)] = protocol_sha
                rows[arm] = {"ids": np.asarray(diag.IMAGE_IDS, dtype=np.int64),
                             "labels": labels, "protocol": protocol,
                             "prediction_sha256": pred_sha,
                             "protocol_sha256": protocol_sha,
                             "prediction_path": pred_path,
                             "protocol_path": fake_protocol_path}

            source_pred, source_protocol = temp / "source.npz", temp / "source.json"
            source_pred.write_bytes(b"source prediction")
            source_protocol.write_bytes(b"source protocol")
            source_sha = summary["source_prediction_sha256"]
            source_protocol_sha = hashlib.sha256(source_protocol.read_bytes()).hexdigest()
            sha_by_path[str(source_pred)] = source_sha
            sha_by_path[str(source_protocol)] = source_protocol_sha
            source = {"ids": np.asarray(diag.IMAGE_IDS, dtype=np.int64), "labels": labels,
                      "protocol": {}, "prediction_sha256": source_sha,
                      "protocol_sha256": source_protocol_sha, "archive_path": source_pred,
                      "protocol_path": source_protocol}

            class Mask:
                shape = (1640, 128, 128, 1)

                def __getitem__(self, slc):
                    return np.zeros((diag.COUNT, 128, 128, 1), dtype=np.uint8)

            class H5:
                def __enter__(self): return self
                def __exit__(self, *_args): return False
                def __contains__(self, key): return key == "mask"
                def __getitem__(self, key): return Mask()

            validation_kwargs = []
            def validate_arm(arm, **kwargs):
                validation_kwargs.append(kwargs)
                return rows[arm]

            with (mock.patch.object(diag.sw139, "validate_arm_prediction", side_effect=validate_arm),
                  mock.patch.object(diag.sw137, "validate_prediction", return_value=source),
                  mock.patch.object(diag.sw139.run, "sha256_file",
                                    side_effect=lambda p: sha_by_path[str(Path(p))]),
                  mock.patch.object(diag.sw139.h5py, "File", return_value=H5()) as open_gt):
                loaded = diag._read_prediction_inputs(diag.SUMMARY, temp / "unused.h5")
                self.assertEqual(loaded[0]["status"], "partial_complete")
                self.assertEqual({str(call["training_root"]) for call in validation_kwargs},
                                 {str(diag.sw139.run.OUT)})
                self.assertEqual(loaded[3].shape, (diag.COUNT, diag.GRID, diag.GRID))
                open_gt.assert_called_once()

                rows["control"]["protocol"] = dict(rows["control"]["protocol"])
                rows["control"]["protocol"]["prediction_sha256"] = "tampered"
                open_gt.reset_mock()
                with self.assertRaisesRegex(ValueError, "prediction hash differs"):
                    diag._read_prediction_inputs(diag.SUMMARY, temp / "unused.h5")
                open_gt.assert_not_called()
        finally:
            for path in temp.iterdir():
                path.unlink()
            temp.rmdir()


if __name__ == "__main__":
    unittest.main()
