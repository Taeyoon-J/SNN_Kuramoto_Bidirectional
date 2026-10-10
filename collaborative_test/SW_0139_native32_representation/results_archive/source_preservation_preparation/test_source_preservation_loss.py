"""Focused CPU tests for the SW0139 source-Q anchor contract."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_preservation_loss as anchor


def _sha(char: str) -> str:
    return char * 64


def _cache_manifest(ids=None, rows=None):
    ids = list(range(100, 108)) if ids is None else list(ids)
    rows = list(range(200, 208)) if rows is None else list(rows)
    groups = [
        {"start": start, "stop": start + 4,
         "image_ids": ids[start:start + 4], "pool_rows": rows[start:start + 4]}
        for start in range(0, len(ids), 4)
    ]
    return {
        "schema": "sw0139_source_q_teacher_v1", "status": "complete",
        "ground_truth_used": False, "dtype": "float32",
        "q_shape": [len(ids), 1024, 1024], "steps": 1024, "settle": 512,
        "affinity_mode": "spike", "patch_order": "native32_row_major",
        "batch_size": 4, "image_ids": ids, "pool_rows": rows,
        "image_ids_sha256": anchor.image_ids_sha256(ids),
        "pool_rows_sha256": anchor.image_ids_sha256(rows), "groups": groups,
        "source_hashes": {
            "source_core_sha256": _sha("a"), "source_manifest_sha256": _sha("b"),
            "encoder_sha256": _sha("e"), "feature_statistics_sha256": _sha("f"),
            "rgb_cache_manifest_sha256": _sha("1"), "implementation_fingerprint_sha256": _sha("2"),
        },
        "cache_sha256": _sha("c"), "chunk_sha256": [_sha("d") for _ in groups],
    }


def _lambda_fixture():
    ids = list(range(1000, 1064))
    return {
        "experiment": "SW0139_source_q_anchor_calibration", "seed": 0,
        "probe_stage": "after_first_C_only_update", "probe_optimizer_updates": 1,
        "lambda_c_source_seed": 0, "lambda_c_value": 3.5,
        "calibration_logical_batches": 4, "logical_batch_size": 16, "microbatch_size": 4,
        "fresh_source_start": True, "fresh_adam_start": True, "probe_discarded": True,
        "actual_arms_restart_fresh": True, "source_unchanged": True, "ground_truth_used": False,
        "source_core_sha256": _sha("a"), "source_manifest_sha256": _sha("b"),
        "source_training_ids_sha256": _sha("c"), "probe_start_state_sha256": _sha("d"),
        "probe_end_state_sha256": _sha("e"), "probe_optimizer_state_sha256": _sha("f"),
        "lambda_c_source_record_sha256": _sha("7"),
        "calibration_image_ids": ids,
        "calibration_image_ids_sha256": anchor.image_ids_sha256(ids),
    }


class SourcePreservationLossTests(unittest.TestCase):
    def test_offdiagonal_known_value_identity_and_live_student_gradient(self):
        teacher = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        student = teacher.clone().requires_grad_(True)
        with torch.no_grad():
            teacher[0].fill_diagonal_(1.0)
            student[0].fill_diagonal_(1.0)
            student[0, 3, 9] = 0.5
            student[0, 9, 3] = 0.5
        teacher_before, student_before = teacher.clone(), student.detach().clone()
        loss = anchor.source_q_anchor(student, teacher)
        expected = 2.0 * 0.25 / (1024 * 1023)
        self.assertAlmostEqual(float(loss.detach()), expected, places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(teacher, teacher_before))
        self.assertTrue(torch.equal(student.detach(), student_before))

        identical = teacher.clone().requires_grad_(True)
        zero = anchor.source_q_anchor(identical, teacher)
        self.assertEqual(float(zero.detach()), 0.0)
        zero.backward()
        self.assertTrue(torch.isfinite(identical.grad).all())
        self.assertEqual(float(identical.grad.abs().sum()), 0.0)

    def test_rejects_wrong_dtype_shape_range_asymmetry_and_grad_teacher(self):
        good = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        student = good.clone().requires_grad_(True)
        with self.assertRaisesRegex(TypeError, "float32"):
            anchor.source_q_anchor(student.double(), good)
        with self.assertRaisesRegex(ValueError, "1024,1024"):
            anchor.source_q_anchor(torch.zeros(1, 32, 32, requires_grad=True), good)
        out_of_range = good.clone()
        out_of_range[0, 0, 1] = 1.1
        out_of_range[0, 1, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "range"):
            anchor.source_q_anchor(out_of_range.requires_grad_(), good)
        nonsymmetric = good.clone()
        nonsymmetric[0, 0, 1] = 0.5
        with self.assertRaisesRegex(ValueError, "symmetric"):
            anchor.source_q_anchor(nonsymmetric.requires_grad_(), good)
        with self.assertRaisesRegex(ValueError, "requires_grad"):
            anchor.source_q_anchor(student, good.clone().requires_grad_())

    def test_cache_identity_order_grouping_and_source_hashes_are_enforced(self):
        ids, rows = list(range(100, 108)), list(range(200, 208))
        expected = {
            "source_core_sha256": _sha("a"), "source_manifest_sha256": _sha("b"),
            "encoder_sha256": _sha("e"), "feature_statistics_sha256": _sha("f"),
            "rgb_cache_manifest_sha256": _sha("1"), "implementation_fingerprint_sha256": _sha("2"),
        }
        cache = _cache_manifest(ids, rows)
        anchor.validate_teacher_cache_manifest(cache, expected_image_ids=ids,
                                               expected_pool_rows=rows,
                                               expected_source_hashes=expected,
                                               expected_cache_sha256=_sha("c"),
                                               expected_chunk_sha256=cache["chunk_sha256"])
        variants = []
        wrong_ids = copy.deepcopy(cache)
        wrong_ids["image_ids"][0], wrong_ids["image_ids"][1] = wrong_ids["image_ids"][1], wrong_ids["image_ids"][0]
        variants.append((wrong_ids, "image IDs or pool-row"))
        wrong_rows = copy.deepcopy(cache)
        wrong_rows["pool_rows"][0], wrong_rows["pool_rows"][1] = wrong_rows["pool_rows"][1], wrong_rows["pool_rows"][0]
        variants.append((wrong_rows, "image IDs or pool-row"))
        wrong_groups = copy.deepcopy(cache)
        wrong_groups["groups"][0]["stop"] = 3
        variants.append((wrong_groups, "B4 grouping"))
        wrong_source = copy.deepcopy(cache)
        wrong_source["source_hashes"]["source_core_sha256"] = _sha("9")
        variants.append((wrong_source, "source hash mismatch"))
        wrong_batch = copy.deepcopy(cache)
        wrong_batch["batch_size"] = 1
        variants.append((wrong_batch, "B4 rollout grouping"))
        wrong_rows_sha = copy.deepcopy(cache)
        wrong_rows_sha["pool_rows_sha256"] = _sha("9")
        variants.append((wrong_rows_sha, "ordered ID/row digest"))
        wrong_chunk_sha = copy.deepcopy(cache)
        wrong_chunk_sha["chunk_sha256"][0] = _sha("9")
        variants.append((wrong_chunk_sha, "chunk hashes"))
        for bad, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                anchor.validate_teacher_cache_manifest(bad, expected_image_ids=ids,
                                                       expected_pool_rows=rows,
                                                       expected_source_hashes=expected,
                                                       expected_cache_sha256=_sha("c"),
                                                       expected_chunk_sha256=cache["chunk_sha256"])

    def test_four_batch_median_calibration_and_no_epsilon_zero_rule(self):
        metadata = _lambda_fixture()
        ids = metadata["calibration_image_ids"]
        source_hashes = {key: metadata[key] for key in
                         ("source_core_sha256", "source_manifest_sha256", "source_training_ids_sha256")}
        rows = [
            {"logical_batch": i, "image_ids": ids[i * 16:(i + 1) * 16],
             "weighted_c_gradient_norm": c, "anchor_gradient_norm": a}
            for i, (c, a) in enumerate(((2, 1), (4, 2), (6, 2), (8, 4)))
        ]
        calibrated = anchor.calibrate_anchor_lambda(
            rows, metadata, expected_calibration_image_ids=ids,
            expected_source_hashes=source_hashes,
            expected_lambda_c_record_sha256=_sha("7"))
        self.assertEqual(calibrated["lambda_A"], 2.0)
        self.assertEqual(calibrated["source_seed"], 0)
        self.assertEqual(calibrated["lambda_c_value"], 3.5)
        self.assertFalse(calibrated["ground_truth_used"])
        self.assertEqual([r["ratio"] for r in calibrated["calibration_batches"]], [2, 2, 3, 2])
        zero = copy.deepcopy(rows)
        zero[2]["anchor_gradient_norm"] = 0.0
        with self.assertRaisesRegex(FloatingPointError, "no epsilon"):
            anchor.calibrate_anchor_lambda(
                zero, metadata, expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))

    def test_calibration_rejects_nonfresh_or_misbound_probe(self):
        ids = _lambda_fixture()["calibration_image_ids"]
        source_hashes = {key: _lambda_fixture()[key] for key in
                         ("source_core_sha256", "source_manifest_sha256", "source_training_ids_sha256")}
        rows = [{"logical_batch": i, "image_ids": ids[i * 16:(i + 1) * 16],
                 "weighted_c_gradient_norm": 2.0, "anchor_gradient_norm": 1.0}
                for i in range(4)]
        bad = _lambda_fixture()
        bad["fresh_adam_start"] = False
        with self.assertRaisesRegex(ValueError, "probe state"):
            anchor.calibrate_anchor_lambda(
                rows, bad, expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))
        bad = _lambda_fixture()
        bad["source_core_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            anchor.calibrate_anchor_lambda(
                rows, bad, expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))
        bad = _lambda_fixture()
        bad["source_core_sha256"] = _sha("9")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            anchor.calibrate_anchor_lambda(
                rows, bad, expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))
        bad = _lambda_fixture()
        reordered = list(reversed(ids))
        bad["calibration_image_ids"] = reordered
        bad["calibration_image_ids_sha256"] = anchor.image_ids_sha256(reordered)
        with self.assertRaisesRegex(ValueError, "registered source0 first64"):
            anchor.calibrate_anchor_lambda(
                rows, bad, expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))
        wrong_batch = copy.deepcopy(rows)
        wrong_batch[1]["image_ids"] = list(wrong_batch[1]["image_ids"])
        wrong_batch[1]["image_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "batch IDs/order"):
            anchor.calibrate_anchor_lambda(
                wrong_batch, _lambda_fixture(), expected_calibration_image_ids=ids,
                expected_source_hashes=source_hashes,
                expected_lambda_c_record_sha256=_sha("7"))


if __name__ == "__main__":
    unittest.main()
