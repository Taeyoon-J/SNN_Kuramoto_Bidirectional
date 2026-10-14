from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np  # Keep NumPy before Torch on the local Windows runtime.
import torch

from collaborative_test.SW_0161_supervised_spike_partition import cache, preflight, train


def _arm_record(initial):
    updates = []
    for index in (1, 2):
        updates.append({"update": index, "finite_loss": True, "mean_loss": 1.25,
            "finite_probabilities": True, "max_probability_row_sum_error": 2e-7,
            "gradient_finite": True,
            "gradient_norms": {"input_projection": 0.2, "gru": 0.1},
            "parameter_changed": {"input_projection": True, "gru": True},
            "whole_binder_gradient_norm": 0.3})
    return {"updates_run": 2, "initial_binder_state_sha256": initial,
        "finite_probabilities": True, "max_probability_row_sum_error": 2e-7,
        "updates": updates, "training_admission": True}


class SW161PreflightTests(unittest.TestCase):
    def setUp(self):
        # Keep nested 04-digit NPZ temp paths below Windows MAX_PATH.
        self.root = Path(preflight.HERE) / ("pf_" + uuid.uuid4().hex[:12])
        self.root.mkdir()

    def tearDown(self):
        if self.root.exists():
            resolved, parent = self.root.resolve(), Path(preflight.HERE).resolve()
            if resolved.parent != parent:
                raise AssertionError("refusing to clean outside this test-owned directory")
            shutil.rmtree(resolved)

    def test_actual_atomic_cache_rows_feed_the_shared_first16_loader(self):
        cache_root = self.root / "cache"
        ids = list(range(8000, 8016))
        rgb = [f"{index:064x}" for index in range(16)]
        bindings = {"fixture": "real cache row consumer", "image_ids": ids}
        store = cache._AtomicTraceCache(cache_root, bindings, ids, rgb)
        target_values = []
        # This exercises the production 04-digit atomic NPZ writer and reader.
        # Sixteen native [4,1024,512] float32 rows are the real ~128 MiB input.
        for index in range(16):
            trace = np.zeros((4, 1024, 512), dtype=np.float32)
            trace[:, :, index] = np.float32(index + 1)
            target = np.full((1024,), index % 4, dtype=np.int64)
            store.write_row(index, ids[index], rgb[index], trace, target)
            target_values.append(index % 4)
            del trace, target
        trace_batches, target_batches = train.load_microbatches(store, list(range(16)), "cpu")
        self.assertEqual([tuple(value.shape) for value in trace_batches],
                         [(4, 4, 1024, 512)] * 4)
        self.assertEqual([tuple(value.shape) for value in target_batches], [(4, 1024)] * 4)
        self.assertTrue(all(value.dtype == torch.float32 for value in trace_batches))
        self.assertTrue(all(value.dtype == torch.int64 for value in target_batches))
        self.assertEqual([int(value[0]) for batch in target_batches for value in batch],
                         target_values)
        self.assertTrue(torch.equal(trace_batches[0][0, :, :, 0],
                                    torch.ones((4, 1024))))
        self.assertTrue(torch.equal(trace_batches[0][0, :, :, 1],
                                    torch.zeros((4, 1024))))
        self.assertTrue(torch.equal(trace_batches[3][3, :, :, 15],
                                    torch.full((4, 1024), 16.0)))
        del trace_batches, target_batches, store

    def test_expected_cache_bindings_unpacks_actual_source_binding_tuple(self):
        class FakeRGB:
            dtype = np.dtype(np.uint8)
            shape = (70000, 128, 128, 3)

            def __getitem__(self, rows):
                self.rows = np.asarray(rows)
                return np.zeros((len(self.rows), 128, 128, 3), dtype=np.uint8)

        fake_rgb = FakeRGB()
        source = {"image_ids": list(range(512)), "rows": np.arange(512, dtype=np.int64),
                  "manifest_path": self.root / "source_manifest.json"}
        source["manifest_path"].write_text("{}", encoding="utf-8")
        expected = {"source_contract": "tuple-unpacked"}
        with (mock.patch.object(train.sw155, "source_contract", return_value=source),
              mock.patch.object(train.sw130, "validate_rgb_assets", return_value={"valid": True}),
              mock.patch.object(train.sw130, "TRAIN_RGB", self.root / "rgb.npy"),
              mock.patch.object(train.np, "load", return_value=fake_rgb),
              mock.patch.object(cache, "_current_score_dependency", return_value={"status": "complete"}),
              mock.patch.object(cache, "_source_bindings",
                                return_value=(expected, ["a" * 64] * 512)) as source_bindings):
            result = train._expected_cache_bindings()
        self.assertEqual(result, expected)
        args = source_bindings.call_args.args
        self.assertIs(args[0], source)
        self.assertEqual(args[1].shape, (512, 128, 128, 3))
        self.assertEqual(args[1].dtype, np.uint8)
        self.assertEqual(args[1][0, 0, 0, 0], 0)

    def test_strict_report_validator_binds_both_arms_and_accepts_finite_rejection(self):
        expected = {"cache_manifest_sha256": "a" * 64, "cache_ledger_sha256": "b" * 64,
            "cache_bindings_file_sha256": "c" * 64, "cache_bindings_sha256": "d" * 64,
            "cache_implementation_fingerprint": {"cache.py": "e" * 64},
            "train_implementation_fingerprint": {"train.py": "f" * 64},
            "preflight_implementation_fingerprint": {"preflight.py": "1" * 64},
            "cache_count": 512, "cache_indices": list(range(16)),
            "image_ids": list(range(200, 216)), "source_model_state_sha256": "2" * 64}
        initial = "3" * 64
        arms = {arm: _arm_record(initial) for arm in train.ARMS}
        report = {"experiment": train.EXPERIMENT, **expected, "status": "complete",
            "disposable_updates": 2, "ground_truth_used": True,
            "ground_truth_scope": "registered 16 TRAIN targets from complete source cache",
            "source_model_unchanged": True, "cache_unchanged": True,
            "cache_manifest_sha256_before": expected["cache_manifest_sha256"],
            "cache_manifest_sha256_after": expected["cache_manifest_sha256"],
            "cache_ledger_sha256_before": expected["cache_ledger_sha256"],
            "cache_ledger_sha256_after": expected["cache_ledger_sha256"],
            "cache_bindings_file_sha256_before": expected["cache_bindings_file_sha256"],
            "cache_bindings_file_sha256_after": expected["cache_bindings_file_sha256"],
            "cache_implementation_fingerprint_before": expected["cache_implementation_fingerprint"],
            "cache_implementation_fingerprint_after": expected["cache_implementation_fingerprint"],
            "train_implementation_fingerprint_before": expected["train_implementation_fingerprint"],
            "train_implementation_fingerprint_after": expected["train_implementation_fingerprint"],
            "arms": arms, "training_admission": True}
        validated = preflight.validate_report_record(report, expected=expected)
        self.assertIs(validated["training_admission"], True)

        rejected = json.loads(json.dumps(report))
        rejected["arms"][train.ARMS[0]]["updates"][1]["gradient_norms"]["gru"] = 0.0
        rejected["arms"][train.ARMS[0]]["updates"][1]["parameter_changed"]["gru"] = False
        rejected["arms"][train.ARMS[0]]["training_admission"] = False
        rejected["training_admission"] = False
        self.assertIs(preflight.validate_report_record(rejected, expected=expected)["training_admission"], False)
        changed = json.loads(json.dumps(report))
        changed["cache_manifest_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "cache_manifest_sha256"):
            preflight.validate_report_record(changed, expected=expected)

    def test_full_training_gate_rejects_missing_or_nonadmitted_receipt(self):
        with self.assertRaisesRegex(ValueError, "requires its completed actual-cache GPU"):
            train.validate_preflight_admission(self.root / "missing.json",
                cache_manifest={"bindings_sha256": "a" * 64}, cache_root=self.root)

        cache_fp = {"cache.py": "e" * 64}
        train_fp = {"train.py": "f" * 64}
        preflight_fp = {"preflight.py": "1" * 64}
        cache_manifest = {"count": 512, "bindings_sha256": "d" * 64,
                          "bindings": {"image_ids": list(range(200, 712))},
                          "source_model_state_before_sha256": "2" * 64}
        expected = {"cache_manifest_sha256": "a" * 64, "cache_ledger_sha256": "b" * 64,
            "cache_bindings_file_sha256": "c" * 64, "cache_bindings_sha256": "d" * 64,
            "cache_implementation_fingerprint": cache_fp,
            "train_implementation_fingerprint": train_fp,
            "preflight_implementation_fingerprint": preflight_fp,
            "cache_count": 512, "cache_indices": list(range(16)),
            "image_ids": list(range(200, 216)), "source_model_state_sha256": "2" * 64}
        report = {"experiment": train.EXPERIMENT, **expected, "status": "complete",
            "disposable_updates": 2, "ground_truth_used": True,
            "ground_truth_scope": "registered 16 TRAIN targets from complete source cache",
            "source_model_unchanged": True, "cache_unchanged": True,
            "cache_manifest_sha256_before": "a" * 64, "cache_manifest_sha256_after": "a" * 64,
            "cache_ledger_sha256_before": "b" * 64, "cache_ledger_sha256_after": "b" * 64,
            "cache_bindings_file_sha256_before": "c" * 64,
            "cache_bindings_file_sha256_after": "c" * 64,
            "cache_implementation_fingerprint_before": cache_fp,
            "cache_implementation_fingerprint_after": cache_fp,
            "train_implementation_fingerprint_before": train_fp,
            "train_implementation_fingerprint_after": train_fp,
            "arms": {arm: _arm_record("3" * 64) for arm in train.ARMS},
            "training_admission": True}
        accepted_path = self.root / "accepted.json"
        accepted_path.write_text(json.dumps(report), encoding="utf-8")
        sha_mock = lambda p: {"manifest.json": "a" * 64, "ledger.json": "b" * 64,
                              "bindings.json": "c" * 64}.get(Path(p).name, "9" * 64)
        with (mock.patch.object(train, "implementation_fingerprint", return_value=train_fp),
              mock.patch.object(preflight, "implementation_fingerprint", return_value=preflight_fp),
              mock.patch.object(cache, "implementation_fingerprint", return_value=cache_fp),
              mock.patch.object(train, "sha256_file", side_effect=sha_mock)):
            admitted = train.validate_preflight_admission(accepted_path,
                cache_manifest=cache_manifest, cache_root=self.root)
        self.assertEqual(admitted["report_sha256"], "9" * 64)
        self.assertEqual(admitted["preflight_implementation_fingerprint"], preflight_fp)

        report["arms"][train.ARMS[1]]["updates"][1]["gradient_norms"]["gru"] = 0.0
        report["arms"][train.ARMS[1]]["updates"][1]["parameter_changed"]["gru"] = False
        report["arms"][train.ARMS[1]]["training_admission"] = False
        report["training_admission"] = False
        path = self.root / "rejected.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        with (mock.patch.object(train, "implementation_fingerprint", return_value=train_fp),
              mock.patch.object(preflight, "implementation_fingerprint", return_value=preflight_fp),
              mock.patch.object(cache, "implementation_fingerprint", return_value=cache_fp),
              mock.patch.object(train, "sha256_file", side_effect=sha_mock)):
            with self.assertRaisesRegex(ValueError, "did not admit full training"):
                train.validate_preflight_admission(path, cache_manifest=cache_manifest,
                                                   cache_root=self.root)


if __name__ == "__main__":
    unittest.main()
