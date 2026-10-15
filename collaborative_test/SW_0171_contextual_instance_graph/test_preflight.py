"""CPU producer/consumer contract tests for the SW0171 preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from collaborative_test.SW_0171_contextual_instance_graph import preflight


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _fixture():
    arrays = {
        "rgb": np.zeros((512, 128, 128, 3), dtype=np.uint8),
        "source_support": np.zeros((512, 32, 32), dtype=np.int32),
        "feasible_targets": np.zeros((512, 32, 32), dtype=np.int32),
        "modal_gt": np.zeros((512, 32, 32), dtype=np.int32),
        "image_ids": np.arange(10000, 10512, dtype=np.int64),
    }
    bindings = {
        "ordered_train_ids": arrays["image_ids"].tolist(),
        "source_support_sha256": _sha(arrays["source_support"].astype("<i4").tobytes()),
        "feasible_targets_sha256": _sha(arrays["feasible_targets"].astype("<i4").tobytes()),
        "train_rgb_sha256": [_sha(row.tobytes()) for row in arrays["rgb"]],
        "hdf5_path": "/registered/train.h5", "hdf5_size_bytes": 1234,
        "hdf5_mtime_ns": 98765,
    }
    manifest = {
        "experiment": "SW0171_original_source_TRAIN_input_cache", "status": "complete",
        "count": 512, "arrays_sha256": "a" * 64,
        "schema": {key: {"shape": list(shape), "dtype": dtype}
                   for key, (shape, dtype) in preflight.ARRAY_SCHEMA.items()},
        "producer_implementation": {"producer.py": "b" * 64},
        "producer_bindings": bindings,
        "validation_data_read": False, "candidate_initialization_not_loaded": True,
        "optimizer_updates": 0, "cuda_initialized": False, "TRAIN_masks_read": True,
        "modal_GT_verified_against_canonical_RGB_rows": True,
        "canonical_hdf5_path": bindings["hdf5_path"],
        "canonical_hdf5_size_bytes": bindings["hdf5_size_bytes"],
        "canonical_hdf5_mtime_ns": bindings["hdf5_mtime_ns"],
    }
    return manifest, arrays, bindings["ordered_train_ids"]


class PreflightContractTests(unittest.TestCase):
    def test_actual_producer_shaped_manifest_roundtrips_cpu_validator(self):
        manifest, arrays, ids = _fixture()
        proof = preflight.validate_dataset_cache(
            manifest, arrays, arrays_sha256="a" * 64,
            producer_fingerprint=manifest["producer_implementation"], expected_ids=ids)
        self.assertEqual(proof["count"], 512)
        self.assertEqual(proof["image_ids"], ids)
        self.assertIs(proof["validation_data_read"], False)

    def test_producer_binding_tamper_and_changed_source_order_reject(self):
        manifest, arrays, ids = _fixture()
        changed = dict(arrays)
        changed["source_support"] = arrays["source_support"].copy()
        changed["feasible_targets"] = arrays["feasible_targets"].copy()
        changed["source_support"][0, 0, 0] = 1
        changed["feasible_targets"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "source support"):
            preflight.validate_dataset_cache(manifest, changed, arrays_sha256="a" * 64,
                                             producer_fingerprint=manifest["producer_implementation"],
                                             expected_ids=ids)
        with self.assertRaisesRegex(ValueError, "canonical ordered source IDs"):
            preflight.validate_dataset_cache(manifest, arrays, arrays_sha256="a" * 64,
                                             producer_fingerprint=manifest["producer_implementation"],
                                             expected_ids=ids[::-1])

    def test_validation_and_reserve_ids_are_rejected(self):
        manifest, arrays, ids = _fixture()
        arrays = dict(arrays)
        arrays["image_ids"] = arrays["image_ids"].copy()
        arrays["image_ids"][0] = 1320
        manifest = json.loads(json.dumps(manifest))
        manifest["producer_bindings"]["ordered_train_ids"] = arrays["image_ids"].tolist()
        with self.assertRaisesRegex(ValueError, "validation/reserve"):
            preflight.validate_dataset_cache(manifest, arrays, arrays_sha256="a" * 64,
                                             producer_fingerprint=manifest["producer_implementation"],
                                             expected_ids=arrays["image_ids"])

    def test_registered_fingerprint_dependencies_resolve_to_real_files(self):
        value = preflight.implementation_fingerprint()
        self.assertTrue(value["files"])
        for relative, digest in value["files"].items():
            path = preflight.ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(len(digest), 64)

    def test_b4_mean_gradient_norm_is_over_full_gamma_tensor(self):
        gamma = torch.ones(4, 2, requires_grad=True)
        losses = gamma[:, 0]
        self.assertAlmostEqual(preflight._b4_mean_gradient_norm(losses, gamma), 0.5, places=7)

    def test_actual_objective_helper_flattens_modal32_and_keeps_zero_loss_rows(self):
        from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective
        q = torch.zeros(4, 1024, 1024, requires_grad=True)
        z = torch.nn.functional.normalize(torch.randn(4, 1024, 3), dim=-1)
        labels = np.zeros((4, 32, 32), dtype=np.int32)
        labels[:, :, 16:] = 1
        zero_target = np.zeros_like(labels)
        support = np.zeros_like(labels)
        q_loss, instance_loss = preflight._losses(
            q, z, zero_target, support, labels, connectivity_objective)
        self.assertEqual(tuple(q_loss.shape), (4,))
        self.assertEqual(tuple(instance_loss.shape), (4,))
        self.assertTrue(torch.equal(q_loss, torch.zeros_like(q_loss)))
        self.assertTrue(torch.isfinite(instance_loss).all())
        with self.assertRaisesRegex(ValueError, "flatten exactly"):
            preflight._losses(q, z, zero_target, support, labels[:, :1], connectivity_objective)

    def test_integration_vectors_are_serialized_without_scalar_cast(self):
        wrapped = type("Integration", (), {})()
        wrapped.a_d = torch.tensor([1., 2., 3., 4.])
        wrapped.a_m = torch.tensor([5., 6., 7., 8.])
        wrapped.b = torch.tensor([9., 10., 11., 12.])
        result = preflight._integration_values(wrapped)
        self.assertEqual(result["a_d"], [1., 2., 3., 4.])
        self.assertEqual(result["b"], [9., 10., 11., 12.])


if __name__ == "__main__":
    unittest.main()
