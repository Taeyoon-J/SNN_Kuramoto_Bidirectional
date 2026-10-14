from __future__ import annotations

import copy
import hashlib
import unittest

import numpy  # initialize the local BLAS runtime before importing Torch-dependent modules
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0160_end_to_end_source_encoder import model, preflight


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _record(admit=True):
    ids = list(range(4096))
    positions = list(preflight.POSITIONS)
    families = set(model.LEARNING_RATES)
    row = lambda base: {
        "update": base,
        "mean_loss": 0.7,
        "family_gradient_norms": {name: (0.4 if name == "source_base" else 0.2)
                                  for name in families},
        "family_parameter_change_l2": {name: 0.01 for name in families},
        "residual_hidden_gradient_norm": 0.1,
        "residual_hidden_parameter_change_l2": 0.01,
    }
    parity = {name: True for name in sw143.TRACE_FIELDS}
    source = {"checkpoint_path": "/registered/checkpoint.pt", "checkpoint_sha256": _sha("ckpt"),
        "manifest_path": "/registered/manifest.json", "manifest_sha256": _sha("manifest"),
        "ordered_train_ids": ids, "ordered_train_ids_sha256": preflight._ids_sha(ids),
        "source_positions": positions, "selected_image_ids": [ids[i] for i in positions],
        "rgb_asset_validation": {"cache_sha256": _sha("rgb")},
        "rgb_batch_sha256": _sha("batch"), "saved_parity_sha256": _sha("parity"),
        "saved_parity_status": "complete", "source_patch_sha256": _sha("patch"),
        "source_base_encoder_sha256": _sha("before"), "feature_preprocess_sha256": _sha("preprocess"),
        "hdf5_identity": {"path": "/canonical/train.h5", "size_bytes": 1000,
            "mtime_ns": 1, "images_shape": [100000,128,128,3],
            "masks_shape": [100000,128,128]}}
    upstream = {"seed": 1, **{key: source[key] for key in
        ("checkpoint_path", "checkpoint_sha256", "manifest_path", "manifest_sha256",
         "ordered_train_ids", "ordered_train_ids_sha256", "source_positions", "selected_image_ids",
         "rgb_asset_validation", "rgb_batch_sha256", "saved_parity_sha256", "saved_parity_status",
         "hdf5_identity")}}
    return {
        "experiment": preflight.EXPERIMENT, "status": "complete", "seed": 1,
        "implementation_fingerprint": {"preflight.py": _sha("fp")},
        "upstream": upstream,
        "source": source,
        "registered": {"arms": list(model.ARMS), "optimizer_updates_per_arm": 2,
            "preflight_images": 8, "logical_batch": 4, "microbatch": 1,
            "steps": preflight.STEPS, "settle": preflight.SETTLE, "live_tail": preflight.TAIL,
            "learning_rates": model.LEARNING_RATES,
            "ground_truth_scope": "the eight registered source TRAIN masks only"},
        "arms": {
            "frozen_base_joint": {"initial_zero_residual_parity": {"exact": True,
                "gamma_exact": True, "production_Q_exact": True, "production_QCC_exact": True,
                "trace_fields": parity}, "base_encoder_sha256_before": _sha("before"),
                "base_encoder_sha256_after": _sha("before"), "base_encoder_changed": False,
                "feature_clip_fraction_after_updates": 0.1,
                "source_feature_clip_fraction": 0.12,
                "initial_trainable_family_sha256": {name: _sha("common-" + name) for name in families},
                "patcher_sha256_before_after": [_sha("patch"), _sha("patch")],
                "feature_preprocess_sha256": _sha("preprocess"),
                "frozen_effective_geodesic_contrast": True,
                "settled_event_diagnostics_before_updates": {"settled_event_on_fraction": 0.5,
                    "settled_gate_positive_count": 100, "settled_event_on_fraction_when_gate_positive": 0.5,
                    "settled_gate_positive_event_transitions": 2,
                    "per_component_settled_event_on_fraction": [0.5] * 4},
                "settled_event_diagnostics_after_updates": {"settled_event_on_fraction": 0.5,
                    "settled_gate_positive_count": 100, "settled_event_on_fraction_when_gate_positive": 0.5,
                    "settled_gate_positive_event_transitions": 2,
                    "per_component_settled_event_on_fraction": [0.5] * 4},
                "two_updates": [row(1), row(2)]},
            "full_encoder_joint": {"initial_zero_residual_parity": {"exact": True,
                "gamma_exact": True, "production_Q_exact": True, "production_QCC_exact": True,
                "trace_fields": parity}, "base_encoder_sha256_before": _sha("before"),
                "base_encoder_sha256_after": _sha("after"), "base_encoder_changed": True,
                "feature_clip_fraction_after_updates": 0.2,
                "source_feature_clip_fraction": 0.12,
                "initial_trainable_family_sha256": {name: _sha("common-" + name) for name in families},
                "patcher_sha256_before_after": [_sha("patch"), _sha("patch")],
                "feature_preprocess_sha256": _sha("preprocess"),
                "frozen_effective_geodesic_contrast": True,
                "settled_event_diagnostics_before_updates": {"settled_event_on_fraction": 0.5,
                    "settled_gate_positive_count": 100, "settled_event_on_fraction_when_gate_positive": 0.5,
                    "settled_gate_positive_event_transitions": 2,
                    "per_component_settled_event_on_fraction": [0.5] * 4},
                "settled_event_diagnostics_after_updates": {"settled_event_on_fraction": 0.5,
                    "settled_gate_positive_count": 100, "settled_event_on_fraction_when_gate_positive": 0.5,
                    "settled_gate_positive_event_transitions": 2,
                    "per_component_settled_event_on_fraction": [0.5] * 4},
                "two_updates": [row(1), row(2)]},
        },
        "ground_truth_used": True, "ground_truth_scope": "the eight registered source TRAIN masks only",
        "validation_or_test_ground_truth_used": False,
        "training_admission": bool(admit), "admission_reasons": [],
    }


class SW0160PreflightContractTests(unittest.TestCase):
    def test_strict_report_accepts_paired_source_bound_shape(self):
        report = _record()
        self.assertTrue(preflight.validate_report_record(report))
        changed_upstream = copy.deepcopy(report)
        changed_upstream["upstream"]["checkpoint_sha256"] = _sha("other-checkpoint")
        self.assertFalse(preflight.validate_report_record(
            changed_upstream, expected_upstream=report["upstream"]))
        tampered = copy.deepcopy(report)
        tampered["source"]["selected_image_ids"][0] += 1
        self.assertFalse(preflight.validate_report_record(tampered))
        tampered = copy.deepcopy(report)
        tampered["arms"]["full_encoder_joint"]["initial_zero_residual_parity"]["trace_fields"].pop(
            next(iter(sw143.TRACE_FIELDS)))
        self.assertFalse(preflight.validate_report_record(tampered))

    def test_scientific_rejection_is_valid_and_base_change_is_required(self):
        report = _record(admit=False)
        report["source"]["source_base_encoder_sha256"] = \
            report["arms"]["full_encoder_joint"]["base_encoder_sha256_before"]
        report["arms"]["full_encoder_joint"]["base_encoder_sha256_after"] = \
            report["arms"]["full_encoder_joint"]["base_encoder_sha256_before"]
        report["arms"]["full_encoder_joint"]["base_encoder_changed"] = False
        report["training_admission"] = False
        report["admission_reasons"] = ["candidate_source_base_unchanged_after_two_updates"]
        self.assertTrue(preflight.validate_report_record(report))
        broken = copy.deepcopy(report)
        broken["training_admission"] = True
        self.assertFalse(preflight.validate_report_record(broken))

    def test_current_implementation_fingerprint_resolves_real_dependencies(self):
        fingerprint = preflight.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0155_supervised_spike_relations/train.py", fingerprint)
        self.assertIn("collaborative_test/SW_0135_native32_spike_binding/foundation.py", fingerprint)
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_actual_sw155_modal_converter_consumes_hdf_shaped_train_masks(self):
        import torch
        from collaborative_test.SW_0155_supervised_spike_relations import objective
        masks = torch.zeros((2, 128, 128), dtype=torch.uint8)
        masks[:, :, 64:] = 7
        labels = objective.modal_train_targets32(masks)
        self.assertEqual(tuple(labels.shape), (2, 1024))
        self.assertEqual(int(labels[0, 0]), 0)
        self.assertEqual(int(labels[0, -1]), 7)


if __name__ == "__main__":
    unittest.main()
