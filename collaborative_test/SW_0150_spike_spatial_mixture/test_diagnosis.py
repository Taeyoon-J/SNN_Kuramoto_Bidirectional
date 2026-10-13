"""CPU checks for the fixed, GT-free SW0150 assignment diagnostic."""
from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from unittest import mock

from collaborative_test.SW_0150_spike_spatial_mixture import diagnosis
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import (
    gaussian_mixture_nll, make_decoder,
)


class ColorBinDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _rgb_two_bins(batch=1):
        rgb = np.zeros((batch, 128, 128, 3), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                color = (0, 0, 0) if x < 16 else (192, 64, 32)
                rgb[:, y * 4:(y + 1) * 4, x * 4:(x + 1) * 4] = color
        return rgb

    @staticmethod
    def _unique_stochastic_rows(batch=1):
        weight = torch.arange(1024, dtype=torch.float32) / 1024.0
        return torch.stack((weight, 1.0 - weight), dim=-1).unsqueeze(0).repeat(batch, 1, 1)

    def test_permutation_is_deterministic_and_never_crosses_joint_rgb_bins(self):
        rgb = self._rgb_two_bins()
        p = self._unique_stochastic_rows()
        first, stats = diagnosis.permute_within_color_bins(p, rgb, [1320])
        second, stats2 = diagnosis.permute_within_color_bins(p, rgb, [1320])
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(stats, stats2)
        bins = diagnosis._patch_color_bins(rgb).reshape(-1)
        source_index = torch.round(first[0, :, 0] * 1024).to(torch.long).numpy()
        self.assertTrue(np.array_equal(bins, bins[source_index]))
        self.assertEqual(stats[0]["occupied_multrow_bin_count"], 2)
        self.assertEqual(stats[0]["eligible_rows"], 1024)
        self.assertGreater(stats[0]["permuted_row_positions"], 0)
        self.assertGreater(stats[0]["changed_assignment_rows"], 0)
        self.assertAlmostEqual(stats[0]["changed_row_fraction"],
                               stats[0]["changed_assignment_rows"] / 1024)

    def test_identical_rows_can_make_index_motion_but_no_assignment_motion(self):
        rgb = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        p = torch.zeros((1, 1024, 2), dtype=torch.float32)
        p[..., 0] = 1.0
        permuted, rows = diagnosis.permute_within_color_bins(p, rgb, [1320])
        self.assertTrue(torch.equal(permuted, p))
        self.assertEqual(rows[0]["eligible_rows"], 1024)
        self.assertEqual(rows[0]["changed_assignment_rows"], 0)
        self.assertEqual(rows[0]["changed_row_fraction"], 0.0)

    def test_background_only_and_zero_padded_slots_remain_valid(self):
        rgb = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        p = torch.zeros((2, 1024, 3), dtype=torch.float32)
        p[0, :, 0] = 1.0  # K=1-equivalent background-only image, padded to batch K=3.
        weight = torch.arange(1024, dtype=torch.float32) / 1024.0
        p[1, :, 0] = weight
        p[1, :, 1] = 1.0 - weight
        permuted, stats = diagnosis.permute_within_color_bins(p, rgb, [1320, 1321])
        self.assertTrue(torch.equal(permuted[0], p[0]))
        self.assertEqual(stats[0]["changed_assignment_rows"], 0)
        self.assertTrue(torch.equal(permuted[:, :, 2], torch.zeros_like(permuted[:, :, 2])))
        self.assertGreater(stats[1]["changed_assignment_rows"], 0)
        gamma = torch.zeros((2, 1024, 8), dtype=torch.float32)
        decoder = make_decoder(8, seed=150001)
        loss = gaussian_mixture_nll(gamma, permuted, torch.from_numpy(rgb), decoder,
                                    sigma2=0.02)
        self.assertTrue(torch.isfinite(loss))

    def test_bad_shapes_duplicate_ids_and_nonstochastic_rows_are_rejected(self):
        rgb = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        p = self._unique_stochastic_rows()
        with self.assertRaises(ValueError):
            diagnosis.permute_within_color_bins(p[:, :100], rgb, [1320])
        with self.assertRaises(ValueError):
            diagnosis.permute_within_color_bins(p.repeat(2, 1, 1), rgb.repeat(2, 0), [1320, 1320])
        broken = p.clone()
        broken[0, 0, 0] += 0.1
        with self.assertRaises(ValueError):
            diagnosis.permute_within_color_bins(broken, rgb, [1320])
        with self.assertRaises(ValueError):
            diagnosis.permute_within_color_bins(p, rgb.astype(np.float32), [1320])

    def test_registered_gaussian_nll_original_repeat_and_intervention_are_finite(self):
        torch.manual_seed(15002)
        rgb = self._rgb_two_bins()
        p = self._unique_stochastic_rows()
        permuted, _ = diagnosis.permute_within_color_bins(p, rgb, [1320])
        gamma = torch.randn((1, 1024, 8), dtype=torch.float32)
        target = torch.from_numpy(rgb)
        decoder = make_decoder(8, seed=150001)
        original_nll = gaussian_mixture_nll(gamma, p, target, decoder, sigma2=0.02)
        repeated_nll = gaussian_mixture_nll(gamma, p.clone(), target, decoder, sigma2=0.02)
        permuted_nll = gaussian_mixture_nll(gamma, permuted, target, decoder, sigma2=0.02)
        self.assertTrue(torch.equal(original_nll, repeated_nll))
        self.assertTrue(torch.isfinite(original_nll))
        self.assertTrue(torch.isfinite(permuted_nll))
        self.assertTrue(torch.isfinite(permuted_nll - original_nll))

    def test_pair_lineage_requires_same_validated_source_preflight_and_warm(self):
        admission = {
            "report_sha256": "preflight", "warm_sha256": "warm",
            "source": {"core_sha256": "core", "manifest_path": Path("manifest"),
                       "ids": [10, 11]},
            "assets": {"train_cache_sha256": "rgb"},
            "report": {"arms": {name: {"initial_parameter_group_sha256": {"same": "init"}}
                                  for name in ("encoder_graph", "joint")}},
        }
        manifests = {}
        for name in ("encoder_graph", "joint"):
            manifests[name] = {
                "manifest": {
                    "preflight_sha256": "preflight", "preflight_warm_sha256": "warm",
                    "source_core_sha256": "core", "source_manifest_sha256": "manifest-sha",
                    "source_training_ids_sha256": "ids-sha", "rgb_asset_validation": {"train_cache_sha256": "rgb"},
                    "implementation_sha256": "impl",
                    "bindings": {"preflight_sha256": "preflight",
                                 "initial_parameter_sha256": {"same": "init"}},
                }
            }
        with mock.patch.object(diagnosis.train, "sha", return_value="manifest-sha"), \
             mock.patch.object(diagnosis.train, "ids_sha", return_value="ids-sha"), \
             mock.patch.object(diagnosis.train, "implementation_sha", return_value="impl"):
            self.assertTrue(diagnosis._validate_pair_lineage(admission, manifests))
            changed = {key: dict(value) for key, value in manifests.items()}
            changed["joint"] = {"manifest": dict(manifests["joint"]["manifest"])}
            changed["joint"]["manifest"]["preflight_warm_sha256"] = "other"
            with self.assertRaises(ValueError):
                diagnosis._validate_pair_lineage(admission, changed)

    def test_declared_protocol_pins_fixed_scope_and_rollout(self):
        protocol = json.loads(diagnosis.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["data"]["image_count"], 64)
        self.assertIs(protocol["data"]["ground_truth_used"], False)
        self.assertEqual(protocol["intervention"]["seed"], diagnosis.PERMUTATION_SEED)
        self.assertEqual(protocol["rollout"]["steps"], 1024)
        self.assertEqual(protocol["rollout"]["live_tail_steps"], 64)

    def test_full_arm_diagnostic_pads_variable_microbatch_k_with_no_nll_effect(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

        wrapped = torch.nn.Module()
        wrapped.spatial_decoder = make_decoder(8, seed=150001)
        encoder = TinyEncoder()
        foundation = SimpleNamespace(wrapped=wrapped, encoder=encoder)
        source = {"ids": list(range(1320, 1384)), "rows": list(range(64))}
        rgb_cache = np.zeros((64, 128, 128, 3), dtype=np.uint8)
        calls = []
        first_gamma = None
        first_probability = None

        def fake_forward(_foundation, rgb, *, live_tail_steps):
            nonlocal first_gamma, first_probability
            self.assertEqual(tuple(rgb.shape), (4, 128, 128, 3))
            self.assertEqual(live_tail_steps, diagnosis.train.TAIL)
            index = len(calls)
            calls.append(index)
            gamma = torch.zeros((4, 1024, 8), dtype=torch.float32,
                                device=rgb.device)
            slots = (1, 3, 2, 1)[index % 4]
            if slots == 1:
                probability = torch.zeros((4, 1024, 1), device=rgb.device)
                probability[..., 0] = 1.0
            else:
                weight = torch.arange(1024, dtype=torch.float32,
                                      device=rgb.device) / 1024.0
                probability = torch.zeros((4, 1024, slots), device=rgb.device)
                probability[..., 0] = weight
                probability[..., 1] = 1.0 - weight
                # For K=3, the third native column is legitimately padded.
            if index == 0:
                first_gamma = gamma[0:1].detach().clone()
                first_probability = probability[0:1].detach().clone()
            return {"gamma_content": gamma, "p": probability,
                    "live_tail_steps": live_tail_steps}

        with mock.patch.object(diagnosis.integration, "forward_assignment",
                               side_effect=fake_forward):
            report = diagnosis._diagnose_arm(
                "encoder_graph", foundation=foundation, source=source,
                admission={"sigma2": 0.02}, rgb_cache=rgb_cache)

        self.assertEqual(len(calls), 16)
        self.assertEqual(report["rollouts"], 64)
        self.assertEqual(len(report["per_image"]), 64)
        self.assertEqual(report["model_state_sha256_before"],
                         report["model_state_sha256_after"])
        # First chunk is K=1/background-only while later chunks make global K=3.
        row0 = report["per_image"][0]
        self.assertEqual(row0["native_microbatch_slots"], 1)
        self.assertEqual(row0["padded_slots"], 3)
        self.assertEqual(row0["nll_original_repeat_delta"], 0.0)
        self.assertEqual(row0["paired_delta_permuted_minus_original"], 0.0)
        target = torch.from_numpy(rgb_cache[:1])
        direct_k1 = gaussian_mixture_nll(
            first_gamma, first_probability, target, wrapped.spatial_decoder,
            sigma2=0.02)
        padded = torch.nn.functional.pad(first_probability, (0, 2), value=0.0)
        direct_padded = gaussian_mixture_nll(
            first_gamma, padded, target, wrapped.spatial_decoder, sigma2=0.02)
        self.assertTrue(torch.equal(direct_k1, direct_padded))
        self.assertEqual(row0["nll_original"], float(direct_padded.item()))

    def test_fingerprint_inherits_registered_native_rollout_dependency(self):
        fingerprint = diagnosis.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0134_native_spike_binding/rollout.py",
                      fingerprint)


if __name__ == "__main__":
    unittest.main()
