"""Synthetic invariants for the read-only SW0121 training-credit probe."""
import numpy as np
import unittest
import torch
from pathlib import Path

from collaborative_test.SW_0121_competitive_assignment import diagnose_training_credit as diag


class TrainingCreditDiagnosticTests(unittest.TestCase):
    def test_production_rho_product_and_negative_factor_blocking(self):
        components = torch.zeros((1, 4, 3, 36), dtype=torch.float32)
        pattern = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        for d in (0, 2, 3):
            components[0, d, 0, 32:] = pattern
            components[0, d, 1, 32:] = -pattern if d == 0 else pattern
            components[0, d, 2, 32:] = pattern
        # One flat component trace must be counted separately from negative rho.
        components[0, 1, 0, 32:] = pattern
        components[0, 1, 1, 32:] = pattern
        components[0, 1, 2, 32:] = 0

        qsummary = diag.pearson_path_summary(components)
        rho = diag._component_rhos(components)
        q = diag.spike_synchrony_affinity(components.mean(dim=1), components=components,
                                           settle=diag.pilot.SETTLE, affinity_mode="spike")
        paths = diag.product_path_summary(rho, torch.ones_like(q))
        self.assertEqual(qsummary["production_q_reconstruction_max_abs"], 0.0)
        self.assertGreater(qsummary["strictly_negative_rho_fraction_any_component_offdiag"], 0)
        self.assertAlmostEqual(qsummary["degenerate_trace_fraction_by_component"][1], 1 / 3, places=6)
        self.assertGreater(paths["strictly_negative_rho_blocks_all_four_fraction_offdiag"], 0)
        self.assertGreater(paths["abs_dC_dQ_mass_fraction_on_all_four_blocked_pairs"], 0)
        self.assertEqual(qsummary["q_exact_zero_fraction_offdiag"], float((q[..., ~torch.eye(3, dtype=torch.bool)] == 0).float().mean()))

    def test_warmup_head_must_match_candidate_manifest_hash(self):
        path = diag.pilot.HERE / ".__diagnostic_test_warm_head"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite test sentinel: {path}")
        path.write_bytes(b"reviewed warmup head")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        manifest = {"head_warmup_sha256": diag.sha(path)}
        self.assertTrue(diag.validate_warm_head_binding(manifest, path))
        manifest["head_warmup_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            diag.validate_warm_head_binding(manifest, path)

    def test_fixed_hard_partition_comparison_includes_background(self):
        groups = [[(1, 2)]]
        self.assertEqual(diag.coassignment_agreement(groups, groups), [1.0])
        # A foreground patch reassigned to production background changes the
        # full BG-inclusive coassignment partition.
        self.assertLess(diag.coassignment_agreement(groups, [[]])[0], 1.0)


if __name__ == "__main__":
    unittest.main()
