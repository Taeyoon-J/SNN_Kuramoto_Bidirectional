from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

import numpy as np
import torch

from collaborative_test.SW_0144_native32_assignment_boundary_diagnostic import run


class AssignmentBoundaryDiagnosticTests(unittest.TestCase):
    def test_mass_weighted_four_connected_fragment_fraction(self):
        labels = np.zeros((4, 6), dtype=np.int64)
        labels[0, :3] = 1
        labels[3, 0] = 1
        labels[1:3, 4:6] = 2
        row = run.four_connected_summary(labels)
        self.assertEqual(row["foreground_patch_mass"], 8)
        self.assertEqual(row["labels"][0]["component_masses"], [3, 1])
        self.assertEqual(row["labels"][0]["disconnected_fraction"], 0.25)
        self.assertEqual(row["labels"][1]["component_masses"], [4])
        self.assertEqual(row["mass_weighted_disconnected_fraction"], 0.125)

    def test_16x16_mask_tiles_map_to_4x4_patch_visibility(self):
        visible = run.visible_patch_mask(np.asarray([0, 9], dtype=np.int16))
        self.assertFalse(visible[:4, :4].any())
        self.assertFalse(visible[4:8, 4:8].any())
        self.assertTrue(visible[0, 4])
        with self.assertRaises(ValueError):
            run.visible_patch_mask(np.asarray([0, 0], dtype=np.int16))

    def test_boundary_energy_is_native_patch_seam_and_visibility_filtered(self):
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[:, 64:, :] = 255
        pairs, energy = run.adjacent_boundary_edges(rgb, np.asarray([0], dtype=np.int16))
        self.assertEqual(pairs.shape[1], 2)
        self.assertTrue(np.all((pairs[:, 0] != pairs[:, 1])))
        # The seam at grid column zero touches masked patches; the next seam
        # between constant-color regions has zero energy.
        self.assertTrue(np.isfinite(energy).all())
        vertical = np.where((pairs[:, 0] // 32 == pairs[:, 1] // 32)
                            & (pairs[:, 1] - pairs[:, 0] == 1))[0]
        self.assertGreater(len(vertical), 0)
        boundary_columns = pairs[vertical, 0] % 32
        changed = boundary_columns == 15
        self.assertTrue(np.any(changed))
        self.assertTrue(np.all(energy[vertical][changed] == 1.0))
        self.assertTrue(np.all(energy[vertical][~changed] == 0.0))

    def test_actual_spike_affinity_and_warm_label_path_are_finite(self):
        self.assertFalse(torch.cuda.is_initialized())
        generator = torch.Generator(device="cpu").manual_seed(144)
        spikes = (torch.rand((1, 4, 1024, 512), generator=generator) > 0.92).float()
        logits = torch.randn((1, 1024, 11), generator=generator)
        assignments = logits.softmax(-1)
        rgb = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        rgb[:, :, 64:, :] = 200
        rows, errors = run.batch_observations(
            spikes, rgb, np.empty((1, 0), dtype=np.int16), assignments)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["assignment"]["labels"]), 10)
        self.assertGreater(len(errors), 0)
        self.assertEqual(len(rows[0]["edges"]["q_affinity"]), len(errors))
        self.assertTrue(np.isfinite(rows[0]["edges"]["q_affinity"]).all())
        scale = float(errors.mean())
        summary = run._edge_summary(rows, scale)
        self.assertEqual(summary["per_image"][0]["visible_adjacent_edges"], len(errors))
        self.assertTrue(np.isfinite(summary["pooled_mean_absolute_difference"]))

    def test_rejects_wrong_actual_spike_or_rgb_schema(self):
        spikes = torch.zeros((1, 4, 1024, 16))
        assignments = torch.full((1, 1024, 11), 1 / 11)
        with self.assertRaisesRegex(ValueError, "cached emitted spikes"):
            run.batch_observations(spikes, np.zeros((1, 128, 128, 3), dtype=np.uint8),
                                   np.empty((1, 0), dtype=np.int16), assignments)


if __name__ == "__main__":
    unittest.main()
