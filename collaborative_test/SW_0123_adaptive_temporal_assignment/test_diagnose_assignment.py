import unittest

import numpy as np
import torch
from unittest.mock import patch

from diagnose_assignment import _diagnostic_training_batch, batch_diagnostics
from model import image_from_patches


class AssignmentDiagnosticTests(unittest.TestCase):
    def test_trained_encoder_batch_does_not_enforce_source_gamma_equality(self):
        with patch("diagnose_assignment.run._batch_data", return_value=("rows", "images", "rgb", "gamma")) as call:
            result = _diagnostic_training_batch(0, "ids", 0, "gamma-cache", "rgb-cache",
                                                "encoder", "patcher", "mean", "std", "clip", "cpu")
        self.assertEqual(result, ("rows", "images", "rgb", "gamma"))
        self.assertIs(call.call_args.kwargs["check_source_cache"], False)

    def test_native_rgb_assignment_fixture_reports_counts_and_reconstruction_controls(self):
        torch.manual_seed(81)
        batch = 2
        probability = torch.softmax(torch.randn(batch, 256, 11), dim=-1)
        slot_embedding = torch.randn(batch, 11, 64)
        decoded = torch.sigmoid(torch.randn(batch, 11, 256, 8, 8, 3))
        mixed = torch.einsum("bnk,bknhwc->bnhwc", probability, decoded)
        rgb = torch.rand(batch, 3, 128, 128)
        out = {"assignment": probability, "slot_embedding": slot_embedding,
               "decoded_slot_patches": decoded,
               "reconstructed_image": image_from_patches(mixed)}
        result = batch_diagnostics(out, rgb, torch.arange(255, -1, -1))
        self.assertEqual(len(result["hard_argmax_slot_patch_counts_per_image"]), batch)
        self.assertTrue(all(sum(row) == 256 for row in result["hard_argmax_slot_patch_counts_per_image"]))
        self.assertEqual(sum(result["readout_label_counts_batch"].values()), batch * 256)
        self.assertEqual(len(result["slot_mean_probability"]), 11)
        self.assertEqual(result["pairwise_slot_latent_l2"]["min"] >= 0, True)
        for key in ("normalized_rgb_reconstruction_loss",
                    "fixed_row_shuffle_reconstruction_loss_seed12301",
                    "image_mean_assignment_reconstruction_loss",
                    "shuffle_minus_real_loss", "mean_assignment_minus_real_loss"):
            self.assertTrue(torch.isfinite(torch.tensor(result[key])))

    def test_nonfinite_assignment_is_rejected(self):
        probability = torch.full((1, 256, 11), 1 / 11)
        probability[0, 0, 0] = float("nan")
        out = {"assignment": probability, "slot_embedding": torch.zeros(1, 11, 64),
               "decoded_slot_patches": torch.zeros(1, 11, 256, 8, 8, 3),
               "reconstructed_image": torch.zeros(1, 3, 128, 128)}
        with self.assertRaises(FloatingPointError):
            batch_diagnostics(out, torch.zeros(1, 3, 128, 128), torch.arange(256))


if __name__ == "__main__":
    unittest.main()
