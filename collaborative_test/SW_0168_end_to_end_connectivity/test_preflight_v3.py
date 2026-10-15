"""Support-mask regression for corrected SW0168 admission v3."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective
from collaborative_test.SW_0168_end_to_end_connectivity import preflight_v3


class SupportMaskTests(unittest.TestCase):
    def test_both_cpu_losses_convert_qcc_component_ids_to_foreground_mask(self):
        # QCC labels include background 0 and positive component IDs 1 and 2.
        labels = np.array([[1, 1, 0, 2], [1, 0, 0, 2]], dtype=np.int32)
        target = np.array([[1, 1, 0, 2], [1, 0, 0, 2]], dtype=np.int32)
        q = torch.eye(8, dtype=torch.float32)
        q[0, 1] = q[1, 0] = .8
        q[3, 7] = q[7, 3] = .7
        foreground = labels > 0
        for arm, loss_fn in (("constrained_maximin", connectivity_objective.constrained_maximin_loss),
                             ("matched_local_edge", connectivity_objective.matched_local_edge_loss)):
            observed = preflight_v3._objective(arm, q, target, labels)
            expected = loss_fn(q, target, foreground)
            self.assertTrue(torch.isfinite(observed))
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_invalid_arm_does_not_silently_select_control(self):
        with self.assertRaises(ValueError):
            preflight_v3._objective("unregistered", torch.eye(1), np.zeros((1, 1), dtype=np.int32),
                                     np.ones((1, 1), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
