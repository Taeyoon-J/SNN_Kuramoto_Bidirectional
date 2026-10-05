import unittest

import torch

from snn_kuramoto_bidirectional.loss_function import (
    UnsupervisedS2NetLoss,
    patch_pool_rgb,
)


class SlotReconstructionTests(unittest.TestCase):
    def test_rgb_target_drives_phase_gradient(self):
        torch.manual_seed(0)
        images = torch.rand(2, 3, 8, 8)
        target = patch_pool_rgb(images, (2, 2))
        theta = torch.randn(2, 6, 4, 2, requires_grad=True)
        criterion = UnsupervisedS2NetLoss(
            slot_reconstruction_weight=1.0,
            slot_num_slots=3,
            slot_temperature=0.3,
        )
        loss, parts = criterion(
            spikes=torch.zeros(2, 4, 6), theta=theta,
            recon_target=target, plv_settle=2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["slot_reconstruction"].detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.isfinite(theta.grad).all())
        self.assertGreater(float(theta.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
