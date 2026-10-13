import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0152_source_preserving_partition_equivariance.deep_encoder import SourcePreservingDeepEncoder
from collaborative_test.SW_0152_source_preserving_partition_equivariance.view_augmentation import make_views
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer


class DeepEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(150201)

    def test_zero_init_preserves_features_and_native32_gamma(self):
        base = CNNFeatureEncoder(8, 3)
        model = SourcePreservingDeepEncoder(base)
        rgb = torch.linspace(0, 1, 2 * 3 * 128 * 128, dtype=torch.float32).reshape(2, 3, 128, 128)
        expected = base(rgb)
        actual = model(rgb)
        self.assertEqual(tuple(actual.shape), (2, 8, 126, 126))
        self.assertTrue(torch.equal(actual, expected))
        patcher = FeaturePatchGammaInitializer(grid_size=32)
        self.assertTrue(torch.equal(patcher(actual), patcher(expected)))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

    def test_residual_head_has_credit_and_changes_spatial_features(self):
        model = SourcePreservingDeepEncoder(CNNFeatureEncoder(8, 3))
        rgb = torch.rand(2, 3, 128, 128)
        initial = model(rgb).detach()
        model.zero_grad(set_to_none=True)
        model(rgb).square().mean().backward()
        final = model.residual[-1]
        self.assertIsNotNone(final.weight.grad)
        self.assertTrue(torch.isfinite(final.weight.grad).all())
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.0)
        # At the exact zero head, upstream residual layers correctly have no first-step credit.
        self.assertEqual(float(model.residual[0].weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            final.weight.fill_(0.01)
        changed = model(rgb)
        self.assertFalse(torch.equal(changed, initial))
        delta = changed - initial
        self.assertGreater(float((delta[..., 0, 0] - delta[..., 20, 20]).abs().sum().detach()), 0.0)
        other = model(rgb.flip(-1))
        self.assertFalse(torch.equal(other, changed.flip(-1)))

    def test_input_contract_and_view_augmentation(self):
        model = SourcePreservingDeepEncoder(CNNFeatureEncoder(8, 3))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 127, 128))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 3, 128, 128, dtype=torch.float64))
        image = torch.zeros(128, 128, 3, dtype=torch.uint8)
        image[:, :, 0] = torch.arange(128, dtype=torch.uint8)[None, :]
        before = image.clone()
        clean1, aug1, record1 = make_views(image, source_seed=1, image_id=68304)
        _, aug2, record2 = make_views(image, source_seed=1, image_id=68304)
        _, aug_other, _ = make_views(image, source_seed=1, image_id=68305)
        self.assertTrue(torch.equal(clean1, before))
        self.assertTrue(torch.equal(aug1, aug2))
        self.assertEqual(record1, record2)
        self.assertFalse(torch.equal(aug1, aug_other))
        expected = torch.round((before.flip(1).float() / 255.0 * torch.tensor(record1["gain"]).view(1, 1, 3)
                                + torch.tensor(record1["offset"]).view(1, 1, 3)).clamp(0, 1) * 255).to(torch.uint8)
        self.assertTrue(torch.equal(aug1, expected))
        self.assertTrue((aug1 <= 255).all())


if __name__ == "__main__":
    unittest.main()
