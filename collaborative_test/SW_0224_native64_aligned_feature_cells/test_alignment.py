import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from . import alignment
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer


class AlignmentTests(unittest.TestCase):
    def found(self):
        encoder=torch.nn.Conv2d(3,8,3,bias=False)
        with torch.no_grad():encoder.weight.zero_();encoder.weight[:,0,1,1]=1.
        return SimpleNamespace(encoder=encoder,feature_mean=np.zeros((1,8,1,1),np.float32),feature_std=np.ones((1,8,1,1),np.float32),feature_clip=3.,patcher=FeaturePatchGammaInitializer(grid_size=64))

    def test_actual_adaptive_pool_geometry_and_exact_aligned_interiors(self):
        g=alignment.geometry();self.assertEqual(g['legacy_window_width_counts'],{'2':4,'3':60});self.assertEqual(g['legacy_overlapping_adjacent_cells_per_axis'],62)
        self.assertEqual(g['legacy_mean_center_abs_error_pixels'],.5);self.assertEqual(g['aligned_mean_center_abs_error_pixels'],.015625);self.assertEqual(g['aligned_interior_max_center_error_pixels'],0.)

    def test_real_valid_conv_feature_centers_match_original_pixel_cells_inside(self):
        found=self.found();rgb=torch.zeros((1,128,128,3),dtype=torch.uint8);rgb[0,:,:,0]=torch.arange(128,dtype=torch.uint8)[None,:]
        original=rgb.clone();weights=found.encoder.weight.detach().clone();gamma=alignment.encode(found,rgb).view(1,8,64,64)
        expected=F.avg_pool2d(rgb.permute(0,3,1,2).float()/255.,2,2)[:,0]
        torch.testing.assert_close(gamma[:,0,1:-1,1:-1],expected[:,1:-1,1:-1],rtol=0,atol=1e-7)
        torch.testing.assert_close(rgb,original,rtol=0,atol=0);torch.testing.assert_close(found.encoder.weight,weights,rtol=0,atol=0)
        self.assertEqual(tuple(gamma.shape),(1,8,64,64))

    def test_wrong_RGB_or_feature_geometry_rejected(self):
        found=self.found()
        with self.assertRaises(ValueError):alignment.encode(found,torch.zeros((1,128,128,3)))
        found.encoder=torch.nn.Conv2d(3,8,3,padding=1)
        with self.assertRaises(ValueError):alignment.encode(found,torch.zeros((1,128,128,3),dtype=torch.uint8))


if __name__=='__main__':unittest.main()
