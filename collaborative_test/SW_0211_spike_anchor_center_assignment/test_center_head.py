import unittest
import numpy as np
import torch
from .center_head import CenterHead,targets,decode,normalization,loss
from . import training

class CenterTests(unittest.TestCase):
    def labels(self):
        y=np.zeros((16,16),np.int64);y[1:4,1:4]=1;y[7:9,1:4]=1;y[10:14,10:14]=2;return y

    def test_targets_balance_each_object_and_project_visible_centers(self):
        y=self.labels();t=targets(y)
        self.assertAlmostEqual(float(t['foreground_weights'].sum()),1.,places=6)
        self.assertAlmostEqual(float(t['offset_weights'][y==1].sum()),.5,places=6)
        self.assertAlmostEqual(float(t['offset_weights'][y==2].sum()),.5,places=6)
        self.assertEqual(int((t['heat']==1).sum()),2)
        for cy,cx in t['centers']:self.assertGreater(y[cy,cx],0)

    def test_disconnected_parts_can_share_one_instance_without_losing_foreground(self):
        y=self.labels();t=targets(y);heat=np.clip(t['heat'],1e-5,1-1e-5)
        source=np.where(y>0,99,0).astype(np.int64);source[5,5]=7
        labels,meta=decode(np.log(heat/(1-heat)),t['offset'],source)
        self.assertTrue(np.array_equal(labels>0,source>0));self.assertEqual(meta['predicted_objects'],2)
        self.assertEqual(np.unique(labels[y==1]).size,1);self.assertEqual(np.unique(labels[y==2]).size,1)
        self.assertNotEqual(labels[y==1][0],labels[y==2][0])
        renamed=np.where(source>0,12345,0)
        self.assertTrue(np.array_equal(labels,decode(np.log(heat/(1-heat)),t['offset'],renamed)[0]))

    def test_fallback_and_empty_foreground_are_safe(self):
        y=self.labels();labels,meta=decode(np.full(y.shape,-20.),np.zeros((2,*y.shape)),y)
        self.assertTrue(meta['fallback']);self.assertEqual(meta['predicted_objects'],1);self.assertTrue(np.array_equal(labels>0,y>0))
        empty,meta=decode(np.zeros(y.shape),np.zeros((2,*y.shape)),np.zeros_like(y))
        self.assertFalse(empty.any());self.assertEqual(meta['predicted_objects'],0)

    def test_TRAIN_only_normalization_floor_and_head_size(self):
        features=np.ones((2,268,8,8),np.float32);mean,scale=normalization(features)
        self.assertTrue(np.array_equal(mean,np.ones_like(mean)));self.assertTrue(np.allclose(scale,1e-4))
        head=CenterHead();self.assertEqual(sum(p.numel() for p in head.parameters()),46052)
        result=head(torch.from_numpy(features));self.assertEqual(result['offset'].shape,(2,2,8,8));self.assertTrue(torch.isfinite(result['heat']).all())

    def test_semantically_correct_outputs_have_lower_TRAIN_loss(self):
        y=self.labels();t=training.prepare_targets(np.stack([y]));eps=1e-4
        fg=t['foreground'].clamp(eps,1-eps);h=t['heat'].clamp(eps,1-eps)
        good=dict(foreground=torch.logit(fg),heat=torch.logit(h),offset=t['offset'])
        bad=dict(foreground=-good['foreground'],heat=-good['heat'],offset=-good['offset'])
        self.assertLess(float(loss(good,t)[0]),float(loss(bad,t)[0]))

    def test_nonfinite_or_invalid_decoder_fields_rejected(self):
        y=self.labels()
        with self.assertRaises(ValueError):decode(np.full(y.shape,np.nan),np.zeros((2,*y.shape)),y)
        with self.assertRaises(ValueError):decode(np.zeros(y.shape),np.zeros((2,*y.shape)),y,threshold=0)
