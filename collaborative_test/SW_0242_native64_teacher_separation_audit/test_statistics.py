import unittest
import numpy as np
from .statistics import describe,aggregate,zero_loss_collision_example


class StatisticsTests(unittest.TestCase):
    def test_separated_and_colliding_teacher_targets_are_distinguished(self):
        labels=np.array([[[0,1],[2,2]]]);f=np.zeros((1,4,4),dtype=np.float32);f[0,1]=1;f[0,2:]=2
        a=describe(f,labels)[0]
        self.assertEqual(a['object_balanced_accuracy'],1)
        self.assertEqual(a['nearest_center_distances'],[2.,2.])
        self.assertEqual(a['zero_within_scale_pairs'],2)
        b=describe(np.ones_like(f),labels)[0]
        self.assertEqual(b['object_balanced_accuracy'],.5)
        self.assertEqual(b['nearest_center_distances'],[0.,0.])
        self.assertFalse(aggregate([a,b])['segmentation_evaluation_score'])

    def test_object_balancing_is_not_pixel_balancing(self):
        labels=np.array([[[1,2],[2,2]]]);r=describe(np.ones((1,4,4),dtype=np.float32),labels)[0]
        self.assertEqual(r['patch_accuracy'],.25);self.assertEqual(r['object_balanced_accuracy'],.5)

    def test_actual_loss_permits_identical_two_object_frequency_targets(self):
        r=zero_loss_collision_example()
        self.assertEqual(r['actual_registered_teacher_loss'],0)
        self.assertEqual(r['distinct_foreground_object_count'],2)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):describe(np.ones((1,4,3),dtype=np.float32),np.ones((1,2,2),dtype=np.int64))


if __name__=='__main__':unittest.main()
