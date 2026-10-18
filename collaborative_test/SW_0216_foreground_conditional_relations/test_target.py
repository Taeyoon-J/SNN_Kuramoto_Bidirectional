import unittest
from unittest.mock import patch
import numpy as np
from .targets import foreground_targets
from collaborative_test.SW_0214_component_relation_merger.components import soft_targets


class TargetTests(unittest.TestCase):
    def test_background_dilution_changes_old_target_but_not_foreground_target(self):
        source=np.array([[1,1,2,2],[1,1,2,2]],np.int64);gt=np.array([[1,0,1,0],[0,0,0,0]],np.int64)
        ids=np.array([1,2]);pairs=np.array([[0,1]])
        self.assertAlmostEqual(float(soft_targets(source,gt,ids,pairs)[0]),1/16)
        self.assertEqual(float(foreground_targets(source,gt,ids,pairs)[0]),1.)

    def test_different_objects_and_background_only_components_are_negative(self):
        source=np.array([[1,2,3],[1,2,3]],np.int64);gt=np.array([[1,2,0],[0,0,0]],np.int64)
        np.testing.assert_array_equal(foreground_targets(source,gt,np.array([1,2,3]),np.array([[0,1],[0,2],[1,2]])),[0,0,0])

    def test_mixed_foreground_retains_soft_overlap_and_object_ID_invariance(self):
        source=np.array([[1,1,2,2]],np.int64);gt=np.array([[1,2,1,1]],np.int64)
        a=foreground_targets(source,gt,np.array([1,2]),np.array([[0,1]]));self.assertEqual(float(a[0]),.5)
        np.testing.assert_array_equal(a,soft_targets(source,gt,np.array([1,2]),np.array([[0,1]])))
        gt[gt==1]=77;gt[gt==2]=15
        np.testing.assert_array_equal(a,foreground_targets(source,gt,np.array([1,2]),np.array([[0,1]])))

    def test_no_relations_returns_empty_targets(self):
        value=foreground_targets(np.ones((2,2),np.int64),np.ones((2,2),np.int64),np.array([1]),np.empty((0,2),np.int64));self.assertEqual(value.shape,(0,))

    def test_GT_barrier_and_exclusive_GPU_assignment(self):
        from . import run
        from .queue import worker_command
        with patch.object(run,'validated_predictions',side_effect=ValueError('missing endpoint')),patch.object(run.parent.cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()
        self.assertEqual(worker_command(1)[-2:],['--physical-gpu','1'])
        with self.assertRaises(ValueError):worker_command(True)


if __name__=='__main__':unittest.main()
