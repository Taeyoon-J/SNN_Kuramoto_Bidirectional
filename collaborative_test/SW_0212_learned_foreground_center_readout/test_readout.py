import unittest
from unittest.mock import patch
import numpy as np
from . import readout


class ReadoutTests(unittest.TestCase):
    def test_fixed_logit_threshold_includes_zero(self):
        values=np.array([[-2.,0.],[.1,8.]])
        np.testing.assert_array_equal(readout.learned_foreground(values),[[0,1],[1,1]])
        with self.assertRaises(ValueError):readout.learned_foreground(np.array([[np.nan]]))

    def test_nonlocal_votes_join_disconnected_visible_parts(self):
        fg=np.full((8,8),-10.);fg[1,1]=fg[6,6]=10.
        heat=np.full((8,8),-10.);heat[1,1]=10.
        offset=np.zeros((2,8,8));offset[:,6,6]=-5/7
        labels,info=readout.assign(fg,heat,offset)
        self.assertEqual(labels[1,1],labels[6,6]);self.assertEqual(info['predicted_objects'],1)
        np.testing.assert_array_equal(labels>0,fg>=0)

    def test_no_foreground_and_no_center_are_valid_predictions(self):
        labels,info=readout.assign(np.full((8,8),-1.),np.zeros((8,8)),np.zeros((2,8,8)))
        self.assertFalse(labels.any());self.assertFalse(info['fallback'])
        labels,info=readout.assign(np.ones((8,8)),np.full((8,8),-10.),np.zeros((2,8,8)))
        self.assertTrue((labels>0).all());self.assertTrue(info['fallback'])

    def test_replay_mismatch_cannot_be_ignored(self):
        with self.assertRaises(ValueError):readout.require_replay(np.zeros((2,2)),np.ones((2,2)))

    def test_GT_cannot_be_read_when_frozen_prediction_replay_fails(self):
        from . import run
        with patch.object(run,'frozen_predictions',side_effect=ValueError('replay mismatch')),patch.object(run.parent.cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
