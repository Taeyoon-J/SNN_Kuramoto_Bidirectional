import unittest
from unittest.mock import patch
import numpy as np
from . import run


class ReadoutTests(unittest.TestCase):
    def test_actual_baseline_replay_and_no_foreground_drop(self):
        q=np.array([[1,.8,.1],[.8,1,.2],[.1,.2,1]],np.float64)
        a=np.full((3,3),.001);gain=1.
        fullq=np.full((4,4),.5);fulla=np.zeros((4,4));f=np.array([0,1,3])
        fullq[np.ix_(f,f)]=q;fulla[np.ix_(f,f)]=a
        support=np.array([[1,1],[0,1]],np.int64)
        baseline=run.native.foreground_induced_merge(fullq,fulla,gain,support,grid_shape=(2,2))['labels'].astype(np.int64)
        expected=run.native.foreground_induced_merge(fullq,np.zeros((4,4)),gain,support,grid_shape=(2,2))['labels'].astype(np.int64)
        actual=run.decode(q,a,gain,baseline)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual>0,baseline>0)

    def test_zero_A_is_identity_and_wrong_baseline_is_rejected(self):
        q=np.array([[1,.8],[.8,1]]);a=np.zeros((2,2));baseline=np.array([[1,1]],np.int64)
        np.testing.assert_array_equal(run.decode(q,a,1.,baseline),baseline)
        with self.assertRaises(ValueError):run.decode(q,a,1.,np.array([[1,2]],np.int64))

    def test_empty_support(self):
        np.testing.assert_array_equal(run.decode(np.zeros((0,0)),np.zeros((0,0)),1.,np.zeros((2,2),np.int64)),np.zeros((2,2),np.int64))

    def test_GT_is_not_read_until_all_predictions_are_valid(self):
        with patch.object(run,'validated_predictions',side_effect=ValueError('missing row')),patch.object(run.source.source.full,'read_gt') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
