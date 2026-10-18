import unittest
import numpy as np
from .refine import refine,groups_of,connected,objective


class RefineTests(unittest.TestCase):
    def test_negative_cross_energy_splits_connected_source_group(self):
        labels=np.ones((2,2),np.int64);energy=np.full((4,4),-1.,np.float64);np.fill_diagonal(energy,0)
        energy[0,2]=energy[2,0]=3.;energy[1,3]=energy[3,1]=3.
        result=refine(energy,labels);self.assertEqual(result['diagnostic']['accepted_source_splits'],1)
        out=result['refined'];self.assertEqual(out[0,0],out[1,0]);self.assertEqual(out[0,1],out[1,1]);self.assertNotEqual(out[0,0],out[0,1])
        self.assertGreater(result['diagnostic']['energy_refined'],result['diagnostic']['energy_before'])

    def test_positive_clique_and_no_change_preserve_exact_labels(self):
        labels=np.full((2,2),17,np.int64);energy=np.ones((4,4));np.fill_diagonal(energy,0)
        result=refine(energy,labels);np.testing.assert_array_equal(result['refined'],labels);self.assertEqual(result['diagnostic']['accepted_source_splits'],0)

    def test_connected_pieces_foreground_and_energy_are_preserved(self):
        labels=np.array([[1,1,0],[1,1,2],[0,2,2]],np.int64)
        for seed in range(8):
            values=np.random.RandomState(seed).normal(size=(9,9));energy=(values+values.T)/2;np.fill_diagonal(energy,0)
            result=refine(energy,labels)
            for arm in ('split','refined'):
                out=result[arm];np.testing.assert_array_equal(out>0,labels>0)
                self.assertTrue(all(len(connected(g,labels.shape))==1 for g in groups_of(out)))
                self.assertGreaterEqual(objective(energy,groups_of(out))+result['diagnostic']['tolerance'],objective(energy,groups_of(labels)))

    def test_ID_permutation_and_deterministic_replay(self):
        labels=np.array([[8,8],[2,2]],np.int64);other=np.where(labels==8,3,91).astype(np.int64)
        energy=np.array([[0,3,2,-4],[3,0,-4,2],[2,-4,0,3],[-4,2,3,0]],float)
        a=refine(energy,labels);b=refine(energy,other);c=refine(energy,labels)
        np.testing.assert_array_equal(a['refined'].reshape(-1)[:,None]==a['refined'].reshape(-1),b['refined'].reshape(-1)[:,None]==b['refined'].reshape(-1))
        np.testing.assert_array_equal(a['refined'],c['refined']);self.assertEqual(a['diagnostic'],c['diagnostic'])

    def test_empty_singleton_and_invalid_source(self):
        for labels in (np.zeros((1,1),np.int64),np.ones((1,1),np.int64)):
            np.testing.assert_array_equal(refine(np.zeros((1,1)),labels)['refined'],labels)
        with self.assertRaises(ValueError):refine(np.zeros((4,4)),np.array([[1,0],[0,1]],np.int64))
        with self.assertRaises(ValueError):refine(np.eye(4),np.ones((2,2),np.int64))


if __name__=='__main__':unittest.main()
