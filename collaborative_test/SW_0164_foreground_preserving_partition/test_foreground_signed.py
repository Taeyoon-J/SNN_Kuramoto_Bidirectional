"""Small exact CPU oracles for the SW0164 induced foreground solver."""
import unittest

import numpy as np

from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as fs


def q_for_energy(e):
    e = np.asarray(e, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-e))


def assert_each_group_connected(test, labels, foreground):
    h, w = labels.shape
    for group in np.unique(labels[foreground]):
        points = set(np.flatnonzero((labels == group).reshape(-1)).tolist())
        seen = {next(iter(points))}; todo = list(seen)
        while todo:
            i = todo.pop(); r, c = divmod(i, w)
            for j in ((i-w,) if r else ()) + ((i+w,) if r+1<h else ()) + ((i-1,) if c else ()) + ((i+1,) if c+1<w else ()):
                if j in points and j not in seen: seen.add(j); todo.append(j)
        test.assertEqual(seen, points)


class ForegroundSignedTests(unittest.TestCase):
    def test_preserves_every_foreground_singleton_and_background_exactly(self):
        h, w = 3, 4; n = h*w
        fg = np.zeros((h,w), dtype=np.int32)
        fg[0,0] = 5; fg[0,1] = 2; fg[1,3] = 9; fg[2,0] = 7
        e = np.full((n,n), -3.0); np.fill_diagonal(e, 0)
        e[0,1] = e[1,0] = 2.0
        e[3,11] = e[11,3] = 100.0 # diagonal/non-neighbour must not merge
        result = fs.foreground_induced_merge(q_for_energy(e), np.zeros_like(e), 1.0, fg, grid_shape=(h,w))
        got = result["labels"]
        np.testing.assert_array_equal(got > 0, fg > 0)
        self.assertEqual(result["merge_count"], 1)
        self.assertEqual(got[0,0], got[0,1])
        self.assertGreater(got[1,3], 0); self.assertGreater(got[2,0], 0)
        self.assertEqual(int(got[0,2]), 0)
        self.assertEqual(result["discarded_mass"], 0)
        assert_each_group_connected(self, got, fg > 0)

    def test_merge_gain_includes_negative_crosspairs_and_can_stop_positive_bridge(self):
        # 0--1 is the best initial adjacency. Merged {0,1} then has complete
        # cross evidence 3 + (-10) to 2, so the positive 1--2 edge cannot merge.
        e = np.zeros((3,3), dtype=np.float64)
        e[0,1] = e[1,0] = 10.0
        e[1,2] = e[2,1] = 3.0
        e[0,2] = e[2,0] = -10.0
        fg = np.ones((1,3), dtype=np.int32)
        result = fs.foreground_induced_merge(q_for_energy(e), np.zeros_like(e), 1.0, fg, grid_shape=(1,3))
        self.assertEqual(result["merge_count"], 1)
        np.testing.assert_array_equal(result["labels"], [[1,1,2]])
        self.assertAlmostEqual(float(result["pair_energy"][0,2]+result["pair_energy"][1,2]), -7.0, places=7)

    def test_background_edges_and_nonadjacent_energy_do_not_enter_induced_merges(self):
        h,w=2,3; n=h*w
        fg=np.array([[1,0,2],[0,3,0]],dtype=np.int32)
        e=np.zeros((n,n)); e[0,2]=e[2,0]=20; e[0,4]=e[4,0]=20
        result=fs.foreground_induced_merge(q_for_energy(e),np.zeros_like(e),1,fg,grid_shape=(h,w))
        self.assertEqual(result["merge_count"],0)
        self.assertEqual(np.count_nonzero(result["labels"]),3)
        assert_each_group_connected(self,result["labels"],fg>0)

    def test_empty_support_and_stable_first_patch_ties(self):
        q=np.full((4,4),.5); a=np.zeros_like(q)
        empty=fs.foreground_induced_merge(q,a,1,np.zeros((2,2),dtype=np.int32),grid_shape=(2,2))
        self.assertEqual(empty["merge_count"],0); self.assertFalse(empty["labels"].any())
        e=np.zeros((4,4)); e[0,1]=e[1,0]=2; e[0,2]=e[2,0]=2
        labels=np.array([[1,1],[1,0]],dtype=np.int32)
        one=fs.foreground_induced_merge(q_for_energy(e),a,1,labels,grid_shape=(2,2))
        two=fs.foreground_induced_merge(q_for_energy(e),a,1,labels,grid_shape=(2,2))
        self.assertEqual(one["merge_log"][0]["patch_pair"],[0,1])
        np.testing.assert_array_equal(one["labels"],two["labels"])

    def test_full_q_a_energy_and_source_validation_are_not_reduced_or_relaxed(self):
        q=np.full((4,4),.5); a=np.zeros_like(q); labels=np.array([[1,0],[0,0]],dtype=np.int32)
        result=fs.foreground_induced_merge(q,a,2,labels,grid_shape=(2,2))
        self.assertEqual(result["pair_energy"].shape,(4,4))
        for badq,bada,gain,badlabels in ((q.copy(),a,0,labels),(q+np.triu(np.ones_like(q),1)*.1,a,1,labels),(q,a,1,np.array([[1.,0],[0,0]]))):
            with self.assertRaises(ValueError): fs.foreground_induced_merge(badq,bada,gain,badlabels,grid_shape=(2,2))
        with self.assertRaises(ValueError): fs.foreground_induced_merge(q,a,1,labels[:1],grid_shape=(2,2))


if __name__ == "__main__": unittest.main()
