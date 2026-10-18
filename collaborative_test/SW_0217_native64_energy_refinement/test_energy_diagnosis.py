import unittest
from unittest.mock import patch
import numpy as np
from . import diagnose_energy as diagnostic


class EnergyDiagnosisTests(unittest.TestCase):
    def test_tied_rank_AUC_matches_explicit_pair_comparison(self):
        scores=np.array([0.,2.,2.,1.,4.,0.]);positive=np.array([1,0,1,0,1,0],bool)
        brute=np.mean([float(a>b)+.5*float(a==b) for a in scores[positive] for b in scores[~positive]])
        self.assertEqual(diagnostic.auc(scores,positive),brute)
        self.assertIsNone(diagnostic.auc([],[]))
        self.assertIsNone(diagnostic.auc([1],[True]))

    def test_ground_truth_split_exposes_conflicting_positive_energy(self):
        source=np.array([[1,1],[1,1]]);truth=np.array([[1,1],[2,2]])
        energy=np.full((4,4),2.);np.fill_diagonal(energy,0)
        q=np.full((4,4),.8)
        result=diagnostic.evidence(energy,q,source,truth,np.arange(4),(2,2))
        self.assertEqual(result['false_merge']['pairs'],4)
        self.assertEqual(result['false_merge']['energy_positive_fraction'],1.)
        self.assertEqual(result['GT_connected_purification_energy_change'],-8.)
        self.assertEqual(result['GT_connected_purification_groups'],2)
        self.assertEqual(result['all_pair_energy_AUC'],.5)

    def test_foreground_background_and_nonadjacent_wrap_are_excluded(self):
        source=np.array([[0,1,1],[1,1,0]]);truth=np.array([[0,1,1],[2,0,0]])
        f=np.flatnonzero(source.reshape(-1)>0);e=np.zeros((4,4));q=np.full((4,4),.5)
        result=diagnostic.evidence(e,q,source,truth,f,(2,3))
        self.assertEqual(result['same_object']['pairs'],1)
        self.assertEqual(result['different_object']['pairs'],2)
        self.assertEqual(result['source_foreground_GT_background_patches'],1)
        # indices2 and3 straddle a row boundary and are not 4-neighbors.
        self.assertIsNone(result['spatial_neighbor_energy_AUC'])
        with self.assertRaises(ValueError):diagnostic.evidence(e,q,source,truth,np.arange(4),(2,3))

    def test_GT_barrier_before_any_diagnostic_reader(self):
        from . import run
        with patch.object(run,'validated_predictions',side_effect=ValueError('pending seed')),patch.object(run.source.full,'read_gt') as reader:
            with self.assertRaises(ValueError):diagnostic.diagnose()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
