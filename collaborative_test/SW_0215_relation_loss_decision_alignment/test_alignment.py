import unittest
from unittest.mock import patch
import numpy as np
from .calibration import training_log_odds_shift,correct


class AlignmentTests(unittest.TestCase):
    def test_ideal_weighted_BCE_odds_are_corrected_to_original_target(self):
        prior=.04;shift,_=training_log_odds_shift(np.array([prior]),np.ones(1))
        for probability in (.04,.2,.5,.8):
            ideal=np.log(probability/(1-probability))+shift
            result=1/(1+np.exp(-correct(np.array([ideal]),shift)))[0]
            self.assertAlmostEqual(result,probability,places=12)
        self.assertLess(correct(np.array([0.]),shift)[0],0)

    def test_base_image_weights_are_used_and_balanced_prior_is_identity(self):
        shift,stats=training_log_odds_shift(np.array([1.,0.]),np.array([1.,3.]))
        self.assertAlmostEqual(shift,np.log(3));self.assertEqual(stats['weighted_positive_prior'],.25)
        shift,_=training_log_odds_shift(np.array([1.,0.]),np.ones(2));self.assertEqual(shift,0.)

    def test_missing_class_mass_rejected(self):
        with self.assertRaises(ValueError):training_log_odds_shift(np.zeros(2),np.ones(2))

    def test_GT_is_not_read_on_parent_replay_failure(self):
        from . import run
        with patch.object(run,'freeze_predictions',side_effect=ValueError('parent replay mismatch')),patch.object(run.parent.parent.cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
