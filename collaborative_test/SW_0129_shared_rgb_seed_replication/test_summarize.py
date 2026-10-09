import unittest
import numpy as np
from collaborative_test.SW_0129_shared_rgb_seed_replication import summarize

class SummaryTests(unittest.TestCase):
 def _scores(self,shift):
  return {m:np.linspace(.1,.9,320,dtype=np.float64)+shift for m in summarize.METRICS}
 def test_common_image_bootstrap_and_registered_three_seed_gate(self):
  source={s:self._scores(0.0) for s in range(3)}
  control={s:self._scores(.002) for s in range(3)}
  candidate={s:self._scores(.01 if s in (0,1) else .001) for s in range(3)}
  out=summarize.summarize(candidate,control,source)
  self.assertTrue(out['promotion_checks']['candidate_mean_fg_above_control'])
  self.assertTrue(out['promotion_checks']['candidate_mean_fg_above_source97'])
  self.assertTrue(out['promotion_checks']['at_least_two_seed_fg_gains_vs_source97'])
  self.assertTrue(out['promotion_checks']['shared_image_ci_lower_positive_vs_source97'])
  self.assertEqual(out['bootstrap']['seed'],129)
 def test_nonpositive_shared_interval_fails_without_changing_recipe(self):
  source={s:self._scores(0.) for s in range(3)}
  control={s:self._scores(0.) for s in range(3)}
  candidate={s:self._scores(-.01) for s in range(3)}
  out=summarize.summarize(candidate,control,source)
  self.assertEqual(out['promotion_status'],'failed')
  self.assertFalse(out['promotion_checks']['shared_image_ci_lower_positive_vs_control'])
 def test_seed0_references_original_trained_outputs_not_archive_shortcuts(self):
  self.assertEqual(summarize.historical_path('candidate').parent.name,'candidate_seed0')
  self.assertEqual(summarize.historical_path('control').parent.name,'control_seed0')
  self.assertEqual(summarize.historical_path('source').name,'evaluation.json')

 def test_bootstrap_requires_exact_three_seed_320_image_pairs(self):
  with self.assertRaises(ValueError):summarize.paired_bootstrap_ci(np.zeros((2,320)))
  with self.assertRaises(ValueError):summarize.paired_bootstrap_ci(np.zeros((3,319)))

if __name__=='__main__':unittest.main()
