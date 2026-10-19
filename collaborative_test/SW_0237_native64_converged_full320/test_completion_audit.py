import copy
import unittest
import numpy as np
from . import completion_audit as audit, evaluate


class AuditTests(unittest.TestCase):
    def fixture(self, ours=(.91, .6, .7), strict=True):
        points = dict(baseline=(.8, .5, .6), joint_projection=ours, slot=(.78, .2, .2))
        vectors = {f'{arm}_seed{s}': np.tile(value, (320, 1)) for arm, value in points.items() for s in (0, 1, 2)}
        scores = {arm: dict(zip(evaluate.METRICS, value)) for arm, value in points.items()}
        paired = {}
        for arm in ('baseline', 'slot'):
            delta = np.asarray(ours) - points[arm]
            paired[arm] = {key: dict(mean=float(delta[i]), ci95=[float(delta[i]), float(delta[i])]) for i, key in enumerate(evaluate.METRICS)}
        score = dict(scores=scores, per_seed_scores={name: copy.deepcopy(scores[name.rsplit('_seed', 1)[0]]) for name in vectors},
                     paired_joint_minus=paired, three_seed_strictly_exceeds_Slot_means=strict,
                     native64_FG090_three_seed_goal_verified=bool(strict))
        accepted = dict(our_three_seed_mean=scores['baseline'], slot_three_seed_mean=scores['slot'])
        return vectors, score, accepted

    def test_full_scope_statistics_and_combined_goal(self):
        maximum, strict, goal = audit.check_statistics(*self.fixture())
        self.assertLess(maximum, 1e-12); self.assertTrue(strict); self.assertTrue(goal)

    def test_high_FG_cannot_pass_when_an_IoU_only_ties_Slot(self):
        _, strict, goal = audit.check_statistics(*self.fixture(ours=(.91, .2, .7), strict=False))
        self.assertFalse(strict); self.assertFalse(goal)

    def test_altered_confidence_interval_is_rejected(self):
        vectors, score, accepted = self.fixture()
        score['paired_joint_minus']['baseline']['fg_ari']['ci95'][0] += .01
        with self.assertRaises(ValueError):
            audit.check_statistics(vectors, score, accepted)

    def test_nonfinite_summary_cannot_be_hidden_by_maximum_error(self):
        vectors, score, accepted = self.fixture()
        score['scores']['joint_projection']['fg_ari'] = float('nan')
        with self.assertRaises(ValueError):
            audit.check_statistics(vectors, score, accepted)


if __name__ == '__main__':
    unittest.main()
