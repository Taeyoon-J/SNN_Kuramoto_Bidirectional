import copy
import unittest
import numpy as np
from . import completion_audit as audit,evaluate


class AuditTests(unittest.TestCase):
    def fixture(self,ours=(.88,.65,.71),gate=True,balanced=True):
        points=dict(baseline=(.84,.6,.7),center1=(.86,.64,.7),relative_separation=ours)
        vectors={f'{arm}_seed{s}':np.tile(value,(16,1)) for arm,value in points.items() for s in (0,1,2)}
        scores={arm:dict(zip(evaluate.METRICS,value)) for arm,value in points.items()}
        paired={}
        for arm in ('baseline','center1'):
            delta=np.asarray(ours)-points[arm]
            paired[arm]={key:dict(mean=float(delta[i]),ci95=[float(delta[i]),float(delta[i])]) for i,key in enumerate(evaluate.METRICS)}
        return vectors,dict(scores=scores,per_seed_scores={name:copy.deepcopy(scores[name.rsplit('_seed',1)[0]]) for name in vectors},
            paired_joint_minus=paired,FG_track_candidate_gate=gate,full320_candidate_gate=gate,balanced_candidate_gate=balanced)

    def test_means_and_candidate_gate_are_independently_reproduced(self):
        error,gate,balanced=audit.check_statistics(*self.fixture())
        self.assertLess(error,1e-12);self.assertTrue(gate);self.assertTrue(balanced)

    def test_IoU_regression_cannot_pass_balanced_gate(self):
        _,gate,balanced=audit.check_statistics(*self.fixture(ours=(.88,.59,.71),balanced=False))
        self.assertTrue(gate);self.assertFalse(balanced)

    def test_no_improvement_over_center1_cannot_pass(self):
        _,gate,balanced=audit.check_statistics(*self.fixture(ours=(.859,.65,.71),gate=False,balanced=False))
        self.assertFalse(gate);self.assertFalse(balanced)

    def test_altered_CI_and_nonfinite_summary_are_rejected(self):
        vectors,score=self.fixture();score['paired_joint_minus']['baseline']['fg_ari']['ci95'][0]+=.01
        with self.assertRaises(ValueError):audit.check_statistics(vectors,score)
        vectors,score=self.fixture();score['scores']['relative_separation']['fg_ari']=float('nan')
        with self.assertRaises(ValueError):audit.check_statistics(vectors,score)


if __name__=='__main__':unittest.main()
