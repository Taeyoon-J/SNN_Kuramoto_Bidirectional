import copy
import unittest
import numpy as np
from . import completion_audit as audit,evaluate


class AuditTests(unittest.TestCase):
    def test_actual512_image_budget_and_order_cannot_be_replaced_with64(self):
        from . import train
        report=dict(unique_TRAIN_stage_images=512,logical_TRAIN_exposures=8192,history=[dict(update=i+1,indices=train.accepted.batch_indices(0,512,i)) for i in range(512)])
        audit.check_train_budget(report,0)
        wrong=copy.deepcopy(report);wrong['unique_TRAIN_stage_images']=64
        with self.assertRaises(ValueError):audit.check_train_budget(wrong,0)
        wrong=copy.deepcopy(report);wrong['history'][0]['indices']=train.accepted.batch_indices(0,64,0)
        with self.assertRaises(ValueError):audit.check_train_budget(wrong,0)

    def fixture(self,ours=(.88,.65,.71),gate=True,balanced=True):
        points=dict(baseline=(.84,.6,.7),train64_512=(.86,.64,.7),train512_512=ours)
        vectors={f'{arm}_seed{s}':np.tile(value,(16,1)) for arm,value in points.items() for s in (0,1,2)}
        scores={arm:dict(zip(evaluate.METRICS,value)) for arm,value in points.items()}
        paired={}
        for arm in ('baseline','train64_512'):
            delta=np.asarray(ours)-points[arm]
            paired[arm]={key:dict(mean=float(delta[i]),ci95=[float(delta[i]),float(delta[i])]) for i,key in enumerate(evaluate.METRICS)}
        return vectors,dict(scores=scores,per_seed_scores={name:copy.deepcopy(scores[name.rsplit('_seed',1)[0]]) for name in vectors},
            paired_joint_minus=paired,data_scaling_FG_CI_positive=bool(ours[0]>.86),FG_track_candidate_gate=gate,full320_candidate_gate=gate,balanced_candidate_gate=balanced)

    def test_means_and_candidate_gate_are_independently_reproduced(self):
        error,gate,balanced=audit.check_statistics(*self.fixture())
        self.assertLess(error,1e-12);self.assertTrue(gate);self.assertTrue(balanced)

    def test_IoU_regression_cannot_pass_balanced_gate(self):
        _,gate,balanced=audit.check_statistics(*self.fixture(ours=(.88,.59,.71),balanced=False))
        self.assertTrue(gate);self.assertFalse(balanced)

    def test_IoU_better_than_baseline_but_worse_than_matched_control_fails(self):
        _,gate,balanced=audit.check_statistics(*self.fixture(ours=(.88,.63,.71),balanced=False))
        self.assertTrue(gate);self.assertFalse(balanced)

    def test_data_scaling_claim_requires_actual_matched_control_CI(self):
        vectors,score=self.fixture();score['data_scaling_FG_CI_positive']=False
        with self.assertRaisesRegex(ValueError,'data scaling CI'):audit.check_statistics(vectors,score)

    def test_no_improvement_over_train64_512_cannot_pass(self):
        _,gate,balanced=audit.check_statistics(*self.fixture(ours=(.859,.65,.71),gate=False,balanced=False))
        self.assertFalse(gate);self.assertFalse(balanced)

    def test_altered_CI_and_nonfinite_summary_are_rejected(self):
        vectors,score=self.fixture();score['paired_joint_minus']['baseline']['fg_ari']['ci95'][0]+=.01
        with self.assertRaises(ValueError):audit.check_statistics(vectors,score)
        vectors,score=self.fixture();score['scores']['train512_512']['fg_ari']=float('nan')
        with self.assertRaises(ValueError):audit.check_statistics(vectors,score)


if __name__=='__main__':unittest.main()
