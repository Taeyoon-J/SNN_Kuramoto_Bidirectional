import unittest
import numpy as np
from . import train,evaluate


class PolicyTests(unittest.TestCase):
    def test_finite_uphill_and_equal_steps_are_retained(self):
        for after in (1.,1.1):
            applied,descent,threshold=train.finite_step_policy(1.,after)
            self.assertTrue(applied);self.assertFalse(descent);self.assertEqual(threshold,1e-6)

    def test_downhill_step_records_both_flags(self):
        applied,descent,_=train.finite_step_policy(1.,.9)
        self.assertTrue(applied);self.assertTrue(descent)

    def test_nonfinite_or_invalid_parameter_proposal_rejected(self):
        for after in (None,float('nan'),float('inf')):
            self.assertEqual(train.finite_step_policy(1.,after)[:2],(False,False))
        self.assertEqual(train.finite_step_policy(1.,.9,False)[:2],(False,False))
        with self.assertRaises(ValueError):train.finite_step_policy(float('nan'),.9)

    def test_real_metric_improvement_and_CNN_learning_still_required(self):
        x=np.zeros((16,3));pair={key:evaluate.paired_change(x+.02,x) for key in ('control','warm')}
        report=dict(new_accepted_updates=2,committed_model_changed=True,CNN_changed_from_warm=True)
        self.assertTrue(evaluate.extension_gate(pair,report))
        pair['warm']['fg_ari']['mean']=-.001
        self.assertFalse(evaluate.extension_gate(pair,report))
        self.assertEqual(train.LIVE_TAIL,64)
