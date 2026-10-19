import unittest
import numpy as np
from .spike_gate_activity import summarize


class ActivityTests(unittest.TestCase):
    def test_gate_copy_with_all_on_observable_events(self):
        g=np.array([[[0.,.2,.8,1.]]],dtype=np.float32)
        s=np.broadcast_to(g[:,None],(1,4,1,4)).copy()
        r=summarize(s,g)
        self.assertTrue(r['gate_copy_bitwise_equal'])
        self.assertEqual(r['positive_gate_event_occupancy'],1.)
        self.assertEqual(r['zero_gate_count'],4)
        self.assertEqual(r['components'][0]['observable_transition_pair_count'],2)
        self.assertEqual(r['components'][0]['event_transition_count'],0)
        self.assertFalse(r['establishes_gate_or_membrane_causal_performance_contribution'])

    def test_real_event_changes_are_not_gate_copy(self):
        g=np.full((1,1,4),.5,dtype=np.float32)
        s=np.array([[[[0.,.5,0.,.5]]]],dtype=np.float32)
        r=summarize(s,g)
        self.assertFalse(r['gate_copy_bitwise_equal'])
        self.assertEqual(r['components'][0]['event_occupancy_given_positive_gate'],.5)
        self.assertEqual(r['components'][0]['transition_rate_given_both_gates_positive'],1.)
        self.assertEqual(r['fraction_of_positive_gate_samples_suppressed_by_event'],.5)

    def test_masked_events_are_not_invented(self):
        g=np.zeros((1,1,3),dtype=np.float32)
        r=summarize(np.zeros((1,2,1,3),dtype=np.float32),g)
        self.assertIsNone(r['positive_gate_event_occupancy'])
        self.assertIsNone(r['components'][0]['transition_rate_given_both_gates_positive'])

    def test_tiny_positive_gate_is_not_misread_as_event_one(self):
        g=np.full((1,1,3),1e-30,dtype=np.float32)
        r=summarize(np.zeros((1,1,1,3),dtype=np.float32),g)
        self.assertEqual(r['positive_gate_event_occupancy'],0.)
        self.assertEqual(r['fraction_of_positive_gate_samples_suppressed_by_event'],1.)

    def test_non_native_or_invalid_traces_rejected(self):
        g=np.full((1,1,3),.5,dtype=np.float32)
        for values in ([.1,.1,.1],[np.nan,0,0],[-.1,0,0]):
            with self.assertRaises(ValueError):summarize(np.array(values,dtype=np.float32).reshape(1,1,1,3),g)
        with self.assertRaises(ValueError):summarize(np.ones((1,1,1,3),dtype=np.float32),np.zeros_like(g))


if __name__=='__main__':unittest.main()
