import unittest
import numpy as np
from .completion_audit import checked_activity
from collaborative_test.SW_0238_native64_spike_event_activity.spike_gate_activity import summarize


class RawActivityAuditTests(unittest.TestCase):
    def test_real_raw_tail_recomputation_detects_event_change(self):
        gates=np.full((1,4,2,514),.25,dtype=np.float32);spikes=gates.copy()
        spikes[0,2,1,-1]=0
        prior=summarize(spikes[...,512:],gates[...,512:])
        self.assertEqual(checked_activity(spikes,gates,prior),prior)
        corrupt=spikes.copy();corrupt[0,2,1,-1]=gates[0,2,1,-1]
        with self.assertRaises(ValueError):checked_activity(corrupt,gates,prior)

    def test_prefix_is_excluded_by_registered_settle(self):
        gates=np.full((1,4,2,514),.25,dtype=np.float32);spikes=gates.copy()
        prior=summarize(spikes[...,512:],gates[...,512:])
        spikes[...,:512]=0
        self.assertEqual(checked_activity(spikes,gates,prior),prior)


if __name__=='__main__':unittest.main()
