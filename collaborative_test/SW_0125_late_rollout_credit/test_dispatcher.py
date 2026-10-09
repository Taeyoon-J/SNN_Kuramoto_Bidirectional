"""Small no-GPU tests for resource and stage-order queue contracts."""
import unittest

from collaborative_test.SW_0125_late_rollout_credit import dispatcher


class DispatcherTests(unittest.TestCase):
    def test_gpu_requires_no_compute_owners_and_registered_memory_headroom(self):
        self.assertTrue(dispatcher.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(dispatcher.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(dispatcher.gpu_is_exclusive_candidate(2, [12345]))


if __name__ == "__main__":
    unittest.main()
