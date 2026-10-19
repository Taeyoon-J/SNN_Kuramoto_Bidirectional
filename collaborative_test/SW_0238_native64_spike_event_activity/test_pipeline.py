import unittest
from . import run

class ScopeTests(unittest.TestCase):
    def test_evaluation_or_duplicated_ids_are_rejected(self):
        run.validate_ids([0,1640])
        for ids in ([1,1],[999,1000],[1320,1640],[-1,1640],[70640,1],[True,2],[1]):
            with self.assertRaises(ValueError):run.validate_ids(ids)

    def test_queue_uses_diagnostic_and_valid_physical_GPU(self):
        from . import queue
        self.assertIs(queue.api,run)
        self.assertIn('SW_0238_native64_spike_event_activity',queue.MODULE)
        self.assertEqual(queue.worker_command(2,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,4)

if __name__=='__main__':unittest.main()
