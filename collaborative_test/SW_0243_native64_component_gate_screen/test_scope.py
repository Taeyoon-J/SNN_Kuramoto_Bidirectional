import unittest
from . import run


class ScreenScopeTests(unittest.TestCase):
    def test_only_preregistered_diagnostic_seed_is_accepted(self):
        self.assertEqual(run.require_seed(1),1)
        for seed in (0,2,True,-1):
            with self.assertRaises(ValueError):run.require_seed(seed)

    def test_queue_uses_component_gate_variant(self):
        from . import queue
        self.assertIs(queue.api,run)
        self.assertEqual(queue.MODULE,'collaborative_test.SW_0243_native64_component_gate_screen.queue')
        self.assertEqual(queue.worker_command(1,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(0,1)


if __name__=='__main__':unittest.main()
