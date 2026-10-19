import unittest
from . import run


class ScopeTests(unittest.TestCase):
    def test_only_registered_seed_and_arms_are_accepted(self):
        self.assertEqual(run.require_seed(1),1)
        for seed in (0,2,True):
            with self.assertRaises(ValueError):run.require_seed(seed)
        with self.assertRaises(ValueError):run.require_arm('within4')

    def test_queue_carries_explicit_matched_arm(self):
        from . import queue
        self.assertIs(queue.api,run);run.activate('control')
        command=queue.worker_command(1,3)
        self.assertEqual(command[-2:],['--arm','control'])
        self.assertEqual(command[command.index('--physical-gpu')+1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,3)


if __name__=='__main__':unittest.main()
