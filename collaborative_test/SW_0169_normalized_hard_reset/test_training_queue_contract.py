"""Actual worker route and exclusive-owner policy for SW169 training."""
import importlib.util
import os
import unittest

import numpy

from collaborative_test.SW_0169_normalized_hard_reset import train, training_queue as queue


class TrainingQueueContractTests(unittest.TestCase):
    def test_worker_resolves_current_module_and_registered_arm(self):
        for arm in train.ARMS:
            command=queue._worker_command(arm,2)
            self.assertEqual(command[1:3],['-m','collaborative_test.SW_0169_normalized_hard_reset.training_queue'])
            self.assertEqual(command[3:],['--worker','--arm',arm,'--physical-gpu','2'])
            self.assertEqual(importlib.util.find_spec(command[2]).origin,queue.__file__)
        with self.assertRaises(ValueError): queue._worker_command('unregistered',0)
        with self.assertRaises(ValueError): queue._worker_command(train.ARMS[0],4)

    def test_foreign_owner_and_uuid_yield_without_rejecting_owned_gpu(self):
        self.assertFalse(queue._foreign_owner_changed('GPU-a','GPU-a',[123],123))
        self.assertFalse(queue._foreign_owner_changed('GPU-a','GPU-a',[],123))
        self.assertTrue(queue._foreign_owner_changed('GPU-a','GPU-a',[123,456],123))
        self.assertTrue(queue._foreign_owner_changed('GPU-a','GPU-b',[123],123))

    def test_arms_have_distinct_runtime_and_parent_hides_cuda(self):
        self.assertNotEqual(queue._state_path(train.ARMS[0]),queue._state_path(train.ARMS[1]))
        self.assertEqual(queue.RUNTIME.parent,train.HERE/'results_archive')
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'')


if __name__=='__main__': unittest.main()
