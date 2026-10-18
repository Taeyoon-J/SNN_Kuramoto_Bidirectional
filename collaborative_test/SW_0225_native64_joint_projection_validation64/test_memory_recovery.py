import json
import sys
import unittest
import uuid
import os
import subprocess
import numpy as np
import torch
from unittest import mock
from . import recover_memory, completion_audit, run


class MemoryRecoveryTests(unittest.TestCase):
    def test_live_or_successful_original_cannot_be_duplicated(self):
        state = dict(status='failed', child_pid=12, supervisor_pid=13)
        self.assertFalse(recover_memory.original_terminal(state, lambda p: p == 12))
        self.assertFalse(recover_memory.original_terminal(dict(status='running_worker'), lambda p: False))
        self.assertFalse(recover_memory.original_terminal(dict(status='complete'), lambda p: False))
        self.assertTrue(recover_memory.original_terminal(state, lambda p: False))

    def test_recovery_requires_exact_replay_and_protocol_receipt(self):
        root = run.ROOT / 'collaborative_testing/temporary' / ('SW0225_memory_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        protocol = dict(binding=dict(candidate='frozen fixture'))
        run.write_json(root / 'protocol.json', protocol)
        run.write_json(root / 'queue/state.json', dict(status='failed'))
        run.write_json(root / 'queue_memory_recovery_v2/state.json', dict(status='complete',
            original_terminal_state_sha256=run.sha((root / 'queue/state.json').read_bytes()),
            protocol_sha256=run.sha((root / 'protocol.json').read_bytes()), per_process_THP_disabled=True))
        replay = dict(binding=protocol['binding'], per_process_THP_disabled=True,
                      mask_bitwise_equal=True, frequency_bitwise_equal=True, optimizer_updates=0)
        run.write_json(root / 'queue_memory_recovery_v2/replay.json', replay)
        self.assertEqual(completion_audit.completed_queue(root, protocol), 'queue_memory_recovery_v2/state.json')
        replay['frequency_bitwise_equal'] = False
        (root / 'queue_memory_recovery_v2/replay.json').write_text(json.dumps(replay))
        with self.assertRaises(ValueError):
            completion_audit.completed_queue(root, protocol)

    @unittest.skipUnless(sys.platform == 'linux', 'actual Linux per-process prctl')
    def test_actual_per_process_policy(self):
        recover_memory.disable_thp()
        child = subprocess.run([sys.executable, '-c',
            'import ctypes; print(ctypes.CDLL(None).prctl(42,0,0,0,0))'],
            check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(child.stdout.strip(), '1')

    def test_replay_executes_fresh_interpreter_before_production_CPU_admission(self):
        root = run.ROOT / 'collaborative_testing/temporary' / ('SW0225_exec_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        labels = np.ones((64, 64), dtype=np.int64)
        frequency = np.zeros((4096, 4), dtype=np.float32)
        p = dict(found=object(), rgb=np.zeros((1, 128, 128, 3), dtype=np.uint8),
                 mini=labels[None], mini_frequency=frequency[None],
                 binding=dict(candidate=dict(expected_runtime='fixture')))
        class ReplacedInterpreter(Exception):
            pass
        with mock.patch.object(run, 'prepare_cpu', return_value=p), \
             mock.patch.object(run, 'archive_for', return_value=root), \
             mock.patch.object(run, 'forward', return_value=(labels, frequency)), \
             mock.patch.object(run.source.source.source.model64, 'move_to_device'), \
             mock.patch.object(run.scope.model.checkpoint, 'runtime_signature'), \
             mock.patch.object(run.scope.model.resume_policy, 'compatible_runtime', return_value='fixture'), \
             mock.patch.object(torch.cuda, 'set_device'), \
             mock.patch.object(torch.cuda, 'set_per_process_memory_fraction'), \
             mock.patch.object(run, 'predict') as predict, \
             mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1'), \
             mock.patch.object(os, 'execv', side_effect=ReplacedInterpreter) as replace:
            with self.assertRaises(ReplacedInterpreter):
                recover_memory.worker(run, 2)
            predict.assert_not_called()
            self.assertEqual(replace.call_args.args[1][-5:], ['--worker', '--seed', '2', '--physical-gpu', '1'])
        self.assertTrue(json.loads((root / recover_memory.RECOVERY_FOLDER / 'replay.json').read_bytes())['mask_bitwise_equal'])
