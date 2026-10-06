"""CPU-only scheduling checks; no real SSH/GPU/training process is started."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import resume_our_official as queue


class Child:
    returncode = None

    def poll(self):
        return self.returncode


class SchedulingChecks(unittest.TestCase):
    def complete(self, root, seed):
        output = root / f'trained_models/SW0092_our_on_official_s{seed}'
        for name in ('core.pt', 'checkpoints/epoch_01.pt', 'checkpoints/epoch_03.pt',
                     'checkpoints/epoch_10.pt', 'TRAINING_COMPLETED'):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('complete')

    def exercise(self, preserve_live):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root, children, calls = Path(folder), [], []
            if not preserve_live:
                self.complete(root, 1)

            def launch(args):
                child = Child()
                children.append((child, int(args[-1])))
                calls.append((int(args[-2]), int(args[-1])))
                return child

            def finish(_):
                self.complete(root, 1)
                for child, seed in children:
                    self.complete(root, seed)
                    child.returncode = 0

            def available(reserved=()):
                for gpu in (0, 1):
                    occupied = preserve_live and gpu == 1 and not (root / 'trained_models/SW0092_our_on_official_s1/TRAINING_COMPLETED').is_file()
                    if gpu not in reserved and not occupied:
                        return gpu
                return None

            with patch.object(queue, 'ROOT', root), patch.object(queue, 'state'), \
                    patch.object(queue, 'matching_train_process', side_effect=lambda output: preserve_live and str(output).endswith('_s1')), \
                    patch.object(queue, 'gpu_available', side_effect=available), \
                    patch.object(queue.subprocess, 'Popen', side_effect=launch), \
                    patch.object(queue.time, 'sleep', side_effect=finish):
                queue.train_all_seeds()
            self.assertEqual([seed for _, seed in calls], [0, 2])
            if not preserve_live:
                self.assertEqual(calls, [(0, 0), (1, 2)])

    def test_preserves_existing_seed(self):
        self.exercise(True)

    def test_reserves_gpu_before_cuda_registration(self):
        self.exercise(False)

    def test_refuses_partial_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / 'trained_models/SW0092_our_on_official_s0').mkdir(parents=True)
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'matching_train_process', return_value=False), \
                    patch.object(queue.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Partial training output preserved'):
                    queue.train_all_seeds()
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
