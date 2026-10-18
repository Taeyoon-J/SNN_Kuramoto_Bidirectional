import unittest
from unittest.mock import patch
from . import run


class PipelineTests(unittest.TestCase):
    def test_GT_reader_not_opened_if_child_masks_or_parent_prefix_fail(self):
        with patch.object(run,'validated_predictions',side_effect=ValueError('incomplete child')),patch.object(run.parent.cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()

    def test_secondary_foreground_and_fixed_primary_are_separate(self):
        self.assertEqual(run.ARMS,('primary512','mean_input','moments_only','learned_foreground'))
        self.assertEqual(run.fingerprint()['total_steps'],512)
        self.assertEqual(run.fingerprint()['new_steps'],384)

    def test_worker_requires_real_physical_GPU_assignment(self):
        from .queue import worker_command
        self.assertEqual(worker_command(1)[-2:],['--physical-gpu','1'])
        with self.assertRaises(ValueError):worker_command(True)


if __name__=='__main__':unittest.main()
