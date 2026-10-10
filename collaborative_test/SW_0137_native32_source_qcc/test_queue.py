from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0137_native32_source_qcc import run, source_queue


class SourceQueueContracts(unittest.TestCase):
    def test_three_seed_plan_keeps_independent_work_parallel_and_reuse_seed_last(self):
        tasks = source_queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 2, 1])
        self.assertEqual(source_queue.MAX_PARALLEL, 2)
        self.assertEqual(source_queue.command(tasks[-1])[-1], "--no-reuse-seed1")
        self.assertTrue(source_queue.command(tasks[0])[1:3] == [
            "-m", "collaborative_test.SW_0137_native32_source_qcc.run"])

    def test_gpu_candidate_requires_no_owner_and_at_most512mib(self):
        self.assertTrue(source_queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(source_queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(source_queue.gpu_is_exclusive_candidate(1, [445]))

    def test_queue_result_requires_seed_specific_completed_prediction_validator(self):
        with mock.patch.object(run, "validate_prediction", return_value={"status": "complete"}) as validator:
            self.assertTrue(source_queue.validate_result({"seed": 2}))
            validator.assert_called_once_with(source_queue.output_path({"seed": 2}), seed=2)
        self.assertFalse(source_queue.validate_result({"seed": 9}))

    def test_scoring_refuses_missing_prediction_before_hdf5_target_open(self):
        with mock.patch.object(run, "validate_prediction", side_effect=FileNotFoundError("missing source row")), \
             mock.patch.object(run.h5py, "File", side_effect=AssertionError("target opened before source checks")):
            with self.assertRaises(FileNotFoundError):
                run.score_sources(output_root=Path("unused"), dataset=Path("must-not-open"))

    def test_seed_paths_are_distinct_and_outputs_never_alias_other_seed(self):
        tasks = source_queue.task_plan()
        paths = [source_queue.output_path(task) for task in tasks]
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(path.name == f"seed{seed}" for path, seed in zip(paths, (0, 2, 1))))


if __name__ == "__main__":
    unittest.main()
