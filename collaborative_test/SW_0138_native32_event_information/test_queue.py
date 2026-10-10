from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0138_native32_event_information import event_queue, run


class EventQueueContracts(unittest.TestCase):
    def test_three_seed_jobs_are_bounded_and_use_module_cli(self):
        tasks = event_queue.task_plan()
        self.assertEqual([item["seed"] for item in tasks], [0, 2, 1])
        self.assertEqual(event_queue.MAX_PARALLEL, 2)
        argv = event_queue.command(tasks[0])
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0138_native32_event_information.run"])

    def test_gpu_requires_no_compute_owner_and_memory_cap(self):
        self.assertTrue(event_queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(event_queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(event_queue.gpu_is_exclusive_candidate(10, [12345]))

    def test_queue_refuses_to_initialize_if_any_sw0137_dependency_is_missing(self):
        with mock.patch.object(event_queue.sw137, "validate_prediction",
                                side_effect=FileNotFoundError("seed2 not complete")) as validate:
            with self.assertRaises(FileNotFoundError):
                event_queue.validate_sw137_dependencies(Path("unused"))
        self.assertEqual(validate.call_count, 1)

    def test_dependency_validator_binds_all_three_seeds_and_common_ids(self):
        import numpy as np
        fake = {seed: {"ids": np.arange(1320, 1640, dtype=np.int64),
                       "prediction_sha256": f"pred{seed}",
                       "protocol_sha256": f"protocol{seed}"}
                for seed in (0, 1, 2)}
        with mock.patch.object(event_queue.sw137, "validate_prediction",
                               side_effect=lambda path, seed: fake[seed]):
            rows = event_queue.validate_sw137_dependencies(Path("unused"))
        self.assertEqual(set(rows), {0, 1, 2})
        self.assertEqual(rows[2]["prediction_sha256"], "pred2")

    def test_dependency_validator_rejects_unpaired_image_ids(self):
        import numpy as np
        fake = {seed: {"ids": np.arange(1320, 1640, dtype=np.int64),
                       "prediction_sha256": f"pred{seed}", "protocol_sha256": f"p{seed}"}
                for seed in (0, 1, 2)}
        fake[2]["ids"][0] = 1321
        with mock.patch.object(event_queue.sw137, "validate_prediction",
                               side_effect=lambda path, seed: fake[seed]):
            with self.assertRaises(ValueError):
                event_queue.validate_sw137_dependencies(Path("unused"))

    def test_seed_outputs_are_distinct_and_do_not_alias_sw0137(self):
        tasks = event_queue.task_plan()
        paths = [event_queue.output_path(task) for task in tasks]
        self.assertEqual(len(set(paths)), 3)
        self.assertNotEqual(paths[0], event_queue.sw137._prediction_dir(0))


if __name__ == "__main__":
    unittest.main()
