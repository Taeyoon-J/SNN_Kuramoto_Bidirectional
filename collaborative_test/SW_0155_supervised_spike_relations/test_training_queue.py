from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0155_supervised_spike_relations import training_queue as queue


class _FakePath:
    def __init__(self, value):
        self.value = str(value)

    def __truediv__(self, child):
        return _FakePath(self.value + "/" + str(child))

    def exists(self):
        return self.value.endswith("/boundary.ptx")

    def is_file(self):
        return self.exists()


class SW0155TrainingQueueTests(unittest.TestCase):
    def test_plan_routes_both_registered_arms_to_fresh_cuda_training(self):
        tasks = queue.task_plan()
        self.assertEqual([task["arm"] for task in tasks], ["encoder_graph", "joint"])
        self.assertEqual([task["stage"] for task in tasks], ["train", "train"])
        for task in tasks:
            argv = queue.command(task)
            self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0155_supervised_spike_relations.train"])
            self.assertEqual(argv[argv.index("--arm") + 1], task["arm"])
            self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
            self.assertEqual(Path(argv[argv.index("--preflight") + 1]), queue.PREFLIGHT_PATH)
        with self.assertRaises(ValueError):
            queue.command(dict(tasks[0], arm="unregistered"))

    def test_resume_gpu_is_bound_to_valid_single_device_checkpoint(self):
        task = {"checkpoint_dir": "/fake/checkpoints"}
        metadata = {"runtime": {"cuda_visible_devices": "3", "device_type": "cuda",
                                "device_index": 0}, "completed_updates": 17}
        with mock.patch.object(queue, "Path", _FakePath), \
             mock.patch.object(queue, "inspect_committed_metadata", return_value=metadata):
            self.assertEqual(queue._required_resume_gpu(task), 3)
        for invalid in (
            {"runtime": {"cuda_visible_devices": "3", "device_type": "cpu", "device_index": 0},
             "completed_updates": 17},
            {"runtime": {"cuda_visible_devices": "3", "device_type": "cuda", "device_index": 1},
             "completed_updates": 17},
            {"runtime": {"cuda_visible_devices": "8", "device_type": "cuda", "device_index": 0},
             "completed_updates": 17},
            {"runtime": {"cuda_visible_devices": "3", "device_type": "cuda", "device_index": 0},
             "completed_updates": True},
        ):
            with mock.patch.object(queue, "Path", _FakePath), \
                 mock.patch.object(queue, "inspect_committed_metadata", return_value=invalid):
                with self.assertRaises(ValueError):
                    queue._required_resume_gpu(task)

    def test_preflight_must_be_current_before_supervisor_admission(self):
        with mock.patch.object(queue.Path, "read_text", return_value="{}"), \
             mock.patch.object(queue.train, "source_contract", return_value={"dummy": True}), \
             mock.patch.object(queue.sw130, "validate_rgb_assets", return_value={"dummy": True}), \
             mock.patch.object(queue.train, "HDF5_PATH") as hdf, \
             mock.patch.object(queue.preflight, "validate_admission", return_value=False):
            hdf.stat.return_value = mock.Mock(st_size=2, st_mtime_ns=3)
            with self.assertRaisesRegex(ValueError, "preflight"):
                queue.preflight_admission()


if __name__ == "__main__":
    unittest.main()
