from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding import resource_queue as queue


class ResourceQueueTests(unittest.TestCase):
    def test_only_seed0_resource_task_and_module_command_are_registered(self):
        task, = queue.task_plan()
        argv = queue.command(task)
        self.assertEqual(task["stage"], "resource_probe")
        self.assertEqual(task["seed"], 0)
        self.assertIn("-m", argv)
        self.assertEqual(argv[argv.index("-m") + 1],
                         "collaborative_test.SW_0135_native32_spike_binding.resource_probe")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            queue.result_path({"stage": "train", "seed": 0})

    def test_gpu_candidate_requires_no_compute_owner_and_under_memory_limit(self):
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(10, [12345]))

    def test_owner_monitor_distinguishes_owned_vanished_and_foreign_pids(self):
        with mock.patch.object(owner, "_pid_is_confirmed_gone",
                               side_effect=lambda pid: pid == 20):
            self.assertEqual(queue.foreign_owners([10, 20, 30], {10}), [30])

    def test_existing_result_is_refused_and_preserved(self):
        tmp = HERE / f"tmp_resource_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        existing = tmp / "result.json"
        try:
            existing.write_text("preserve-me", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "preserve existing"):
                queue.ensure_absent([existing])
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve-me")
        finally:
            if existing.exists():
                existing.unlink()
            tmp.rmdir()

    def test_validator_accepts_only_resource_record_without_scientific_admission(self):
        tmp = HERE / f"tmp_resource_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        path = tmp / "probe.json"
        try:
            record = {
                "experiment": "SW0135_native32_spike_binding",
                "status": "resource_probe_complete", "resource_only": True,
                "training_admission": False, "ground_truth_used": False,
                "optimizer_updates": 0,
                "resource_optimizer_steps": {"joint": 1, "head_decoder": 1},
                "image_id_count": 16, "microbatch_seconds": [0.1] * 16,
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(queue.validate_probe_record(path))
            record["training_admission"] = True
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(queue.validate_probe_record(path))
        finally:
            if path.exists():
                path.unlink()
            tmp.rmdir()


if __name__ == "__main__":
    unittest.main()
