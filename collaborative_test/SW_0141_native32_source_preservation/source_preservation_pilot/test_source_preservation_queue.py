"""CPU-only topology and command contract tests for the private queue draft."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_preservation_queue as queue


class QueueDraftTests(unittest.TestCase):
    def test_fixed_dag_contains_caches_calibration_preflights_arms_and_gtlast_score(self):
        tasks = queue.task_plan()
        by_id = {row["task_id"]: row for row in tasks}
        self.assertEqual(len(tasks), 10)
        for row in tasks:
            self.assertTrue(set(row["depends_on"]).issubset(by_id))
        self.assertEqual(by_id["anchor_lambda_s0"]["depends_on"], ["source_q_cache_s0"])
        for task_id in ("preflight_c_only_s1", "preflight_c_plus_anchor_s1"):
            self.assertEqual(set(by_id[task_id]["depends_on"]),
                             {"source_q_cache_s1", "anchor_lambda_s0"})
        self.assertEqual(set(by_id["score_seed1"]["depends_on"]),
                         {"predict_c_only_s1", "predict_c_plus_anchor_s1"})
        self.assertEqual(queue.stage_groups(),
                         (("cache",), ("calibrate",), ("preflight",),
                          ("train",), ("predict",), ("score",)))

    def test_cache_commands_bind_distinct_registered_counts_and_paths(self):
        zero = queue.command({"task_id": "z", "stage": "cache", "seed": 0})
        one = queue.command({"task_id": "o", "stage": "cache", "seed": 1})
        self.assertIn(str(queue.pilot.CACHE_ROOT / "source0_n64"), zero)
        self.assertIn(str(queue.pilot.CACHE_ROOT / "source1_n4096"), one)
        self.assertNotEqual(queue.output_paths({"stage": "cache", "seed": 0}),
                            queue.output_paths({"stage": "cache", "seed": 1}))

    def test_score_stage_is_cuda_hidden_while_gpu_stages_require_mapping(self):
        score = queue.environment_for({"stage": "score"})
        self.assertEqual(score["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(score["PATCH_V2_REPO"], str(queue.ROOT))
        self.assertNotIn("TRITON_CACHE_DIR", score)
        for stage in ("cache", "calibrate", "preflight", "train", "predict"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "exclusively reserved"):
                queue.environment_for({"stage": stage})
        env = queue.environment_for({"stage": "train"}, 3,
                                   triton_cache=Path("/tmp/private-triton"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(env["TRITON_CACHE_DIR"].replace("\\", "/").endswith("/tmp/private-triton"))

    def test_commands_never_launch_score_on_gpu_or_change_registered_driver(self):
        tasks = queue.task_plan()
        for task in tasks:
            argv = queue.command(task)
            self.assertEqual(Path(argv[1]), queue.HERE / "pilot.py")
            self.assertIn(task["stage"], argv)
        score = queue.command(next(t for t in tasks if t["stage"] == "score"))
        self.assertNotIn("--device", score)
        self.assertEqual(queue.MAX_PARALLEL, 2)
        self.assertFalse(queue.STATE.exists())


if __name__ == "__main__":
    unittest.main()
