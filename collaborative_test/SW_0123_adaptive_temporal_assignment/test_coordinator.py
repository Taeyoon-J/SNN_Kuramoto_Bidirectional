import unittest
import os
from pathlib import Path
from unittest.mock import patch

from coordinator import _score_valid, artifact_path, command, task_plan
import dispatcher
from dispatcher import (Dispatcher, _pid_is_confirmed_gone,
                        adaptive_preflight_all_passed, foreign_owner_pids)


class QueuePlanTests(unittest.TestCase):
    def test_compute_owners_allow_worker_descendants_but_reject_foreign_processes(self):
        pid = os.getpid()
        self.assertEqual(foreign_owner_pids([pid], set()), [pid])
        self.assertEqual(foreign_owner_pids([pid], {pid}), [])
        self.assertEqual(foreign_owner_pids([], {101}), [])

    def test_vanished_compute_pid_is_ignored_but_live_foreign_pid_is_retained(self):
        vanished = 2**30
        with patch.object(dispatcher.os, "name", "posix"), \
                patch("dispatcher.pathlib.Path.stat", side_effect=FileNotFoundError):
            self.assertTrue(_pid_is_confirmed_gone(vanished))
        with patch.object(dispatcher.os, "name", "posix"), \
                patch("dispatcher.pathlib.Path.stat", side_effect=PermissionError):
            self.assertFalse(_pid_is_confirmed_gone(vanished))
        with patch("dispatcher._pid_is_confirmed_gone", return_value=True):
            self.assertEqual(foreign_owner_pids([vanished], set()), [])
        self.assertEqual(foreign_owner_pids([os.getpid()], set()), [os.getpid()])

    def test_one_adaptive_preflight_failure_blocks_all_training_tasks(self):
        tasks = task_plan()
        rows = {task["task_id"]: {"task": task, "status": "passed"} for task in tasks}
        failed_adaptive = rows["sw0123_preflight_s1_adaptive_full"]
        failed_adaptive["status"] = "failed"
        self.assertFalse(adaptive_preflight_all_passed(list(rows.values())))
        dispatcher = Dispatcher.__new__(Dispatcher)
        dispatcher.tasks = tasks
        dispatcher.state = {"tasks": rows}
        updates = []
        with patch.object(dispatcher, "_update",
                          side_effect=lambda task_id, **fields: updates.append((task_id, fields))):
            dispatcher._run_phase("train")
        train_rows = [task for task in tasks if task["stage"] == "train"]
        self.assertEqual(len(updates), len(train_rows))
        self.assertTrue(all(fields["status"] == "blocked_adaptive_recipe_guard"
                            for _, fields in updates))

    def test_all_nine_trajectories_have_ordered_preflight_train_endpoint(self):
        tasks = task_plan()
        by_id = {task["task_id"]: task for task in tasks}
        self.assertEqual(len(tasks), 27)
        self.assertEqual(len(by_id), len(tasks))
        for seed in range(3):
            for arm in ("legacy_full", "adaptive_full", "gate_only_control"):
                pre = by_id[f"sw0123_preflight_s{seed}_{arm}"]
                train = by_id[f"sw0123_train_s{seed}_{arm}"]
                endpoint = by_id[f"sw0123_eval_s{seed}_{arm}"]
                self.assertEqual(pre["depends_on"], [])
                self.assertEqual(train["depends_on"], [pre["task_id"]])
                self.assertEqual(endpoint["depends_on"], [train["task_id"]])
                self.assertEqual(artifact_path(endpoint).name, "evaluation.json")
                self.assertEqual(command(train, "cuda:1")[-1], str(artifact_path(train).parent))
                self.assertEqual(command(endpoint, "cuda:1")[-1], str(artifact_path(endpoint)))

    def test_metric_validator_requires_complete_finite_per_image_arrays(self):
        names = ("fg_ari", "foreground_iou", "matched_object_iou")
        score = {"metrics": {name: 0.5 for name in names},
                 "valid_count": {name: 320 for name in names},
                 "per_image": {name: [0.5] * 320 for name in names}}
        self.assertTrue(_score_valid(score))
        broken = {**score, "per_image": {**score["per_image"], "fg_ari": [float("nan")] + [0.5] * 319}}
        self.assertFalse(_score_valid(broken))
        incomplete = {**score, "valid_count": {**score["valid_count"], "foreground_iou": 319}}
        self.assertFalse(_score_valid(incomplete))


if __name__ == "__main__":
    unittest.main()
