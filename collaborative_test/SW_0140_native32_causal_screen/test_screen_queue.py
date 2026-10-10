from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0140_native32_causal_screen import screen_queue


class SW0140QueueTests(unittest.TestCase):
    def test_plan_is_three_ordered_exclusive_screen_tasks_only(self):
        tasks = screen_queue.task_plan()
        self.assertEqual([task["seed"] for task in tasks], [0, 1, 2])
        self.assertEqual([task["depends_on"] for task in tasks], [[], ["screen_seed0"], ["screen_seed1"]])
        self.assertTrue(all(task["stage"] == "screen" for task in tasks))
        self.assertTrue(all("--device" in screen_queue.command(task) for task in tasks))

    def test_terminal_failure_blocks_later_seeds_and_preserves_order(self):
        queue = screen_queue.ScreenQueue.__new__(screen_queue.ScreenQueue)
        queue.tasks = screen_queue.task_plan()
        queue.guard = __import__("threading").RLock()
        queue.state = {"tasks": {t["task_id"]: {"status": "queued"} for t in queue.tasks}}
        queue._unlock = lambda: None
        invoked = []

        def execute(task):
            invoked.append(task["seed"])
            return task["seed"] == 0

        queue._execute = execute
        with mock.patch.object(screen_queue, "_atomic", side_effect=lambda _p, _d: None):
            result = queue.run()
        self.assertFalse(result)
        self.assertEqual(invoked, [0, 1])
        self.assertEqual(queue.state["status"], "screen_incomplete")
        self.assertEqual(queue.state["task_statuses"],
                         {"screen_seed0": "passed", "screen_seed1": "failed", "screen_seed2": "blocked"})

    def test_foreign_gpu_owners_are_not_ignored(self):
        with mock.patch.object(screen_queue.owner, "foreign_owner_pids", return_value=[99123]) as fn:
            self.assertEqual(screen_queue.foreign_owners([123, 99123], {123, 124}), [99123])
        fn.assert_called_once_with([123, 99123], {123, 124})


if __name__ == "__main__":
    unittest.main()
