"""Registered three-seed barrier and command-contract tests."""
import unittest

from collaborative_test.SW_0125_late_rollout_credit import coordinator


class CoordinatorTests(unittest.TestCase):
    def test_all_three_preflights_gate_each_training_seed(self):
        plan = coordinator.task_plan()
        by_id = {task["task_id"]: task for task in plan}
        preflights = {task["task_id"] for task in plan if task["stage"] == "preflight"}
        self.assertEqual(len(plan), 9)
        self.assertEqual(len(preflights), 3)
        for task in plan:
            if task["stage"] == "train":
                self.assertEqual(set(task["depends_on"]), preflights)
            elif task["stage"] == "evaluate":
                dependency = by_id[task["depends_on"][0]]
                self.assertEqual(dependency["stage"], "train")
                self.assertEqual(dependency["seed"], task["seed"])

    def test_stage_commands_bind_expected_outputs_and_fixed_evaluator(self):
        plan = coordinator.task_plan()
        sample = {stage: next(row for row in plan if row["seed"] == 1 and row["stage"] == stage)
                  for stage in ("preflight", "train", "evaluate")}
        pre = coordinator.command(sample["preflight"], device="cuda:0")
        train = coordinator.command(sample["train"], device="cuda:0")
        endpoint = coordinator.command(sample["evaluate"], device="cuda:0")
        self.assertIn("preflight", pre)
        self.assertIn("train", train)
        self.assertTrue(endpoint[1].endswith("evaluate.py"))
        self.assertIn(str(coordinator.run.OUT / "seed1_late_candidate" / "core.pt"), endpoint)
        self.assertIn(str(coordinator.artifact_path(sample["evaluate"])), endpoint)


if __name__ == "__main__":
    unittest.main()
