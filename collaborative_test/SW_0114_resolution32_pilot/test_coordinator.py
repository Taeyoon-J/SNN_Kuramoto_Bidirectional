import unittest

from coordinator import task_plan


class CoordinatorTests(unittest.TestCase):
    def test_unique_tasks_and_shared_cache_singletons(self):
        tasks = task_plan()
        ids = [t["task_id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(tasks), 20)
        self.assertEqual(sum(t["stage"] == "cache_validation" for t in tasks), 1)
        self.assertEqual(sum(t["stage"] == "cache_training" for t in tasks), 3)

    def test_training_requires_arm_preflight_and_control_microbatch_audit(self):
        by_id = {t["task_id"]: t for t in task_plan()}
        for grid in (16, 32):
            for seed in (0, 1, 2):
                task = by_id[f"sw0114_train_g{grid}_s{seed}"]
                self.assertIn(f"sw0114_preflight_g{grid}_s{seed}", task["depends_on"])
                self.assertIn("sw0114_preflight_g16_s0", task["depends_on"])
        for seed in (0, 1, 2):
            self.assertIn("sw0114_cache_train_s" + str(seed),
                          by_id[f"sw0114_preflight_g32_s{seed}"]["depends_on"])

    def test_evaluation_waits_for_both_arms_and_validation_cache(self):
        by_id = {t["task_id"]: t for t in task_plan()}
        for seed in (0, 1, 2):
            deps = by_id[f"sw0114_evaluate_s{seed}"]["depends_on"]
            self.assertIn(f"sw0114_train_g16_s{seed}", deps)
            self.assertIn(f"sw0114_train_g32_s{seed}", deps)
            self.assertIn("sw0114_cache_val", deps)
        summary = by_id["sw0114_summary"]
        self.assertEqual(set(summary["depends_on"]), {f"sw0114_evaluate_s{s}" for s in (0, 1, 2)})


if __name__ == "__main__":
    unittest.main()
