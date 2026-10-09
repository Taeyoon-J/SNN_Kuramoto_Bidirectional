import unittest

import numpy as np

from collaborative_test.SW_0122_joint_rgb_seed_replication import coordinator, run


class SW0122ReplicationTests(unittest.TestCase):
    def test_plan_pairs_both_arms_only_for_seeds_one_and_two(self):
        tasks = coordinator.task_plan()
        scientific = [t for t in tasks if t["seed"] is not None]
        self.assertEqual(len(scientific), 12)
        self.assertEqual({t["seed"] for t in scientific}, {1, 2})
        self.assertEqual({t["arm"] for t in scientific}, set(run.ARMS))
        ids = [t["task_id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(next(t for t in tasks if t["stage"] == "validate-baselines")["seed"], None)
        for task in scientific:
            if task["stage"] == "preflight":
                self.assertEqual(task["depends_on"], ["sw0122_validate_seed0_references"])
            if task["stage"] == "train":
                self.assertEqual(len(task["depends_on"]), 1)

    def test_commands_bind_each_seed_and_do_not_reuse_seed0_outputs(self):
        for seed in (1, 2):
            task = next(t for t in coordinator.task_plan()
                        if t["stage"] == "train" and t["seed"] == seed
                        and t["arm"] == "analytic_candidate")
            argv = coordinator.command(task, "cuda:1")
            self.assertIn(str(seed), argv)
            self.assertIn("analytic_candidate", argv)
            self.assertIn("cuda:1", argv)
            output = argv[argv.index("--output") + 1]
            self.assertIn(f"seed{seed}_analytic_candidate", output)
            self.assertNotIn("SW0117_joint_analytic_rgb", output)

    def test_source_baseline_contract_requires_exact_ids_and_finite_full_metrics(self):
        ids = np.arange(4096, dtype=np.int64)
        metrics = {"fg_ari": 0.7, "foreground_iou": 0.6, "matched_object_iou": 0.5}
        score = {"metrics": metrics,
                 "valid_count": {key: 320 for key in metrics},
                 "per_image": {key: [value] * 320 for key, value in metrics.items()}}
        manifest = {"status": "complete", "arm": "positive_frozen", "source_model_seed": 1,
                    "unique_images_seen": 4096, "steps": 256, "batch": 16, "seed": 118,
                    "training_ids": ids.tolist(), "ground_truth_used_for_training": False}
        report = {"checkpoint": "/immutable/seed1/core.pt", "ids": [1320, 1639], "images": 320,
                  "ground_truth_used_for_prediction": False,
                  "sweep": [{"scored_targets": {"our_hdf5": score}}]}
        validated = run._validate_source_evaluation_records(1, "/immutable/seed1/core.pt",
                                                            ids, manifest, report)
        self.assertEqual(validated, metrics)
        report["ids"] = [1321, 1640]
        with self.assertRaises(AssertionError):
            run._validate_source_evaluation_records(1, "/immutable/seed1/core.pt", ids, manifest, report)

    def test_historical_lambda_must_match_seed0_calibration_artifact(self):
        summary = {"status": "completed_seed0_not_promoted", "lambda_joint": run.LAMBDA}
        lambda_record = {"status": "passed", "seed": 0, "lambda_joint": run.LAMBDA,
                         "preflight_sha256": "a" * 64, "implementation_fingerprint": {"runner": "sha"}}
        self.assertEqual(run._validate_seed0_lambda_records(
            summary, lambda_record, "a" * 64, {"runner": "sha"}), run.LAMBDA)
        lambda_record["lambda_joint"] += 1e-9
        with self.assertRaises(AssertionError):
            run._validate_seed0_lambda_records(summary, lambda_record, "a" * 64, {"runner": "sha"})


if __name__ == "__main__":
    unittest.main()
