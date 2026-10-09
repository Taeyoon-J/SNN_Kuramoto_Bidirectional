import unittest
import json
from pathlib import Path
from unittest import mock

import coordinator
from coordinator import task_plan
from collaborative_test.SW_0110_xy_graph_route import run as common


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

    def test_train_cache_accepts_physical_hdf5_shape_but_rejects_wrong_geometry(self):
        task = next(t for t in task_plan() if t["task_id"] == "sw0114_cache_train_s0")
        source = Path("/frozen/source.pt")
        source_manifest = Path("/frozen/source_manifest.json")
        ids = list(range(4096))
        meta = {
            "gamma32_sha256": "gamma-sha", "seed": 0, "count": 4096, "ids": ids,
            "source_core_sha256": "source-sha", "source_manifest_sha256": "manifest-sha",
            "encoder_sha256": common.EXPECTED_ENCODER_SHA256,
            "preprocessing_sha256": common.EXPECTED_PREPROCESSING_SHA256,
            "selected_rgb_sha256": "rgb-sha",
            "dataset_identity": {"image_shape": [100000, 128, 128, 3]},
            "registered_gamma16_max_abs_diff": 1e-6,
        }

        def fake_sha(path):
            return {Path(task["output"]): "gamma-sha", source: "source-sha",
                    source_manifest: "manifest-sha"}[Path(path)]

        with mock.patch.object(common, "source_paths",
                               return_value=(source, source_manifest, {"training_ids": ids})), \
             mock.patch.object(coordinator.experiment, "sha", side_effect=fake_sha), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "read_text", side_effect=lambda *args, **kwargs: json.dumps(meta)):
            self.assertTrue(coordinator.valid_result(task))
            meta["dataset_identity"]["image_shape"] = [70640, 128, 128, 3]
            self.assertFalse(coordinator.valid_result(task))


if __name__ == "__main__":
    unittest.main()
