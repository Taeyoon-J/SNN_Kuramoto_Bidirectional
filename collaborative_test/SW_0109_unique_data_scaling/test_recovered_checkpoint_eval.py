import json
import shutil
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import recovered_checkpoint_eval as queue


class CheckpointOnlyQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture_root = queue.HERE / ".checkpoint_eval_test_fixture"
        if self.fixture_root.exists():
            raise RuntimeError(f"preserve pre-existing test fixture path: {self.fixture_root}")
        self.fixture_root.mkdir()
        self.fixture_assets = self.fixture_root / "assets"
        self.fixture_assets.mkdir()
        for name in ("train.pt", "train.json", "validation.pt", "validation.json", "dataset.h5"):
            (self.fixture_assets / name).write_bytes(name.encode("ascii"))
        self.asset_patch = mock.patch.multiple(
            queue,
            TRAIN_GAMMA=self.fixture_assets / "train.pt",
            TRAIN_MANIFEST=self.fixture_assets / "train.json",
            VAL_GAMMA=self.fixture_assets / "validation.pt",
            VAL_MANIFEST=self.fixture_assets / "validation.json",
            DATASET=self.fixture_assets / "dataset.h5")
        self.asset_patch.start()
        self.addCleanup(self.asset_patch.stop)

    def tearDown(self):
        if self.fixture_root.exists() and self.fixture_root.parent.resolve() == queue.HERE.resolve():
            shutil.rmtree(self.fixture_root)

    def test_plan_is_three_existing_seed0_cores_and_never_marks_training_complete(self):
        tasks = queue.task_plan()
        self.assertEqual([t["size"] for t in tasks], [2500, 10000, 70000])
        self.assertEqual(len({t["checkpoint"] for t in tasks}), 3)
        self.assertTrue(all(t["seed"] == 0 for t in tasks))
        self.assertTrue(all("not certified complete" in t["training_manifest_status"] for t in tasks))
        self.assertTrue(all(t["stage"] == "checkpoint-only-eval" for t in tasks))

    def test_command_uses_registered_cache_and_dataset_arguments(self):
        task = queue.task_plan()[0]
        argv = queue.command(task, "cuda:0")
        for expected in ("--stage", "eval", "--seed", "0", "--size", "2500",
                         "--train-gamma", str(queue.TRAIN_GAMMA), "--val-gamma", str(queue.VAL_GAMMA),
                         "--dataset", str(queue.DATASET), "--device", "cuda:0"):
            self.assertIn(expected, argv)
        self.assertEqual(argv[argv.index("--checkpoint") + 1], task["checkpoint"])

    def test_gpu_owner_check_accepts_only_confirmed_child_processes(self):
        parents = {301: 200, 200: 100, 401: 1}
        lookup = lambda pid: parents.get(pid)
        self.assertTrue(queue.is_descendant(301, 100, lookup))
        self.assertFalse(queue.is_descendant(401, 100, lookup))
        self.assertFalse(queue.is_descendant(999, 100, lookup))

    def test_owner_check_ignores_only_confirmed_vanished_pid(self):
        parents = {401: 1, 501: 200, 200: 100}
        lookup = lambda pid: parents.get(pid)
        self.assertEqual(queue.foreign_owners([401, 999], 100, lookup,
                                              lambda pid: pid == 401), [401])
        self.assertEqual(queue.foreign_owners([999], 100, lookup,
                                              lambda pid: False), [])

    def test_owner_check_bounds_parent_walk(self):
        self.assertFalse(queue.is_descendant(1000, 1, lambda pid: pid - 1))

    def _fixture(self):
        root = self.fixture_root
        checkpoint = root / "core.pt"
        checkpoint.write_bytes(b"immutable-checkpoint-fixture")
        task = {"task_id": "fixture", "size": 2500, "checkpoint": str(checkpoint),
                "checkpoint_sha256": queue.sha256(checkpoint), "output": str(root / "result")}
        out = Path(task["output"])
        out.mkdir()
        values = {name: [float(i) / 320 for i in range(320)] for name in queue.METRICS}
        means = {name: float(np.mean(array)) for name, array in values.items()}
        score = {"per_image": values, "metrics": means,
                 "valid_count": {name: 320 for name in queue.METRICS}}
        report = {"ids": [1320, 1639], "images": 320,
                  "ground_truth_used_for_prediction": False,
                  "sweep": [{"scored_targets": {"our_hdf5": score}}]}
        (out / "evaluation.json").write_text(json.dumps(report))
        manifest = {"status": "complete", "seed": 0, "pool_size": 2500,
                    "checkpoint_sha256": task["checkpoint_sha256"],
                    "checkpoint": str(checkpoint.resolve()),
                    "runner_sha256": queue.EXPECTED_RUNNER_SHA256,
                    "validation_gamma_sha256": queue.sha256(queue.VAL_GAMMA),
                    "validation_gamma_manifest_sha256": queue.sha256(queue.VAL_MANIFEST),
                    "dataset_path": str(queue.DATASET.resolve()),
                    "metrics": means, "valid_count": {name: 320 for name in queue.METRICS},
                    "ids": [1320, 1639], "images": 320,
                    "ground_truth_used_for_prediction": False}
        (out / "evaluation_manifest.json").write_text(json.dumps(manifest))
        (out / "COMPLETED").write_text("evaluation only; training remains uncertified\n")
        return task, out

    def test_validates_full_finite320_report_and_recomputed_means(self):
        task, _ = self._fixture()
        self.assertTrue(queue.valid_result(task))

    def test_rejects_nan_or_mean_mismatch(self):
        task, out = self._fixture()
        report_path = out / "evaluation.json"
        data = json.loads(report_path.read_text())
        data["sweep"][0]["scored_targets"]["our_hdf5"]["per_image"]["fg_ari"][4] = float("nan")
        report_path.write_text(json.dumps(data))
        self.assertFalse(queue.valid_result(task))

    def test_rejects_wrong_checkpoint_provenance(self):
        task, out = self._fixture()
        path = out / "evaluation_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        self.assertFalse(queue.valid_result(task))

    def test_rejects_manifest_metric_mean_that_disagrees_with_report(self):
        task, out = self._fixture()
        path = out / "evaluation_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["metrics"]["fg_ari"] += 0.01
        path.write_text(json.dumps(manifest))
        self.assertFalse(queue.valid_result(task))


if __name__ == "__main__":
    unittest.main()
