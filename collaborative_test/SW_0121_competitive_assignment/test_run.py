"""CPU stage-dispatch and artifact-adoption regressions for SW0121."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0121_competitive_assignment import coordinator, run


class SW0121RunnerTests(unittest.TestCase):
    def fixture(self, name, content="fixture"):
        path = run.HERE / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite test fixture: {path}")
        path.write_text(content, encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_feature_cache_inverse_and_stage_plan_dependencies(self):
        flat = torch.arange(2 * 256 * 128, dtype=torch.float32).reshape(2, 256, 128)
        restored = run._features_to_spikes(flat)
        self.assertTrue(torch.equal(run.patch_features(restored), flat))
        plan = coordinator.task_plan()
        self.assertEqual([row["stage"] for row in plan],
                         ["control_validation", "preflight", "train", "evaluate"])
        self.assertEqual(plan[1]["depends_on"], [coordinator.CONTROL_ID])
        self.assertEqual(plan[2]["depends_on"], [plan[1]["task_id"]])
        self.assertEqual(plan[3]["depends_on"], [plan[2]["task_id"]])
        self.assertEqual(coordinator.command(plan[0]), [])
        self.assertIn("preflight", coordinator.command(plan[1]))
        self.assertIn(str(run.OUT / "seed0_analytic_candidate"), coordinator.command(plan[2]))
        self.assertIn("evaluation.json", coordinator.command(plan[3])[-1])

    def test_preflight_artifact_adoption_requires_complete_bound_two_optimizer_evidence(self):
        archive = run.HERE
        pf_path = self.fixture(".__sw0121_test_preflight")
        with mock.patch.multiple(run, ARCHIVE=archive, OUT=archive,
                                 WARMUP_HEAD=self.fixture("warmhead"),
                                 WARMUP_OPTIMIZER=self.fixture("warmopt"),
                                 WARMUP_FEATURES=self.fixture("warmfeatures"),
                                 WARMUP_METADATA=self.fixture("warmmeta"),
                                 LAMBDA_PATH=self.fixture("lambda")), \
                mock.patch.object(run, "implementation_fingerprint", return_value={"test": "sha"}), \
                mock.patch.object(coordinator, "artifact_path", return_value=pf_path):
                artifacts = {run.WARMUP_HEAD: "head", run.WARMUP_OPTIMIZER: "headopt",
                             run.WARMUP_FEATURES: "features", run.WARMUP_METADATA: "warmmeta"}
                for path, value in artifacts.items():
                    path.write_text(value, encoding="utf-8")
                lambda_path = run.LAMBDA_PATH
                row = {"old_norm": 1.0, "aux_norm": 0.5, "R_to_head_norm": 2.0,
                       "R_to_component_spikes_norm": 1.5, "C_to_Q_norm": 3.0,
                       "assignment_patch_std": 0.05,
                       "family_norms": {k: {"old_norm": 1., "aux_norm": .5}
                                        for k in ("encoder", "graph", "core")}}
                record = {
                    "status": "passed", "experiment": "SW0121", "seed": 0,
                    "implementation_fingerprint": {"test": "sha"},
                    "ground_truth_used": False, "matched_shuffle_seed": 117,
                    "training_ids": list(range(4096)), "calibration_batches": [row] * 4,
                    "lambda": 0.5, "warmup_R_mean": 0.2,
                    "global_mean_R_baseline": 0.3, "fixed_scramble_R_delta": 0.1,
                    "throwaway_update": {"passed": True, "changed": {
                        "encoder": True, "graph": True, "core": True, "head": True}},
                    "artifacts": {key: run.sha(path) for path, key in (
                        (run.WARMUP_HEAD, "head_sha256"),
                        (run.WARMUP_OPTIMIZER, "optimizer_sha256"),
                        (run.WARMUP_FEATURES, "features_sha256"),
                        (run.WARMUP_METADATA, "warmup_metadata_sha256"))}}
                run.write(pf_path, record)
                run.write(lambda_path, {"status": "passed", "lambda": 0.5,
                                         "preflight_sha256": run.sha(pf_path),
                                         "implementation_fingerprint": {"test": "sha"}})
                task = {"stage": "preflight", "seed": 0}
                self.assertTrue(coordinator.valid_result(task))
                bad = json.loads(pf_path.read_text(encoding="utf-8"))
                bad["ground_truth_used"] = True
                run.write(pf_path, bad)
                self.assertFalse(coordinator.valid_result(task))

    def test_train_and_evaluation_validators_reach_their_own_stage_contracts(self):
        archive = run.HERE
        warm_head = self.fixture(".__sw0121_train_warmhead")
        warm_opt = self.fixture(".__sw0121_train_warmopt")
        warm_features = self.fixture(".__sw0121_train_warmfeatures")
        with mock.patch.multiple(run, ARCHIVE=archive, OUT=archive,
                                 WARMUP_HEAD=warm_head, WARMUP_OPTIMIZER=warm_opt,
                                 WARMUP_FEATURES=warm_features), \
                mock.patch.object(run, "implementation_fingerprint", return_value={"test": "sha"}), \
                mock.patch.object(coordinator, "artifact_path", side_effect=lambda task:
                                  run.HERE / (".__sw0121_test_manifest" if task["stage"] == "train"
                                              else ".__sw0121_test_evaluation")):
                folder = run.HERE
                pf_path = self.fixture("preflight_seed0.json")
                preflight_record = {"lambda": 0.75, "training_ids_sha256": "ids",
                    "source_core_sha256": "src", "source_manifest_sha256": "srcmanifest",
                    "encoder_sha256": "enc", "preprocessing_sha256": "stats",
                    "gamma_train_sha256": "gamma", "gamma_train_manifest_sha256": "gammameta",
                    "rgb_cache_sha256": "rgb", "rgb_cache_manifest_sha256": "rgbmeta"}
                run.write(pf_path, preflight_record)
                for name in ("TRAINING_COMPLETED", "core.pt", "encoder.pt", "head.pt",
                             "core_optimizer.pt", "head_optimizer.pt", "optimizer.pt"):
                    self.fixture(name, name)
                history = [{k: 1.0 for k in ("old", "primary", "positive_actual_Q", "R", "C",
                                               "total", "core_gradient_norm_preclip",
                                               "head_gradient_norm_preclip")} for _ in range(256)]
                history_path = self.fixture("history.json")
                run.write(history_path, history)
                manifest = {"status": "training_complete", "experiment": "SW0121", "seed": 0,
                    "arm": "analytic_candidate", "updates": 256, "batch_size": 16,
                    "ground_truth_used_for_training": False,
                    "implementation_fingerprint": {"test": "sha"},
                    "preflight_sha256": run.sha(pf_path), "lambda": 0.75,
                    "training_ids_sha256": "ids", "source_core_sha256": "src",
                    "source_manifest_sha256": "srcmanifest", "encoder_source_sha256": "enc",
                    "preprocessing_sha256": "stats", "gamma_train_sha256": "gamma",
                    "gamma_train_manifest_sha256": "gammameta", "rgb_cache_sha256": "rgb",
                    "rgb_cache_manifest_sha256": "rgbmeta",
                    "head_warmup_sha256": run.sha(warm_head),
                    "head_warmup_optimizer_sha256": run.sha(warm_opt),
                    "head_warmup_features_sha256": run.sha(warm_features),
                    "core_sha256": run.sha(folder / "core.pt"),
                    "encoder_sha256": run.sha(folder / "encoder.pt"),
                    "head_sha256": run.sha(folder / "head.pt"),
                    "core_optimizer_sha256": run.sha(folder / "core_optimizer.pt"),
                    "head_optimizer_sha256": run.sha(folder / "head_optimizer.pt"),
                    "optimizer_sha256": run.sha(folder / "optimizer.pt"),
                    "history_sha256": run.sha(history_path)}
                manifest_path = run.HERE / ".__sw0121_test_manifest"
                run.write(manifest_path, manifest)
                self.addCleanup(lambda: manifest_path.unlink(missing_ok=True))
                train_task = {"stage": "train", "seed": 0}
                self.assertTrue(coordinator.valid_result(train_task))

                metrics = {key: 0.5 for key in ("fg_ari", "foreground_iou", "matched_object_iou")}
                score = {"metrics": metrics, "valid_count": {key: 320 for key in metrics},
                         "per_image": {key: [0.5] * 320 for key in metrics}}
                report = {"experiment": "SW0121", "seed": 0, "arm": "analytic_candidate",
                    "ids": [1320, 1639], "images": 320,
                    "ground_truth_used_for_prediction": False,
                    "assignment_head_used_for_prediction": False,
                    "training_manifest_sha256": run.sha(manifest_path),
                    "checkpoint_sha256": run.sha(folder / "core.pt"),
                    "encoder_checkpoint_sha256": run.sha(folder / "encoder.pt"),
                    "head_checkpoint_sha256": run.sha(folder / "head.pt"),
                    "evaluation_runner_sha256": "runner", "shared_evaluator_sha256": "eval",
                    "sweep": [{"scored_targets": {"our_hdf5": score}}]}
                eval_path = self.fixture(".__sw0121_test_evaluation")
                run.write(eval_path, report)
                sidecar_path = self.fixture("evaluation_manifest.json")
                run.write(sidecar_path, {
                    "experiment": "SW0121", "evaluation_file": eval_path.name,
                    "evaluation_sha256": run.sha(eval_path),
                    "training_manifest_sha256": run.sha(manifest_path),
                    "checkpoint_sha256": run.sha(folder / "core.pt"),
                    "evaluation_runner_sha256": "runner", "shared_evaluator_sha256": "eval",
                    "evaluation_contract": {"ids": [1320, 1639], "images": 320,
                        "batch_size": 8, "time_steps": 1024, "settle": 512,
                        "membrane_vth": 0.06, "readout_threshold": 0.50,
                        "min_group_size": 2, "background": "largest_component",
                        "ground_truth_used_for_prediction": False}})
                eval_task = {"stage": "evaluate", "seed": 0}
                self.assertTrue(coordinator.valid_result(eval_task))
                report["ground_truth_used_for_prediction"] = True
                run.write(eval_path, report)
                self.assertFalse(coordinator.valid_result(eval_task))


if __name__ == "__main__":
    unittest.main()
