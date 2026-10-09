"""CPU checks for the precise SW0119 loss ablation and evaluator guard."""
import numpy as np
import unittest
import json
from pathlib import Path
from unittest import mock

import torch

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0119_density_free_objective import coordinator, run


class SW0119DensityFreeTests(unittest.TestCase):
    def test_only_balance_terms_are_removed_and_gradient_decomposition_matches(self):
        torch.manual_seed(119)
        full_fn = base.criterion()
        plv = torch.sigmoid(torch.randn(1, 256, 256, dtype=torch.float64, requires_grad=True))
        q = torch.sigmoid(torch.randn(1, 256, 256, dtype=torch.float64, requires_grad=True))
        plv.retain_grad(); q.retain_grad()
        full_p, pparts = full_fn(plv=plv)
        free_p, removed_p, full_p_again, _, free_parts = run.loss_without_density(full_fn, plv)
        full_q, qparts = full_fn(plv=q)
        free_q, removed_q, full_q_again, _, _ = run.loss_without_density(full_fn, q)
        self.assertTrue(torch.equal(full_p, full_p_again))
        self.assertTrue(torch.equal(full_q, full_q_again))
        self.assertTrue(torch.allclose(full_p - free_p, removed_p, rtol=1e-6, atol=2e-6))
        self.assertTrue(torch.allclose(full_q - free_q, removed_q, rtol=1e-6, atol=2e-6))
        removed = removed_p + 5.0 * removed_q
        full_objective = full_p + 5.0 * full_q
        free_objective = free_p + 5.0 * free_q
        self.assertTrue(torch.allclose(full_objective - free_objective, removed,
                                       rtol=1e-6, atol=2e-6))
        g_full = torch.autograd.grad(full_objective, (plv, q), retain_graph=True)
        g_free = torch.autograd.grad(free_objective, (plv, q), retain_graph=True)
        g_removed = torch.autograd.grad(removed, (plv, q))
        for a, b, c in zip(g_full, g_free, g_removed):
            self.assertTrue(torch.allclose(a - b, c, rtol=1e-5, atol=2e-7))
        self.assertEqual(full_fn.plv_balance_weight, 10.0)
        self.assertEqual(full_fn.plv_bimodality_weight, 6.0)
        self.assertEqual(full_fn.plv_coherence_weight, 0.5)
        self.assertEqual(full_fn.plv_collapse_weight, 1.0)
        self.assertIn("plv_balance", pparts)
        self.assertIn("plv_balance", qparts)
        self.assertNotIn("plv_balance", free_parts)

    def test_task_plan_has_one_fixed_candidate_and_strict_dependencies(self):
        tasks = coordinator.task_plan()
        self.assertEqual([t["stage"] for t in tasks],
                         ["validate-control", "preflight", "train", "evaluate"])
        self.assertEqual({t["arm"] for t in tasks}, {"analytic_candidate"})
        for prior, task in zip(tasks, tasks[1:]):
            self.assertEqual(task["depends_on"], [prior["task_id"]])
        self.assertIn("seed0_density_free_candidate", str(coordinator.artifact_path(tasks[2])))

    def test_score_validator_checks_finite_per_image_mean_and_count(self):
        names = ("fg_ari", "foreground_iou", "matched_object_iou")
        values = [0.25] * 320
        valid = {"metrics": {n: 0.25 for n in names},
                 "valid_count": {n: 320 for n in names},
                 "per_image": {n: values[:] for n in names}}
        self.assertTrue(coordinator._finite_score(valid))
        bad = json_roundtrip(valid)
        bad["per_image"][names[0]][10] = float("nan")
        self.assertFalse(coordinator._finite_score(bad))
        bad = json_roundtrip(valid)
        bad["metrics"][names[0]] = 0.251
        self.assertFalse(coordinator._finite_score(bad))

    def test_preflight_validator_accepts_recorded_three_family_schema(self):
        task = coordinator.task_plan()[1]
        artifact = Path("C:/fixture/preflight.json")
        families = ("encoder", "graph", "core", "joint")
        batches = []
        for index in range(4):
            batches.append({
                "global_ids": list(range(index * 16, (index + 1) * 16)),
                "cached_labels_H_exact": True,
                "same_input_reference_identity_exact": True,
                "cache_gamma_max_abs_diff": 0.0,
                "cached_baseline_old_loss_abs_diff": 0.0,
                "legacy_minus_density_free": 0.5,
                "removed_balance_terms": 0.5,
                "old_gradient_norms": {key: 1.0 for key in families},
                "rgb_gradient_norms": {key: 0.5 for key in families},
                "removed_balance_gradient_decomposition": {
                    key: {"relative_max_abs_difference": 0.0}
                    for key in ("encoder", "graph", "core")},
                "rgb_to_Q_gradient_norm": 0.25,
            })
        record = {"status": "passed", "experiment": "SW0119", "seed": 0,
                  "arm": "analytic_candidate", "implementation_fingerprint": {"fixture": "sha"},
                  "lambda_joint": run.LAMBDA, "lambda_sha256": "a" * 64,
                  "ground_truth_used": False, "batches": batches,
                  "throwaway_parameter_groups_changed": {
                      "graph": True, "core": True, "encoder": True},
                  "throwaway_b16_adam_update": True, "throwaway_gradient_norm_preclip": 1.0,
                  "training_ids": list(range(4096)), "matched_shuffle_seed": 117,
                  "scrambled_minus_real_rgb_loss": 0.1, "source_core_sha256": "b" * 64}
        serialized = json.dumps(record)
        with (mock.patch.object(coordinator, "artifact_path", return_value=artifact),
              mock.patch.object(Path, "is_file", return_value=True),
              mock.patch.object(Path, "read_text", return_value=serialized),
              mock.patch.object(run, "implementation_fingerprint", return_value={"fixture": "sha"}),
              mock.patch.object(run, "sha", return_value="a" * 64)):
            self.assertTrue(coordinator.valid_result(task))

    def test_train_result_reaches_training_validator_and_accepts_complete_fixture(self):
        task = coordinator.task_plan()[2]
        artifact = Path("C:/fixture/seed0/manifest.json")
        history = [{"update": i, "total": 1.0, "density_free_old": 0.5,
                    "primary": 0.2, "positive_actual_spike_product": 0.3,
                    "rgb": 0.1, "gradient_norm_preclip": 0.9}
                   for i in range(1, 257)]
        preflight = {"source_core_sha256": "b" * 64,
                     "source_manifest_sha256": "c" * 64,
                     "training_ids_sha256": "d" * 64}
        record = {"status": "training_complete", "experiment": "SW0119", "seed": 0,
                  "arm": "analytic_candidate", "updates": 256, "batch_size": 16,
                  "matched_shuffle_seed": 117, "ground_truth_used_for_training": False,
                  "implementation_fingerprint": {"fixture": "sha"},
                  "preflight_sha256": "a" * 64,
                  "source_core_sha256": preflight["source_core_sha256"],
                  "source_manifest_sha256": preflight["source_manifest_sha256"],
                  "training_ids_sha256": preflight["training_ids_sha256"],
                  "lambda_joint": run.LAMBDA, "lambda_artifact_sha256": "a" * 64,
                  "core_sha256": "a" * 64, "encoder_sha256": "a" * 64,
                  "optimizer_sha256": "a" * 64, "history_sha256": "a" * 64}

        def read_fixture(path, *args, **kwargs):
            if path.name == "history.json":
                return json.dumps(history)
            if path.name == "preflight_seed0_density_free.json":
                return json.dumps(preflight)
            return json.dumps(record)

        with (mock.patch.object(coordinator, "artifact_path", return_value=artifact),
              mock.patch.object(Path, "is_file", return_value=True),
              mock.patch.object(Path, "read_text", new=read_fixture),
              mock.patch.object(run, "implementation_fingerprint", return_value={"fixture": "sha"}),
              mock.patch.object(run, "sha", return_value="a" * 64)):
            self.assertTrue(coordinator.valid_result(task))


def json_roundtrip(value):
    # JSON-like copy that keeps ordinary numeric types and fresh nested lists.
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
