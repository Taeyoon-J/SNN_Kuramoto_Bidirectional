import numpy as np  # Keep NumPy before torch on Windows.
import contextlib
import io
import json
import hashlib
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from collaborative_test.SW_0117_joint_analytic_rgb import coordinator, run


class JointRunnerTests(unittest.TestCase):
    def test_task_plan_contains_only_seed0_matched_six_stages(self):
        tasks = coordinator.task_plan()
        self.assertEqual(len(tasks), 6)
        self.assertEqual({t["arm"] for t in tasks}, {"control", "analytic_candidate"})
        self.assertEqual({t["seed"] for t in tasks}, {0})
        by_id = {t["task_id"]: t for t in tasks}
        for arm in run.ARMS:
            pre = f"sw0117_preflight_s0_{arm}"
            train = f"sw0117_train_s0_{arm}"
            evaluate = f"sw0117_eval_s0_{arm}"
            self.assertEqual(by_id[train]["depends_on"],
                             [pre, coordinator.SOURCE_ENDPOINT_REVIEW_ID])
            self.assertIn(coordinator.SOURCE_ENDPOINT_REVIEW_ID, by_id[train]["depends_on"])
            self.assertEqual(by_id[evaluate]["depends_on"], [train])

    @staticmethod
    def rollout_parts(core, gamma, rgb, lossfn):
        value = gamma.sum() * core.offset
        labels = torch.tensor([[0, 1]], dtype=torch.long)
        hard = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        trace = gamma * getattr(core, "trace_scale", 1.0)
        return (value, value, value, value, value, labels, hard, [value], {},
                trace, trace * 2, trace * 3, trace * 4)

    def test_rollout_parity_separates_small_input_drift_from_same_input_identity(self):
        live = type("Core", (), {"offset": 1.0, "trace_scale": 100.0})()
        reference = type("Core", (), {"offset": 1.0, "trace_scale": 100.0})()
        cache = torch.tensor([[0.25]])
        regenerated = torch.tensor([[0.250001]])
        with mock.patch.object(run, "objective_parts", side_effect=self.rollout_parts):
            parts, parity = run.compare_initial_rollout(
                live, reference, regenerated, cache, None, None)
        self.assertGreater(parity["cache_vs_live_trace_max_abs_diff"]["spikes"], 2e-5)
        self.assertLessEqual(parity["cached_baseline_old_loss_abs_diff"], 2e-5)
        self.assertTrue(parity["cached_baseline_labels_H_exact"])
        self.assertTrue(parity["same_input_reference_traces_exact"])
        self.assertTrue(parity["same_input_reference_loss_exact"])
        self.assertTrue(torch.equal(parts[0], regenerated.sum()))

    def _source_review_fixture(self):
        def score(metrics):
            return {"mean": metrics, "valid_count": {name: 320 for name in metrics},
                    "per_image": {name: [value] * 320 for name, value in metrics.items()}}
        report = {
            "status": "complete", "source_checkpoint_sha256": "source",
            "encoder_sha256": "encoder", "preprocessing_sha256": "stats",
            "max_gamma_abs_diff": 7.152557e-7, "fixed_source_reproduced": True,
            "source_baseline_max_diffs": {"spikes": 0.0, "labels": 0.0},
            "ids": [1320, 1639], "images": 320, "time_steps": 1024,
            "settle": 512, "batch_size": 8,
            "ground_truth_used_for_prediction": False, "training_started": False,
            "scores": {
                "source16_vs_modal8": score(run.SOURCE_ENDPOINT_METRICS_CACHED),
                "control16_vs_modal8": score(run.SOURCE_ENDPOINT_METRICS_LIVE),
                "control_repeated32_vs_modal4": score(run.SOURCE_ENDPOINT_METRICS_LIVE),
            },
        }
        report_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        relative_path = "source_live_gamma_endpoint/evaluation.json"
        review = {
            "status": "reviewed", "experiment": "SW0117",
            "report_path": relative_path, "report_sha256": report_sha,
            "source_core_sha256": "source", "encoder_sha256": "encoder",
            "preprocessing_sha256": "stats", "fixed_source_reproduced": True,
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_training": False, "ids": [1320, 1639],
            "images": 320, "batch_size": 8, "time_steps": 1024,
            "settle": 512, "membrane_threshold": 0.06,
            "readout_threshold": 0.5, "max_gamma_cache_abs_diff": 7.152557e-7,
            "cached_metrics": run.SOURCE_ENDPOINT_METRICS_CACHED,
            "live_metrics": run.SOURCE_ENDPOINT_METRICS_LIVE,
        }
        return review, report, report_sha, relative_path

    def test_source_endpoint_review_binds_exact_report_and_endpoint_metrics(self):
        review, report, report_sha, rel = self._source_review_fixture()
        result = run._validate_source_endpoint_review_records(
            review, report, report_sha, report_sha, "source", "encoder", "stats", rel)
        self.assertEqual(result["report_sha256"], report_sha)

    def test_source_endpoint_review_rejects_tampered_report_or_wrong_source(self):
        review, report, report_sha, rel = self._source_review_fixture()
        tampered = dict(report, images=319)
        tampered_sha = hashlib.sha256(json.dumps(tampered, sort_keys=True).encode()).hexdigest()
        with self.assertRaises(AssertionError):
            run._validate_source_endpoint_review_records(
                review, tampered, tampered_sha, report_sha, "source", "encoder", "stats", rel)
        with self.assertRaises(AssertionError):
            run._validate_source_endpoint_review_records(
                review, report, report_sha, report_sha, "wrong-source", "encoder", "stats", rel)

    def test_training_refuses_without_review_before_creating_output(self):
        output = run.HERE / "test_output_must_not_be_created"
        self.assertFalse(output.exists())
        with mock.patch.object(run, "validate_source_endpoint_review",
                               side_effect=FileNotFoundError("review missing")):
            with self.assertRaisesRegex(FileNotFoundError, "review missing"):
                run.train(0, "control", output, device="cpu")
        self.assertFalse(output.exists())

    def test_rollout_parity_rejects_cached_oldloss_or_hard_partition_drift(self):
        reference = type("Core", (), {"offset": 1.0})()
        gamma, cached = torch.tensor([[0.25]]), torch.tensor([[0.25]])
        with mock.patch.object(run, "objective_parts", side_effect=self.rollout_parts):
            with self.assertRaisesRegex(AssertionError, "old loss"):
                run.compare_initial_rollout(type("Core", (), {"offset": 1.001})(),
                                            reference, gamma, cached, None, None)

            def changed_h(core, value, rgb, lossfn):
                parts = self.rollout_parts(core, value, rgb, lossfn)
                if core is not reference:
                    parts = list(parts)
                    parts[6] = [torch.tensor([[0.0, 1.0], [1.0, 0.0]])]
                return tuple(parts)

            with mock.patch.object(run, "objective_parts", side_effect=changed_h):
                with self.assertRaisesRegex(AssertionError, "labels/H"):
                    run.compare_initial_rollout(type("Core", (), {"offset": 1.0})(),
                                                reference, gamma, cached, None, None)

    def test_rollout_parity_rejects_same_input_implementation_mismatch(self):
        reference = type("Core", (), {"offset": 1.0})()
        live = type("Core", (), {"offset": 1.0000001})()
        gamma, cached = torch.tensor([[0.25]]), torch.tensor([[0.25]])
        with mock.patch.object(run, "objective_parts", side_effect=self.rollout_parts):
            with self.assertRaisesRegex(AssertionError, "same-input live/reference"):
                run.compare_initial_rollout(live, reference, gamma, cached, None, None)

    def test_train_command_writes_directory_containing_checkpoint_and_manifest(self):
        task = next(t for t in coordinator.task_plan() if t["stage"] == "train")
        argv = coordinator.command(task, "cuda:0")
        target = argv[argv.index("--output") + 1]
        self.assertEqual(target, str(coordinator.artifact_path(task).parent))
        self.assertEqual(coordinator.artifact_path(task).name, "manifest.json")

    def test_control_gradient_guard_does_not_require_control_rgb_gradient(self):
        control_stats = {name: {"old_gradient_norm": 2.0, "rgb_gradient_norm": 0.0}
                         for name in ("encoder", "graph", "core", "joint")}
        run.validate_gradient_families(control_stats, "control", q_gradient_norm=0.0)
        with self.assertRaises(AssertionError):
            run.validate_gradient_families(control_stats, "analytic_candidate", q_gradient_norm=1.0)
        candidate_stats = {name: {"old_gradient_norm": 2.0, "rgb_gradient_norm": 1.0}
                           for name in ("encoder", "graph", "core", "joint")}
        run.validate_gradient_families(candidate_stats, "analytic_candidate", q_gradient_norm=0.5)
        with self.assertRaises(AssertionError):
            run.validate_gradient_families(candidate_stats, "analytic_candidate", q_gradient_norm=0.0)

    def test_joint_gradient_application_matches_direct_combined_objective(self):
        core = nn.Module()
        core.graph_generator = nn.Linear(1, 1, bias=False)
        core.dynamics = nn.Linear(1, 1, bias=False)
        encoder = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            core.graph_generator.weight.fill_(0.7)
            core.dynamics.weight.fill_(0.9)
            encoder.weight.fill_(1.2)
        families = run.eligible_params(core, encoder)
        e = encoder.weight.sum()
        g = core.graph_generator.weight.sum()
        c = core.dynamics.weight.sum()
        old = (e * g + c).square()
        rgb = (e - c * g).square()
        stats, old_grads, rgb_grads = run.collect_joint_gradients(old, rgb, families, candidate=True)
        self.assertTrue(all(stats[name]["old_gradient_norm"] > 0 for name in ("encoder", "graph", "core")))
        self.assertTrue(all(stats[name]["rgb_gradient_norm"] > 0 for name in ("encoder", "graph", "core")))
        lam = 0.37
        params = families["all"]
        expected = torch.autograd.grad(old + lam * rgb, params, retain_graph=True)
        run.apply_family_gradients(families, old_grads, rgb_grads, lam)
        for param, grad in zip(params, expected):
            self.assertTrue(torch.allclose(param.grad, grad, rtol=0, atol=1e-7))

    def test_throwaway_change_guard_uses_independent_before_after_snapshots(self):
        core = nn.Module()
        core.graph_generator = nn.Linear(1, 1, bias=False)
        core.dynamics = nn.Linear(1, 1, bias=False)
        encoder = nn.Linear(1, 1, bias=False)
        before_core, before_encoder = run.clone_state(core), run.clone_state(encoder)
        with torch.no_grad():
            core.graph_generator.weight.add_(1)
            core.dynamics.weight.add_(1)
            encoder.weight.add_(1)
        self.assertEqual(run.changed_trainable_groups(before_core, core.state_dict(),
                                                      before_encoder, encoder.state_dict()),
                         {"graph": True, "core": True, "encoder": True})
        unchanged = run.changed_trainable_groups(before_core, before_core,
                                                 before_encoder, before_encoder)
        self.assertEqual(unchanged, {"graph": False, "core": False, "encoder": False})

    def test_preprocessing_stats_accept_scalar_or_channelwise_broadcast(self):
        mean, std, clip = run.preprocessing_tensors(
            {"mode": "standardize", "mean": torch.tensor(0.0),
             "std": torch.tensor(2.0), "clip": 3.0}, "cpu")
        self.assertEqual(tuple(mean.shape), ())
        self.assertEqual(float(std), 2.0)
        self.assertEqual(clip, 3.0)
        mean, std, _ = run.preprocessing_tensors(
            {"mode": "standardize", "mean": torch.zeros(8, 1, 1),
             "std": torch.ones(8, 1, 1), "clip": 3.0}, "cpu")
        self.assertEqual(tuple(mean.shape), (8, 1, 1))
        with self.assertRaises(AssertionError):
            run.preprocessing_tensors({"mode": "standardize", "mean": 0.0,
                                       "std": 0.0, "clip": 3.0}, "cpu")

    def test_encoder_path_scales_uint8_once_and_produces_registered_shape(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.last = None

            def forward(self, x):
                self.last = x.detach().clone()
                return x.mean(dim=(1, 2, 3), keepdim=True).expand(-1, 8, 126, 126)

        class Patcher(nn.Module):
            def forward(self, x):
                return torch.nn.functional.adaptive_avg_pool2d(x, (16, 16)).flatten(2)

        encoder = Encoder()
        images = torch.full((2, 3, 128, 128), 255.0)
        gamma = run.encode_rgb(encoder, Patcher(), torch.tensor(0.0), torch.tensor(1.0), 3.0, images)
        self.assertTrue(torch.equal(encoder.last, torch.ones_like(encoder.last)))
        self.assertEqual(tuple(gamma.shape), (2, 8, 256))
        self.assertTrue(torch.isfinite(gamma).all())

    def test_cli_reports_parseable_preflight_completion(self):
        old_argv = run.sys.argv
        output = io.StringIO()
        try:
            run.sys.argv = ["run.py", "preflight", "--seed", "0", "--arm", "control",
                            "--device", "cpu", "--output", "unused.json"]
            with mock.patch.object(run, "preflight", return_value={"status": "passed"}), \
                    contextlib.redirect_stdout(output):
                run.main()
        finally:
            run.sys.argv = old_argv
        self.assertEqual(json.loads(output.getvalue()), {
            "status": "passed", "command": "preflight", "seed": 0, "arm": "control"})

    def test_evaluation_score_validator_checks_exact_metrics_finite_values_and_means(self):
        names = ("fg_ari", "foreground_iou", "matched_object_iou")
        score = {"metrics": {}, "valid_count": {}, "per_image": {}}
        for name in names:
            score["metrics"][name] = 0.5
            score["valid_count"][name] = 320
            score["per_image"][name] = [0.5] * 320
        self.assertTrue(coordinator._finite_score(score))
        score["per_image"]["fg_ari"][3] = float("nan")
        self.assertFalse(coordinator._finite_score(score))
        score["per_image"]["fg_ari"][3] = 0.5
        score["metrics"]["foreground_iou"] = 0.6
        self.assertFalse(coordinator._finite_score(score))


if __name__ == "__main__":
    unittest.main()
