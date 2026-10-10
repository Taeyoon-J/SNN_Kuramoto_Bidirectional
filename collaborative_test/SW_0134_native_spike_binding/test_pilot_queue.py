"""Focused contracts for the SW0134 seed-1 train/evaluation queue."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0134_native_spike_binding import pilot_queue as queue
from collaborative_test.SW_0134_native_spike_binding import train_diagnostic
from collaborative_test.SW_0134_native_spike_binding import run
from collaborative_test.SW_0134_native_spike_binding.binder import NativeSpikeSlotBinder, RelativeSlotRGBDecoder
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.test_train_evaluate import _source_core


class _DiagnosticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


class PilotQueueTests(unittest.TestCase):
    def test_real_child_entrypoints_import_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for task in (queue.task_plan()[0], queue.task_plan()[3], queue.task_plan()[4]):
            argv = queue.command(task)
            self.assertEqual(argv[1], "-m")
            child = subprocess.run(argv[:3] + ["--help"], cwd=queue.ROOT,
                                   env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("--seed", child.stdout)

    def test_registered_stage_dependencies_and_outputs(self):
        tasks = queue.task_plan()
        self.assertEqual([t["stage"] for t in tasks], ["train"] * 3 + ["diagnostic", "evaluate"])
        self.assertEqual({t["arm"] for t in tasks[:3]}, set(queue.ARMS))
        self.assertEqual(tasks[3]["depends_on"], [t["task_id"] for t in tasks[:3]])
        self.assertTrue(all(t["depends_on"] == ["preflight_seed0", "preflight_seed1",
                                                 "preflight_seed2"] for t in tasks[:3]))
        self.assertIn("--arm", queue.command(tasks[0]))
        self.assertIn("evaluation_seed1", str(queue.artifact_path(tasks[4])))

    def test_evaluator_validator_uses_real_nested_score_schema(self):
        root = Path(tempfile.gettempdir()) / f"sw134_queue_eval_{uuid.uuid4().hex[:10]}"
        root.mkdir()
        try:
            task = queue.task_plan()[4]
            out = root / "eval"
            out.mkdir()
            pred = root / "predictions.pt"
            pred.write_bytes(b"prediction-array-fixture")
            manifest = pred.parent / "prediction_manifest.json"
            source_path = root / "source_evaluation.json"
            source_path.write_text("{}", encoding="utf-8")
            predictions = {arm: {kind: __import__("torch").zeros((320, 16, 16), dtype=
                               __import__("torch").int64) for kind in ("primary", "qcc")}
                           for arm in ("source97_native", *queue.ARMS)}
            torch = __import__("torch")
            scores = {}
            for arm in ("source97_native", *queue.ARMS):
                scores[arm] = {}
                for readout in ("primary", "qcc"):
                    scores[arm][readout] = {}
                    for metric in queue.REPORT_METRICS:
                        values = [0.25] * 320
                        scores[arm][readout][metric] = {
                            "per_image": values, "mean": 0.25, "valid_count": 320}
            torch.save({"image_ids": list(range(1320, 1640)), "predictions": predictions}, pred)
            prediction_sha = queue.run.sha(pred)
            manifest_payload = {"seed": 1, "count": 320, "image_ids": [1320, 1639],
                                "arms": ["source97_native", *queue.ARMS],
                                "prediction_sha256": prediction_sha,
                                "ground_truth_used_for_prediction": False}
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            prediction_manifest_sha = queue.run.sha(manifest)
            training_provenance = {}
            training_manifests = {}
            checkpoints = {}
            for arm in queue.ARMS:
                checkpoint = root / f"{arm}.pt"
                checkpoint.write_bytes(arm.encode())
                checkpoints[arm] = checkpoint
                training_manifests[arm] = {"source_core_sha256": "c" * 64}
                training_provenance[arm] = {"checkpoint_sha256": queue.run.sha(checkpoint),
                                            "source_core_sha256": "c" * 64,
                                            "ground_truth_used_for_prediction": False}
            source_provenance = {"checkpoint_sha256": queue.run.source97.EXPECTED_SOURCE_SHAS[1]}
            report_implementation = {**queue.run.implementation_fingerprint(),
                                     (queue.HERE / "train.py").relative_to(queue.ROOT).as_posix():
                                     queue.run.sha(queue.HERE / "train.py"),
                                     (queue.HERE / "evaluate.py").relative_to(queue.ROOT).as_posix():
                                     queue.run.sha(queue.HERE / "evaluate.py")}
            report = {"status": "complete", "experiment": "SW0134_native_spike_binding",
                      "seed": 1, "count": 320, "image_ids": [1320, 1639],
                      "ground_truth_used_for_prediction": False,
                      "ground_truth_used_for_scoring": True,
                      "prediction_path": str(pred), "prediction_sha256": prediction_sha,
                      "prediction_manifest_sha256": prediction_manifest_sha,
                      "source97_evaluation_sha256": queue.run.sha(source_path),
                      "arm_provenance": {"source97_native": source_provenance,
                                          **training_provenance},
                      "implementation_fingerprint": report_implementation, "scores": scores,
                      "source97_qcc_per_image_reproduction": {"passed": True}}
            report_path = out / "evaluation.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            (out / "evaluation_manifest.json").write_text(json.dumps({
                "experiment": "SW0134_native_spike_binding", "seed": 1,
                "evaluation_sha256": queue.run.sha(report_path),
                "prediction_sha256": prediction_sha,
                "source97_evaluation_sha256": queue.run.sha(source_path),
                "implementation_fingerprint": report_implementation,
                "ground_truth_used_for_prediction": False}), encoding="utf-8")
            with mock.patch.object(queue, "artifact_path", return_value=out):
                with mock.patch.object(queue.evaluate, "_source_reference",
                                       return_value=(source_path, {})), \
                     mock.patch.object(queue.train, "validate_completed_training",
                                       side_effect=lambda seed, arm, folder:
                                       (training_manifests[arm], {}, checkpoints[arm])):
                    self.assertTrue(queue.valid_result(task))
                scores["actual_joint"]["primary"]["patch_fg_ari"]["per_image"][0] = float("nan")
                (out / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
                (out / "evaluation_manifest.json").write_text(json.dumps({
                    "experiment": "SW0134_native_spike_binding", "seed": 1,
                    "evaluation_sha256": queue.run.sha(report_path),
                    "prediction_sha256": prediction_sha,
                    "source97_evaluation_sha256": queue.run.sha(source_path),
                    "implementation_fingerprint": report_implementation,
                    "ground_truth_used_for_prediction": False}), encoding="utf-8")
                with mock.patch.object(queue.evaluate, "_source_reference",
                                       return_value=(source_path, {})), \
                     mock.patch.object(queue.train, "validate_completed_training",
                                       side_effect=lambda seed, arm, folder:
                                       (training_manifests[arm], {}, checkpoints[arm])):
                    self.assertFalse(queue.valid_result(task))
        finally:
            out = root / "eval"
            if out.exists():
                for item in out.iterdir():
                    item.unlink()
                out.rmdir()
            for item in root.iterdir():
                item.unlink()
            root.rmdir()

    def test_registered_gate_is_seed1_only_and_uses_paired_primary_arrays(self):
        report = {"status": "complete", "seed": 1, "scores": {}}
        values = {"source97_native": 0.2, "actual_joint": 0.6,
                  "gate_joint": 0.3, "actual_frozen": 0.4}
        iou_values = {"source97_native": 0.5, "actual_joint": 0.1,
                      "gate_joint": 0.3, "actual_frozen": 0.4}
        for arm, base in values.items():
            report["scores"][arm] = {"primary": {}, "qcc": {}}
            for metric in queue.REPORT_METRICS:
                score_value = base if metric == "patch_fg_ari" else iou_values[arm]
                report["scores"][arm]["primary"][metric] = {
                    "per_image": [score_value] * 320, "mean": score_value, "valid_count": 320}
                report["scores"][arm]["qcc"][metric] = {
                    "per_image": [score_value] * 320, "mean": score_value, "valid_count": 320}
        gate = queue.registered_pilot_gate(report, bootstrap_samples=100)
        self.assertEqual(gate["status"], "seed1_pilot_gate_passed")
        self.assertTrue(gate["bootstrap"]["common_indices_across_comparisons"])
        self.assertLess(gate["comparisons_actual_joint_minus_reference"][
            "source97_native"]["patch_foreground_iou"]["ci95"][0], 0.0)
        report["seed"] = 2
        with self.assertRaises(ValueError):
            queue.registered_pilot_gate(report, bootstrap_samples=10)

    def test_diagnostic_real_rollout_and_renderer_smoke_all_arms(self):
        torch.manual_seed(13491)
        core = _source_core()
        wrapped = PhaseStateIntegration(core, "phase")
        encoder = _DiagnosticEncoder()
        binder = NativeSpikeSlotBinder(seed=134)
        decoder = RelativeSlotRGBDecoder(seed=106)
        checkpoint = Path(tempfile.gettempdir()) / f"sw134_diag_{uuid.uuid4().hex[:8]}.pt"
        checkpoint.write_bytes(b"fixture")
        image = torch.rand(1, 3, 128, 128) * 255.0
        gamma = torch.randn(1, 8, 256) * 2.0
        manifest = {"source_core_sha256": "a" * 64}
        try:
            loader_tuple = (wrapped, encoder, None, None, None, 3.0, binder, decoder,
                            manifest, {}, checkpoint)
            with mock.patch.object(train_diagnostic, "BATCH", 1), \
                 mock.patch.object(train_diagnostic.evaluate, "_load_training",
                                   return_value=loader_tuple), \
                 mock.patch.object(run.sw130, "read_batch", return_value=image), \
                 mock.patch.object(run.sw130, "encode", return_value=gamma):
                for arm in run.ARMS:
                    row = train_diagnostic._arm_diagnostics(
                        1, arm, torch.device("cpu"), np.array([0]), np.zeros((1,)),
                        {"train_rgb": "b" * 64})
                    self.assertEqual(row["optimizer_updates"], 0)
                    self.assertFalse(row["ground_truth_used"])
                    self.assertEqual(len(row["per_image_rgb_mse"]), 1)
                    self.assertEqual(len(row["mean_slot_probability_occupancy"]), 11)
                    self.assertTrue(np.isfinite(row["mean_rgb_mse"]))
                    self.assertTrue(np.isfinite(row["mean_slot_column_scramble_rgb_mse"]))
        finally:
            checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
