"""Focused provenance and CLI regression checks for the SW0125 runner."""
import contextlib
import io
import json
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0125_late_rollout_credit import coordinator, evaluate, run


class RunTests(unittest.TestCase):
    def test_foundation_proof_must_bind_exact_contract_and_credit(self):
        archive = run.HERE / f".foundation-proof-test-{uuid.uuid4().hex}"
        archive.mkdir()
        proof_path = archive / "foundation_layoutfix_seed0_source_20261009.json"
        try:
            proof = {
                "status": "complete", "experiment": "SW0125", "seed": 0,
                "optimizer_updates": 0, "ground_truth_used": False,
                "source_core_sha256": "h", "source_manifest_sha256": "h",
                "encoder_sha256": "h", "preprocessing_sha256": "h",
                "gamma_train_sha256": "h", "gamma_train_manifest_sha256": "h",
                "rgb_train_cache_manifest_sha256": "h", "rgb_train_cache_sha256": "h",
                "total_time_steps": 1024, "loss_settle_steps": 512,
                "gradient_tail_steps": 64, "detached_prefix_steps": 960,
                "global_image_ids": list(range(16)), "gamma_cache_max_abs_diff": 0.0,
                "trace_parity_exact": {key: True for key in (
                    "theta", "spikes", "membrane", "component_spikes", "component_membrane")},
                "loss_parity_exact": {key: True for key in (
                    "primary_phase", "positive_actual_q", "analytic_actual_h_rgb")},
                "production_hard_labels_and_H_exact": True,
                "late_tail_rgb_gradient_norm_by_family": {key: 1.0 for key in (
                    "encoder", "graph", "oscillator_drive", "kuramoto", "dendritic", "membrane")},
                "implementation_sha256": {key: "h" for key in (
                    "foundation", "foundation_verifier", "source_core", "synchrony",
                    "kuramoto", "dendrite", "membrane", "gating", "phase_loss", "rgb_loss")},
            }
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            with mock.patch.object(run, "ARCHIVE", archive), mock.patch.object(run, "sha", return_value="h"):
                accepted = run.validate_foundation_report(
                    0, Path("source.pt"), Path("source.json"), np.arange(4096), "h", "h", "h")
                self.assertEqual(accepted["sha256"], "h")
                proof["trace_parity_exact"]["membrane"] = False
                proof_path.write_text(json.dumps(proof), encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, "proof is stale or invalid"):
                    run.validate_foundation_report(
                        0, Path("source.pt"), Path("source.json"), np.arange(4096), "h", "h", "h")
        finally:
            if proof_path.exists():
                proof_path.unlink()
            archive.rmdir()

    def test_evaluate_cli_prints_valid_terminal_json(self):
        output = Path("evaluation.json")
        expected = {"status": "complete", "experiment": "SW0125", "seed": 1,
                    "output": str(output)}
        with mock.patch.object(evaluate.run, "evaluate", return_value=expected), \
                mock.patch("sys.argv", ["evaluate.py", "--seed", "1", "--checkpoint", "core.pt",
                                        "--output", str(output)]), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            evaluate.main()
        self.assertEqual(json.loads(stdout.getvalue()), expected)

    def test_evaluate_produces_coordinator_valid_artifacts(self):
        """Exercise gamma-manifest validation, shared report wrapping, and adoption."""
        archive = run.HERE / f".e{uuid.uuid4().hex[:8]}"
        archive.mkdir()
        seed_dir = archive / "seed0_late_candidate"
        eval_dir = seed_dir / "evaluation"
        seed_dir.mkdir()
        checkpoint = seed_dir / "core.pt"
        encoder_path = seed_dir / "encoder.pt"
        optimizer_path = seed_dir / "optimizer.pt"
        history_path = seed_dir / "history.json"
        completion = seed_dir / "TRAINING_COMPLETED"
        manifest_path = seed_dir / "manifest.json"
        output = eval_dir / "evaluation.json"

        encoder = torch.nn.Linear(1, 1)
        torch.save({"stub": torch.tensor([1.0])}, checkpoint)
        torch.save(encoder.state_dict(), encoder_path)
        torch.save({"stub": torch.tensor([2.0])}, optimizer_path)
        history_path.write_text("[]", encoding="utf-8")
        completion.write_text("complete\n", encoding="utf-8")
        manifest = {
            "status": "training_complete", "experiment": "SW0125", "seed": 0,
            "arm": run.ARM, "core_sha256": run.sha(checkpoint),
            "encoder_sha256": run.sha(encoder_path),
            "history_sha256": run.sha(history_path),
            "optimizer_sha256": run.sha(optimizer_path),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        fixed_asset_hashes = {
            str(run.sw117.STATS_PATH): "registered-stats-sha",
            str(run.sw117.VAL_RGB_MANIFEST): "registered-rgb-manifest-sha",
        }
        original_sha = run.sha

        def test_sha(path):
            key = str(Path(path))
            return fixed_asset_hashes[key] if key in fixed_asset_hashes else original_sha(path)

        def fake_shared_eval(seed, arm, core_path, report_path, device="cuda"):
            value, metadata = run.base.validate_gamma_cache(
                run.base.GAMMA_VAL, run.base.GAMMA_VAL_MANIFEST, validation=True)
            self.assertEqual(tuple(value.shape), (320, 8, 256))
            self.assertEqual(metadata["image_ids"], [1320, 1639])
            score = {
                "metrics": {"fg_ari": 0.2, "foreground_iou": 0.3,
                            "matched_object_iou": 0.4},
                "valid_count": {key: 320 for key in (
                    "fg_ari", "foreground_iou", "matched_object_iou")},
                "per_image": {key: [value] * 320 for key, value in (
                    ("fg_ari", 0.2), ("foreground_iou", 0.3),
                    ("matched_object_iou", 0.4))},
            }
            report_path.write_text(json.dumps({
                "sweep": [{"scored_targets": {"our_hdf5": score}}],
                "ids": [1320, 1639], "images": 320,
            }), encoding="utf-8")

        try:
            with mock.patch.object(run, "OUT", archive), \
                    mock.patch.object(run, "sha", side_effect=test_sha), \
                    mock.patch.object(run.sw117, "validate_rgb_validation_cache",
                                      return_value=(object(), None, "registered-rgb-cache-sha")), \
                    mock.patch.object(run.sw117, "read_rgb",
                                      side_effect=lambda _cache, ids, device: torch.zeros(
                                          len(ids), 3, 128, 128, device=device)), \
                    mock.patch.object(run.sw117, "encode_rgb",
                                      side_effect=lambda _encoder, _patcher, _mean, _std, _clip, images:
                                      torch.zeros(len(images), 8, 256, device=images.device)), \
                    mock.patch.object(run, "load_models", return_value=(
                        object(), encoder, torch.nn.Identity(), torch.zeros(3),
                        torch.ones(3), 1.0, Path("source.pt"))), \
                    mock.patch.object(run.base, "evaluate", side_effect=fake_shared_eval):
                result = run.evaluate(0, checkpoint, output, device="cpu")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["evaluation_contract"]["ids"], [1320, 1639])
                self.assertFalse(result["ground_truth_used_for_prediction"])
                task = next(task for task in coordinator.task_plan()
                            if task["stage"] == "evaluate" and task["seed"] == 0)
                self.assertTrue(coordinator.valid_result(task))
        finally:
            # Remove only files created by this fixture, then empty known dirs.
            for path in sorted(archive.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            archive.rmdir()


if __name__ == "__main__":
    unittest.main()
