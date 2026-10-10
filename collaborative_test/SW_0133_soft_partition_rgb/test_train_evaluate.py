import json
import hashlib
import math
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0133_soft_partition_rgb import coordinator, evaluate, pilot_queue, run, train
from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import RelativeRGBDecoder


def _metrics(offset=0.0):
    values = np.linspace(0.1, 0.9, 320) + offset
    return {name: values.tolist() for name in evaluate.METRICS}


class SW0133TrainEvaluateTests(unittest.TestCase):
    def test_real_two_update_train_loop_saves_live_and_detached_arm_artifacts(self):
        """Run production forward, gradients, Adam, history and saves on a tiny batch."""
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        root = Path(tempfile.gettempdir()) / ("sw133train_" + uuid.uuid4().hex[:10])
        root.mkdir()
        archive = root / "archive"
        archive.mkdir()
        assets = {"synthetic": "b" * 64}
        source_sha = "a" * 64
        pool = np.arange(32, dtype=np.int64)
        ids = list(range(5000, 5032))
        ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        pool_sha = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
        source_manifest_path = root / "source_manifest.json"
        source_manifest = {"source_model_seed": 1, "steps": 256}
        source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
        decoder_warm = RelativeRGBDecoder()
        warm_optimizer = torch.optim.Adam(decoder_warm.parameters(), lr=run.DECODER_LR)
        for _ in range(32):
            warm_optimizer.zero_grad(set_to_none=True)
            for parameter in decoder_warm.parameters():
                parameter.grad = torch.full_like(parameter, 1e-3)
            warm_optimizer.step()
        warm_path = archive / "preflight_decoder_seed1.pt"
        with warm_path.open("xb") as stream:
            torch.save({"decoder_state_dict": decoder_warm.state_dict(),
                        "optimizer_state_dict": warm_optimizer.state_dict(),
                        "source_core_sha256": source_sha,
                        "training_ids_sha256": ids_sha, "asset_hashes": assets,
                        "warmup_updates": 32}, stream)
        lambda_value = 0.7
        rows = {}
        for seed in run.SEEDS:
            row = {"seed": seed, "source_core_sha256": source_sha,
                   "source_manifest_sha256": run.sha(source_manifest_path),
                   "training_ids": ids, "training_ids_sha256": ids_sha,
                   "pool_indices_sha256": pool_sha, "asset_hashes": assets,
                   "implementation_fingerprint": run.implementation_fingerprint(),
                   "lambda": lambda_value, "lambda_source_seed": 0,
                   "decoder_warmup_artifact_sha256": run.sha(warm_path)}
            if seed:
                row["seed0_lambda_record_sha256"] = "seed0-proof"
            rows[seed] = {"path": str(archive / f"preflight_seed{seed}.json"),
                          "sha256": "seed0-proof" if seed == 0 else f"pf{seed}",
                          "row": row}

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, kernel_size=1)

            def forward(self, images):
                return self.conv(images)

        image_batch = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8).float()
        fake_cache = np.zeros((32, 128, 128, 3), dtype=np.uint8)

        def make_model(seed, device, arm):
            torch.manual_seed(600 + seed)
            core = run.sw130.source97.make_core(device, steps=64)
            core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
            wrapped = run.sw130.PhaseStateIntegration(
                core, "constant" if arm == "constant_live" else "phase")
            encoder = TinyEncoder().to(device)
            patcher = run.sw130.FeaturePatchGammaInitializer(grid_size=16)
            decoder = RelativeRGBDecoder().to(device)
            return (wrapped, encoder, patcher, torch.zeros(1, 8, 1, 1),
                    torch.ones(1, 8, 1, 1), 3.0, decoder, pool.copy(), ids, source_sha)

        outputs = []
        try:
            for arm in ("phase_live", "phase_detached"):
                output = root / f"train_{arm}"
                with (mock.patch.object(run, "ARCHIVE", archive),
                      mock.patch.object(run, "source_contract", return_value=(
                          root / "source.pt", source_manifest_path, source_manifest,
                          pool, ids, source_sha)),
                      mock.patch.object(run.sw130, "validate_rgb_assets", return_value=assets),
                      mock.patch.object(train, "_passed_preflights", return_value=rows),
                      mock.patch.object(run, "load_models", side_effect=make_model),
                      mock.patch.object(run.sw130, "UPDATES", 2),
                      mock.patch.object(run.sw130, "read_batch", return_value=image_batch),
                      mock.patch.object(run.np, "load", return_value=fake_cache)):
                    manifest = train.train(1, arm, torch.device("cpu"), output)
                outputs.append((arm, output, manifest))
                self.assertEqual(manifest["status"], "training_complete")
                self.assertEqual(manifest["updates"], 2)
                self.assertEqual(manifest["artifact_sha256"].keys(), {
                    "core.pt", "integration.pt", "encoder.pt", "decoder.pt",
                    "optimizers.pt", "history.json"})
                self.assertTrue((output / "TRAINING_COMPLETED").is_file())
                history = json.loads((output / "history.json").read_text(encoding="utf-8"))
                self.assertEqual([row["update"] for row in history], [1, 2])
                self.assertTrue(all(row["assignment_live"] == (arm == "phase_live")
                                    for row in history))
                self.assertTrue(all(math.isfinite(row["total"]) for row in history))
                optimizers = torch.load(output / "optimizers.pt", map_location="cpu",
                                        weights_only=True)
                decoder_steps = [int(value["step"].item()) for value in
                                 optimizers["decoder"]["state"].values()]
                self.assertEqual(len(decoder_steps), 6)
                self.assertEqual(set(decoder_steps), {34})
        finally:
            for path in root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir():
                    path.rmdir()
            root.rmdir()
            torch.set_num_threads(old_threads)

    def test_pilot_queue_is_seed1_only_train_then_matching_eval_without_retries(self):
        tasks = pilot_queue.task_plan()
        self.assertEqual(len(tasks), 6)
        self.assertEqual([(row["stage"], row["arm"]) for row in tasks], [
            (stage, arm) for arm in run.ARMS for stage in ("train", "evaluate")])
        self.assertTrue(all(row["seed"] == 1 for row in tasks))
        self.assertEqual(len({row["task_id"] for row in tasks}), 6)
        for task in tasks:
            command = pilot_queue.command(task)
            self.assertIn("--seed", command)
            self.assertIn("1", command)
            self.assertIn(task["arm"], command)

    def test_train_rejects_missing_source_preflight_before_creating_output(self):
        root = Path(__file__).resolve().parent
        output = root / ("test_output_" + uuid.uuid4().hex)
        try:
            with mock.patch.object(run.sw130, "validate_rgb_assets", return_value={"x": "y"}), \
                 mock.patch.object(train.preflight_queue, "valid_result", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "preflight is absent or invalid"):
                    train.train(1, "phase_live", "cpu", output)
            self.assertFalse(output.exists())
        finally:
            if output.exists():
                output.rmdir()

    def test_pilot_gate_reports_all_three_paired_comparisons_and_closes_negative(self):
        source = {name: {"mean": float(np.mean(values)), "valid_count": 320,
                         "per_image": values}
                  for name, values in _metrics().items()}
        arms = {
            "phase_live": {"scores": {"metrics": {m: 0.8 for m in evaluate.METRICS},
                                        "per_image": _metrics(0.2)}},
            "constant_live": {"scores": {"metrics": {m: 0.5 for m in evaluate.METRICS},
                                           "per_image": _metrics()}},
            "phase_detached": {"scores": {"metrics": {m: 0.5 for m in evaluate.METRICS},
                                            "per_image": _metrics()}},
        }
        passed = coordinator.pilot_gate(source, arms)
        self.assertEqual(passed["status"], "pilot_gate_passed")
        self.assertEqual(set(passed["comparisons"]), {"source97", "constant_live", "phase_detached"})
        arms["phase_live"]["scores"]["per_image"] = _metrics(-0.2)
        failed = coordinator.pilot_gate(source, arms)
        self.assertEqual(failed["status"], "pilot_gate_failed")
        with self.assertRaisesRegex(ValueError, "all three"):
            coordinator.pilot_gate(source, {"phase_live": arms["phase_live"]})
        source["fg_ari"]["mean"] += 0.01
        with self.assertRaisesRegex(ValueError, "mean does not match"):
            coordinator.pilot_gate(source, arms)

    def test_evaluation_validator_binds_sidecar_manifest_and_exact_fixed320_means(self):
        base = Path(__file__).resolve().parent / ("fixture_" + uuid.uuid4().hex)
        base.mkdir()
        try:
            training = base / "train"
            training.mkdir()
            manifest = {"seed": 1, "arm": "phase_live", "artifact_sha256": {},
                        "source_core_sha256": "a" * 64}
            manifest_path = training / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            pred_path, gamma_path = base / "pred.pt", base / "gamma.pt"
            pred_path.write_bytes(b"predictions")
            gamma_path.write_bytes(b"gamma")
            report_path = base / "evaluation.json"
            per_image = _metrics()
            means = {name: float(np.mean(per_image[name])) for name in evaluate.METRICS}
            counts = {name: 320 for name in evaluate.METRICS}
            report = {
                "status": "complete", "experiment": "SW0133_soft_partition_rgb",
                "seed": 1, "arm": "phase_live", "images": 320, "ids": [1320, 1639],
                "time_steps": 1024, "settle": 512, "batch_size": 8,
                "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
                "scores": {"metrics": means, "valid_count": counts, "per_image": per_image},
                "training_manifest_path": str(manifest_path),
                "training_manifest_sha256": run.sha(manifest_path),
                "training_artifact_sha256": {}, "source_core_sha256": "a" * 64,
                "frozen_predictions_sha256": "b" * 64, "gamma_sha256": "c" * 64,
                "frozen_predictions_path": str(pred_path), "gamma_path": str(gamma_path),
                "readout": {"affinity_mode": "spike", "threshold": 0.50,
                            "minimum_group_size": 2, "background": "largest_component"},
            }
            report["frozen_predictions_sha256"] = run.sha(pred_path)
            report["gamma_sha256"] = run.sha(gamma_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            sidecar = {"status": "complete", "experiment": "SW0133_soft_partition_rgb",
                       "seed": 1, "arm": "phase_live", "evaluation_sha256": run.sha(report_path),
                       "evaluation_fingerprint": evaluate.evaluation_fingerprint(),
                       "frozen_predictions_sha256": report["frozen_predictions_sha256"],
                       "gamma_sha256": report["gamma_sha256"]}
            (base / "evaluation_manifest.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (base / "COMPLETED").write_text("complete\n", encoding="utf-8")
            self.assertTrue(coordinator.evaluation_valid(1, "phase_live", report_path, training))
            report["scores"]["per_image"]["fg_ari"][0] = float("nan")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            sidecar["evaluation_sha256"] = run.sha(report_path)
            (base / "evaluation_manifest.json").write_text(json.dumps(sidecar), encoding="utf-8")
            self.assertFalse(coordinator.evaluation_valid(1, "phase_live", report_path, training))
        finally:
            for child in base.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(base.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if child.is_dir():
                    child.rmdir()
            base.rmdir()


if __name__ == "__main__":
    unittest.main()
