from __future__ import annotations

import sys
import hashlib
import json
import uuid
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collaborative_test.SW_0139_native32_representation import run
from collaborative_test.SW_0139_native32_representation import evaluate
from collaborative_test.SW_0139_native32_representation.model import ContextResidualEncoder


class SW0139RepresentationTests(unittest.TestCase):
    def test_eval_rgb_cache_seam_produces_real_encoder_nchw_gamma(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Conv2d(3, 8, kernel_size=1)

            def forward(self, x):
                return self.proj(torch.nn.functional.avg_pool2d(x, 4))

        encoder = TinyEncoder()
        foundation = type("Foundation", (), {
            "encoder": encoder, "feature_mean": torch.zeros(1, 8, 1, 1),
            "feature_std": torch.ones(1, 8, 1, 1), "feature_clip": 3.0,
            "patcher": evaluate.sw135.FeaturePatchGammaInitializer(grid_size=32),
        })()
        cache = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        rgb = evaluate.read_eval_rgb(cache, 0, "cpu")
        self.assertEqual(tuple(rgb.shape), (1, 3, 128, 128))
        self.assertEqual(rgb.dtype, torch.float32)
        gamma = run._features_to_gamma(foundation, rgb)
        self.assertEqual(tuple(gamma.shape), (1, 8, 1024))
        self.assertTrue(torch.isfinite(gamma).all())

    def test_partial_endpoint_scores_surviving_arm_without_claiming_expansion(self):
        fixture = ROOT / ("sw139_partial_score_" + uuid.uuid4().hex[:10])
        pred_root = fixture / "pred"
        (pred_root / "control").mkdir(parents=True)
        output = fixture / "evaluation.json"
        arm_row = {"labels": np.zeros((320, 32, 32), dtype=np.int64),
                   "prediction_path": fixture / "arm.npz", "protocol_path": fixture / "arm.json",
                   "prediction_sha256": "arm-pred", "protocol_sha256": "arm-protocol"}
        source_row = {"ids": np.asarray(evaluate.contract.IMAGE_IDS, dtype=np.int64),
                      "labels": np.zeros((320, 32, 32), dtype=np.int64),
                      "archive_path": fixture / "source.npz", "protocol_path": fixture / "source.json",
                      "prediction_sha256": "source-pred", "protocol_sha256": "source-protocol"}

        class FakeMasks:
            shape = (1640, 128, 128, 1)
            def __getitem__(self, key):
                return np.zeros((320, 128, 128, 1), dtype=np.uint8)

        class FakeH5:
            def __enter__(self):
                return {"mask": FakeMasks()}
            def __exit__(self, *_args):
                return False

        def sha(path):
            name = Path(path).name
            return {"arm.npz": "arm-pred", "arm.json": "arm-protocol",
                    "source.npz": "source-pred", "source.json": "source-protocol"}.get(name, "other")

        metric = {name: {"mean": .5, "per_image": np.full(320, .5).tolist()}
                  for name in evaluate.METRICS}
        fixture.mkdir(exist_ok=True)
        try:
            with mock.patch.object(evaluate, "validate_arm_prediction", return_value=arm_row), \
                 mock.patch.object(evaluate.sw137, "validate_prediction", return_value=source_row), \
                 mock.patch.object(evaluate.run, "sha256_file", side_effect=sha), \
                 mock.patch.object(evaluate.h5py, "File", return_value=FakeH5()), \
                 mock.patch.object(evaluate.contract, "modal_native32", return_value=np.zeros((320, 32, 32))), \
                 mock.patch.object(evaluate.contract, "production_metrics32", return_value=metric):
                result = evaluate.score_all(prediction_root=pred_root, output=output,
                                            dataset=fixture / "unused.h5", bootstrap_samples=10)
            self.assertEqual(result["status"], "partial_complete")
            self.assertEqual(result["valid_arms"], ["control"])
            self.assertIn("source97", result["scores"])
            self.assertFalse(result["candidate_expansion_gates"]["cross_view"]["may_justify_seeds0_2"])
            self.assertIn("context_residual", result["rejected_arms"])
        finally:
            if output.exists(): output.unlink()
            (pred_root / "control").rmdir()
            pred_root.rmdir()
            fixture.rmdir()

    def test_context_zero_output_and_two_step_hidden_credit(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(139)
            model = ContextResidualEncoder(139)
            x = torch.randn(1, 8, 126, 126) * 0.25
        y = model(x)
        self.assertTrue(torch.equal(y, x.clamp(-3, 3)))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                            for branch in model.branches for p in branch.parameters()))
        opt.step()
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        hidden = [p.grad for branch in model.branches for p in branch.parameters()]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) > 0
                            for g in hidden))

    def test_input_geometry_fails_closed(self):
        model = ContextResidualEncoder()
        with self.assertRaisesRegex(ValueError, "standardized"):
            model(torch.zeros(1, 8, 128, 128))
        with self.assertRaisesRegex(ValueError, "standardized"):
            model(torch.zeros(1, 7, 126, 126))

    def test_negative_sampler_never_uses_same_image(self):
        image, patch = run._sample_other_image_negatives(4, 1024, 32, 1001, "cpu")
        self.assertEqual(tuple(image.shape), (4, 1024, 32))
        self.assertEqual(tuple(patch.shape), (4, 1024, 32))
        for row in range(4):
            self.assertTrue(torch.all(image[row] != row))
        image2, patch2 = run._sample_other_image_negatives(4, 1024, 32, 1001, "cpu")
        self.assertTrue(torch.equal(image, image2))
        self.assertTrue(torch.equal(patch, patch2))

    def test_info_nce_is_finite_and_differentiable_on_raw_spikes(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            a = torch.randn(2, 4, 1024, 512, requires_grad=True)
            b = torch.randn(2, 4, 1024, 512, requires_grad=True)
        loss = run.symmetric_temporal_infonce(a, b, seed=912, negatives=3, patch_chunk=64)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())

    def test_symmetric_loss_rejects_single_image_and_wrong_trace(self):
        with self.assertRaisesRegex(ValueError, "different images"):
            run.symmetric_temporal_infonce(torch.zeros(1, 4, 1024, 512),
                                           torch.zeros(1, 4, 1024, 512), seed=1)
        with self.assertRaisesRegex(ValueError, "actual spikes"):
            run.symmetric_temporal_infonce(torch.zeros(2, 4, 1024, 64),
                                           torch.zeros(2, 4, 1024, 64), seed=1)

    def test_view_transform_is_deterministic_and_bounded(self):
        rgb = torch.linspace(0, 1, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
        x1, c1, b1 = run._paired_view(rgb, seed=123)
        x2, c2, b2 = run._paired_view(rgb, seed=123)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(b1, b2))
        self.assertTrue(bool(((c1 >= .9) & (c1 <= 1.1)).all()))
        self.assertTrue(bool(((b1 >= -.03) & (b1 <= .03)).all()))
        self.assertGreaterEqual(float(x1.min()), 0.0)
        self.assertLessEqual(float(x1.max()), 1.0)

    def test_registered_seed_and_update_contract(self):
        with self.assertRaisesRegex(ValueError, "seed1"):
            run._validate_seed_ids(type("F", (), {"image_ids": list(range(4096))})(), 0)
        self.assertEqual(run.UPDATES, 256)
        self.assertEqual(run.LOGICAL_BATCH, 16)
        self.assertEqual(run.MICROBATCH, 4)
        self.assertEqual(run.STEPS, 1024)
        self.assertEqual(run.TAIL, 64)

    def test_seed_validation_binds_actual_registered_rows_not_guessed_first_id(self):
        ids = list(range(4095)) + [68304]
        rows = np.arange(4096, dtype=np.int64) + 20
        foundation = type("Foundation", (), {"image_ids": ids,
                                               "pool_indices": rows})()
        source = (None, None, None, rows.copy(), ids.copy(), "source-sha")
        with mock.patch.object(run.sw130, "source_contract", return_value=source):
            run._validate_seed_ids(foundation, 1)
            foundation.pool_indices = rows[::-1].copy()
            with self.assertRaisesRegex(AssertionError, "IDs/rows"):
                run._validate_seed_ids(foundation, 1)

    def test_lambda_record_recomputes_median_and_quarter_scale(self):
        ratios = [2.0, 4.0, 6.0, 8.0]
        rows = [{"logical_batch": i, "old_encoder_graph_grad_norm": r * 3.0,
                 "contrast_encoder_graph_grad_norm": 3.0, "ratio": r,
                 "gradient_norms_by_family": {
                     "old": {"encoder": 1.0, "graph": 1.0},
                     "contrastive": {"encoder": 1.0, "graph": 1.0}}}
                for i, r in enumerate(ratios)]
        record = {"experiment": "SW0139_native32_representation", "status": "lambda_calibrated",
                  "seed": 0, "source_core_sha256": "core", "source_manifest_sha256": "manifest",
                  "source_training_ids_sha256": "ids", "first_four_logical_batches": rows,
                  "median_ratio": 5.0, "lambda": 1.25, "ground_truth_used": False,
                  "optimizer_updates": 0, "implementation_fingerprint": run.implementation_fingerprint()}
        source = type("Foundation", (), {"provenance": {"source_core_sha256": "core",
                  "source_manifest_sha256": "manifest", "source_training_ids_sha256": "ids"}})()
        fixture = ROOT / ("sw139_fixture_" + uuid.uuid4().hex[:10])
        fixture.mkdir()
        path = fixture / "lambda.json"
        try:
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            record["self_sha256"] = hashlib.sha256(canonical).hexdigest()
            path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(run.sw135, "load_native32_foundation", return_value=source):
                loaded = run.load_lambda_record(path)
            self.assertEqual(loaded["lambda"], 1.25)
            record["lambda"] = 1.5
            path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(run.sw135, "load_native32_foundation", return_value=source):
                with self.assertRaisesRegex(ValueError, "lambda arithmetic"):
                    run.load_lambda_record(path)
        finally:
            if path.exists():
                path.unlink()
            fixture.rmdir()

    def test_disposable_optimizer_runs_control_and_context_two_step_paths(self):
        class Graph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.7]]))

        class Native(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_generator = Graph()
                self.frozen = torch.nn.Parameter(torch.tensor([0.2]))

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = Native()

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.5]]))

        ids = list(range(4096))
        foundation = type("Foundation", (), {
            "wrapped": Wrapped(), "encoder": Encoder(),
            "image_ids": ids, "pool_indices": np.arange(4096, dtype=np.int64),
            "provenance": {"source_core_sha256": "core", "source_manifest_sha256": "manifest",
                           "source_training_ids_sha256": "idsha", "rgb_asset_validation": {}}
        })()
        feature = torch.randn(1, 8, 126, 126) * .2

        def fake_losses(_foundation, arm, context, _rgb, _ids, *, nonce):
            graph = _foundation.wrapped.core.graph_generator.weight
            encoder = _foundation.encoder.weight
            old = graph.square().sum() + encoder.square().sum()
            if context is not None:
                old = old + context(feature).square().mean()
            contrast = old * 0.5 if arm == "cross_view" else None
            return old, contrast, None, None, None

        base_patches = (
            mock.patch.object(run, "source_bundle", return_value=foundation),
            mock.patch.object(run, "_validate_seed_ids"),
            mock.patch.object(run, "_logical_losses", side_effect=fake_losses),
            mock.patch.object(run, "read_rgb_rows", return_value=torch.zeros(1, 3, 128, 128)),
            mock.patch.object(run.np, "load", return_value=object()),
            mock.patch.object(run, "MICROS", 1),
        )
        for patch in base_patches:
            patch.start()
        try:
            control = run.disposable_update(1, "control")
            self.assertEqual(control["optimizer_updates"], 1)
            self.assertGreater(control["steps"][0]["changed_parameters"], 0)
            self.assertEqual(set(control["steps"][0]["gradient_norm_by_family"]), {"graph", "encoder"})
            self.assertGreater(control["steps"][0]["gradient_norm_by_family"]["graph"], 0.0)
            self.assertGreater(control["steps"][0]["gradient_norm_by_family"]["encoder"], 0.0)
            self.assertGreaterEqual(control["resource_measurement"]["elapsed_seconds"], 0.0)
            self.assertIsNone(control["resource_measurement"]["max_memory_allocated_bytes"])
            context_foundation = type("Foundation", (), {
                "wrapped": Wrapped(), "encoder": Encoder(),
                "image_ids": ids, "pool_indices": np.arange(4096, dtype=np.int64),
                "provenance": {"source_core_sha256": "core", "source_manifest_sha256": "manifest",
                               "source_training_ids_sha256": "idsha", "rgb_asset_validation": {}}
            })()
            with mock.patch.object(run, "source_bundle", return_value=context_foundation), \
                 mock.patch.object(run, "_feature_map_from_rgb", return_value=feature):
                ctx = run.disposable_update(1, "context_residual")
            self.assertEqual(ctx["optimizer_updates"], 2)
            self.assertTrue(ctx["context_exact_initial_feature_parity"])
            self.assertTrue(any(key.startswith("core.graph_generator") for key in ctx["changed_source_state_keys"]))
        finally:
            for patch in reversed(base_patches):
                patch.stop()

    def test_train_stage_writes_bound_small_fixture_completion(self):
        class Graph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.7]]))

        class Native(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_generator = Graph()

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = Native()

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.5]]))

        ids = list(range(16))
        foundation = type("Foundation", (), {
            "wrapped": Wrapped(), "encoder": Encoder(), "pool_indices": np.arange(16),
            "image_ids": ids,
            "provenance": {"source_core_sha256": "core", "source_manifest_sha256": "manifest",
                           "source_training_ids_sha256": "source_ids", "rgb_asset_validation": {}}
        })()
        fixture = ROOT / ("sw139_train_fixture_" + uuid.uuid4().hex[:10])
        fixture.mkdir()
        preflight_path = fixture / "preflight.json"
        output_root = fixture / "trained"
        preflight = {"status": "disposable_update_complete", "seed": 1, "arm": "control",
                     "implementation_fingerprint": run.implementation_fingerprint(),
                     "ground_truth_used": False, "optimizer_updates": 1,
                     "source_unchanged_except_trainable_graph": True,
                     "steps": [{"changed_parameters": 1, "gradient_norm": 1.0}]}
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

        def train_loss(_foundation, arm, context, rgb, image_ids, *, lambda_value, nonce):
            loss = (_foundation.wrapped.core.graph_generator.weight.square().sum()
                    + _foundation.encoder.weight.square().sum())
            return loss, {"old": loss, "contrast": None, "view_c": None, "view_offset": None}

        patches = [
            mock.patch.object(run, "TRAIN_IMAGES", 16),
            mock.patch.object(run, "UPDATES", 1),
            mock.patch.object(run, "MICROBATCH", 4),
            mock.patch.object(run, "MICROS", 4),
            mock.patch.object(run, "LOGICAL_BATCH", 16),
            mock.patch.object(run, "source_bundle", return_value=foundation),
            mock.patch.object(run, "_validate_seed_ids"),
            mock.patch.object(run, "_train_micro", side_effect=train_loss),
            mock.patch.object(run, "read_rgb_rows", return_value=torch.zeros(4, 3, 128, 128)),
            mock.patch.object(run.np, "load", return_value=object()),
        ]
        for patch in patches:
            patch.start()
        try:
            manifest = run.train(1, "control", "cpu", preflight_path=preflight_path,
                                 output_root=output_root)
            folder = output_root / "control_seed1"
            history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
            marker = json.loads((folder / "TRAINING_COMPLETED.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["updates"], 1)
            self.assertEqual(manifest["total_image_exposures"], 16)
            self.assertEqual(history["records"][0]["training_ids"], ids)
            self.assertEqual(marker["status"], "training_complete")
            self.assertTrue((folder / "checkpoint.pt").is_file())
            self.assertTrue((folder / "optimizer.pt").is_file())
        finally:
            for patch in reversed(patches):
                patch.stop()
            for file in output_root.glob("control_seed1/*") if output_root.exists() else ():
                file.unlink()
            if (output_root / "control_seed1").exists():
                (output_root / "control_seed1").rmdir()
            if output_root.exists():
                output_root.rmdir()
            if preflight_path.exists():
                preflight_path.unlink()
            fixture.rmdir()


if __name__ == "__main__":
    unittest.main()
