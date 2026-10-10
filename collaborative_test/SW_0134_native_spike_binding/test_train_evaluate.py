"""Production-loop CPU smoke for the separate SW0134 train/evaluation stages."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np
import torch
from torch import nn
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding import run, train
from collaborative_test.SW_0134_native_spike_binding import evaluate


def _source_core():
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=256, sc=torch.eye(256), osc_dim=4,
        gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
        theta_init="gamma", graph_mode="learned", graph_top_k=4,
        graph_spatial_decay=.35, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
        kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
        num_time_steps=1024, spike_per_component=True, gate_mode="raw",
        membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
        spike_spatial_grid_size=16,
    ).validate()
    return S2NetCore(hp, device="cpu")


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("train_forward_count", torch.zeros((), dtype=torch.long))


class TrainLoopSmokeTests(unittest.TestCase):
    def test_frozen_prediction_manifest_records_requested_seed_and_full_arrays(self):
        output = Path(tempfile.gettempdir()) / f"sw134_eval_commit_{uuid.uuid4().hex[:10]}"
        output.mkdir()
        try:
            predictions = {
                arm: {kind: torch.zeros(320, 16, 16, dtype=torch.int64)
                      for kind in ("primary", "qcc")}
                for arm in ("source97_native", *evaluate.ARMS)
            }
            provenance = {arm: {"checkpoint_sha256": "a" * 64}
                          for arm in predictions}
            path, prediction_sha, manifest_sha = evaluate._commit_predictions(
                output, predictions, provenance, seed=2)
            payload = json.loads((output / "prediction_manifest.json").read_text(encoding="utf-8"))
            stored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["seed"], 2)
            self.assertEqual(payload["prediction_sha256"], prediction_sha)
            self.assertEqual(run.sha(output / "prediction_manifest.json"), manifest_sha)
            self.assertEqual(stored["image_ids"], list(evaluate.IDS))
            self.assertTrue(all(tuple(x.shape) == (320, 16, 16)
                                for arm in stored["predictions"].values()
                                for x in arm.values()))
        finally:
            for item in output.iterdir():
                item.unlink()
            output.rmdir()

    def test_real_two_update_train_loop_and_completed_history_for_each_arm(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1341302)
        root = Path(tempfile.gettempdir()) / f"sw134_train_smoke_{uuid.uuid4().hex[:10]}"
        root.mkdir()
        try:
            source_path, manifest_path = root / "source.pt", root / "source_manifest.json"
            core = _source_core()
            source_state = core.state_dict()
            torch.save(source_state, source_path)
            manifest_path.write_text('{"steps":256,"source_model_seed":1}', encoding="utf-8")
            source_sha = run.sha(source_path)
            manifest_sha = run.sha(manifest_path)
            pool = np.arange(2, dtype=np.int64)
            ids = [101, 102]
            assets = {"train_rgb": "a" * 64, "validation_rgb": "b" * 64}
            raw_gamma = torch.randn(1, 8, 256) * 2.0
            image = torch.rand(1, 3, 128, 128) * 255.0
            warm_entries = {}
            for arm in ("actual_joint", "gate_joint"):
                binder = run.NativeSpikeSlotBinder(seed=134)
                decoder = run.RelativeSlotRGBDecoder(seed=106)
                params = list(binder.parameters()) + list(decoder.parameters())
                optimizer = torch.optim.Adam(params, lr=run.HEAD_LR)
                for _ in range(run.WARMUP_UPDATES):
                    optimizer.zero_grad(set_to_none=True)
                    sum((i + 1) * p.square().sum() for i, p in enumerate(params)).backward()
                    optimizer.step()
                path = root / f"warm_{arm}_seed1.pt"
                torch.save({
                    "experiment": "SW0134_native_spike_binding", "seed": 1, "arm": arm,
                    "updates": run.WARMUP_UPDATES, "batch_size": 1,
                    "training_ids": ids, "training_ids_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "all_training_ids_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "source_core_sha256": source_sha,
                    "source_manifest_sha256": manifest_sha,
                    "asset_hashes": assets,
                    "implementation_fingerprint": run.implementation_fingerprint(),
                    "binder_state_dict": binder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, path)
                warm_entries[arm] = {"path": str(path.resolve()), "sha256": run.sha(path),
                                     "updates": run.WARMUP_UPDATES,
                                     "loss_first_last": [1.0, .5]}
            preflight = {"asset_hashes": assets, "warm_artifacts": warm_entries}
            preflight_path = root / "preflight_seed1.json"
            preflight_path.write_text(json.dumps({"status": "passed", "seed": 1}),
                                      encoding="utf-8")
            preflight_sha = run.sha(preflight_path)

            def source_contract(_seed):
                return source_path, manifest_path, {"steps": 256}, pool, ids, source_sha

            def model_factory(seed, device):
                native = _source_core()
                native.load_state_dict(source_state, strict=True)
                native._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
                wrapped = PhaseStateIntegration(native, "phase")
                encoder = _Encoder()
                binder = run.NativeSpikeSlotBinder(seed=134)
                decoder = run.RelativeSlotRGBDecoder(seed=106)
                return (wrapped, encoder, None, None, None, 3.0, binder, decoder,
                        pool, ids, source_sha)

            original = {"batch": run.BATCH, "updates": train.UPDATES,
                        "passes": train.PASSES}
            run.BATCH, train.UPDATES, train.PASSES = 1, 2, 1
            try:
                with patch.object(run, "source_contract", side_effect=source_contract), \
                     patch.object(run.sw130, "source_contract", side_effect=source_contract), \
                     patch.object(run, "load_models", side_effect=model_factory), \
                     patch.object(run.source97, "EXPECTED_SOURCE_SHAS",
                                  {**run.source97.EXPECTED_SOURCE_SHAS, 1: source_sha}), \
                     patch.object(run.sw130, "validate_rgb_assets", return_value=assets), \
                     patch.object(run.sw130, "read_batch", return_value=image), \
                     patch.object(run.sw130, "encode", side_effect=lambda enc, *args:
                                  (enc.train_forward_count.add_(1) if enc.training else None,
                                   enc.scale * raw_gamma)[1]), \
                     patch.object(run.sw130, "TRAIN_RGB", str(root / "unused.npy")), \
                     patch.object(np, "load", return_value=np.zeros((1,), dtype=np.uint8)), \
                     patch.object(train, "_validate_preflight",
                                  return_value=(preflight, .1, preflight_sha)), \
                     patch.object(run, "ARCHIVE", root):
                    output_root = root / "trained"
                    for arm in run.ARMS:
                        manifest = train.train_arm(1, arm, device=torch.device("cpu"),
                                                   output_root=output_root, archive=root)
                        self.assertEqual(manifest["updates"], 2)
                        self.assertEqual(manifest["total_image_exposures"], 2)
                        checked, state, _ = train.validate_completed_training(
                            1, arm, output_root / f"{arm}_seed1", assets=assets, archive=root)
                        self.assertEqual(checked["updates"], 2)
                        self.assertEqual(len(json.loads(
                            (output_root / f"{arm}_seed1" / "history.json").read_text(
                                encoding="utf-8"))["records"]), 2)
                        expected_encoder_buffer = 0 if arm == "actual_frozen" else 2
                        self.assertEqual(int(state["encoder_state_dict"]["train_forward_count"]),
                                         expected_encoder_buffer)
                        if arm == "actual_frozen":
                            restored_core = {key.removeprefix("core."): value
                                             for key, value in state["wrapped_state_dict"].items()
                                             if key.startswith("core.")
                                             and key.removeprefix("core.") in source_state}
                            self.assertEqual(set(restored_core), set(source_state))
                            self.assertTrue(all(torch.equal(source_state[key], value)
                                                for key, value in restored_core.items()))
                            self.assertIsNone(torch.load(
                                output_root / f"{arm}_seed1" / "optimizers.pt",
                                map_location="cpu", weights_only=True)["joint"])
                            self.assertTrue(all(torch.isfinite(t).all() for t in
                                state["wrapped_state_dict"].values() if torch.is_tensor(t)))
            finally:
                run.BATCH, train.UPDATES, train.PASSES = (
                    original["batch"], original["updates"], original["passes"])
        finally:
            for path in root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir():
                    path.rmdir()
            root.rmdir()
            torch.random.set_rng_state(rng_state)


if __name__ == "__main__":
    unittest.main()
