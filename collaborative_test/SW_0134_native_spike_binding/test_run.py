"""Focused CPU contract tests for the SW0134 production batch path."""
from __future__ import annotations

import sys
import tempfile
import unittest
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

from collaborative_test.SW_0134_native_spike_binding import run


class _Criterion:
    def __call__(self, *, plv, theta=None):
        value = plv.sum() * 0 + 1.0
        if theta is not None:
            value = value + theta.sum() * 0
        return value, {}


class SW0134RunnerTests(unittest.TestCase):
    def test_production_batch_routes_emitted_spikes_or_native_gate_history(self):
        batch = 1
        images = torch.zeros(batch, 3, 128, 128)
        emitted = torch.full((batch, 4, 256, 1024), 0.125)
        # Fractional gates make this catch accidental event reconstruction or
        # routing the component spikes to the gate-control binder.
        gates = torch.full_like(emitted, 0.375)
        trace = {"component_spikes": emitted, "component_gates": gates,
                 "theta": torch.zeros(batch, 1024, 256, 4)}
        calls = []

        def fake_reconstruct(values, binder, decoder, **kwargs):
            calls.append(values.detach().clone())
            prediction = torch.zeros(batch, 128, 128, 3)
            assignments = torch.zeros(batch, 256, 11)
            slots = torch.zeros(batch, 11, 64)
            labels = torch.zeros(batch, 16, 16, dtype=torch.long)
            return prediction, assignments, slots, torch.empty(0), labels

        q = torch.ones(batch, 256, 256)
        with patch.object(run.sw130, "encode", return_value=torch.zeros(batch, 256, 8)), \
             patch.object(run, "late_rollout", return_value=trace), \
             patch.object(run, "spike_synchrony_affinity", return_value=q), \
             patch.object(run, "phase_locking_value", return_value=q), \
             patch.object(run.sw130, "make_criterion", return_value=_Criterion()), \
             patch.object(run, "reconstruct_from_spikes", side_effect=fake_reconstruct):
            run._batch_forward(None, None, None, None, None, 3.0, None, None,
                               images, "actual_joint", live_tail_steps=0,
                               checkpoint_chunks=False)
            run._batch_forward(None, None, None, None, None, 3.0, None, None,
                               images, "gate_joint", live_tail_steps=0,
                               checkpoint_chunks=False)

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0], emitted[..., -512:]))
        self.assertTrue(torch.equal(calls[1], gates[..., -512:]))
        self.assertFalse(torch.equal(calls[0], calls[1]))

    def test_registered_seed_order_and_training_contract_constants(self):
        self.assertEqual(run.SEEDS, (0, 1, 2))
        self.assertEqual(run.ARMS, ("actual_joint", "gate_joint", "actual_frozen"))
        self.assertEqual((run.BATCH, run.TRAIN_STEPS, run.TRAIN_SETTLE, run.TAIL),
                         (16, 1024, 512, 64))
        self.assertEqual(run.WARMUP_UPDATES, 32)
        self.assertEqual(run.HEAD_LR, 3e-4)

    def test_actual_disposable_optimizer_path_for_all_three_arms(self):
        """Exercise the real rollout, RGB loss, gradient, and Adam update path."""
        from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1340134)
        fixture = Path(tempfile.gettempdir()) / f"sw0134_cpu_disposable_{__import__('uuid').uuid4().hex[:10]}"
        fixture.mkdir(parents=False, exist_ok=False)
        try:
            checkpoint = fixture / "source.pt"
            manifest = fixture / "source_manifest.json"
            def source_core():
                hp = S2NetHyperparameters(
                    num_feature_maps=8, num_regions=256, sc=torch.eye(256), osc_dim=4,
                    gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
                    theta_init="gamma", graph_mode="learned", graph_top_k=4,
                    graph_spatial_decay=.35, geodesic_steps=3, geodesic_radius=1.5,
                    geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
                    kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
                    num_time_steps=1024, spike_per_component=True, gate_mode="raw",
                    membrane_vth=.06, spike_pulse_gain=0.0,
                    dendritic_projection="shared", spike_spatial_grid_size=16,
                ).validate()
                return S2NetCore(hp, device="cpu")

            core0 = source_core()
            source_state = core0.state_dict()
            torch.save(source_state, checkpoint)
            manifest.write_text('{"status":"complete"}', encoding="utf-8")
            source_sha = run.sha(checkpoint)
            ids = list(range(4096))
            pool = np.arange(4096, dtype=np.int64)
            id_sha = __import__("hashlib").sha256(
                np.asarray(ids[:512], dtype="<i8").tobytes()).hexdigest()
            all_id_sha = __import__("hashlib").sha256(
                np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            assets = {"synthetic_fixture": "fixed"}
            fp = {"synthetic": "fingerprinted"}
            warm_files = {}
            for arm in ("actual_joint", "gate_joint"):
                binder = run.NativeSpikeSlotBinder(seed=134)
                decoder = run.RelativeSlotRGBDecoder(seed=106)
                head = list(binder.parameters()) + list(decoder.parameters())
                opt = torch.optim.Adam(head, lr=run.HEAD_LR)
                # Real Adam moments at the registered warmup step count.
                for step in range(run.WARMUP_UPDATES):
                    opt.zero_grad(set_to_none=True)
                    objective = sum(parameter.square().sum() * (0.5 + i / 100.)
                                     for i, parameter in enumerate(head))
                    objective.backward(); opt.step()
                path = fixture / f"warm_{arm}.pt"
                warm_payload = {"experiment":"SW0134_native_spike_binding", "seed":1,
                    "arm":arm, "updates":run.WARMUP_UPDATES, "batch_size":run.BATCH,
                    "training_ids":ids[:512], "training_ids_sha256":id_sha,
                    "all_training_ids_sha256":all_id_sha,
                    "source_core_sha256":source_sha,
                    "source_manifest_sha256":run.sha(manifest), "asset_hashes":assets,
                    "implementation_fingerprint":fp,
                    "binder_state_dict":binder.state_dict(),
                    "decoder_state_dict":decoder.state_dict(),
                    "optimizer_state_dict":opt.state_dict()}
                torch.save(warm_payload, path)
                warm_files[arm] = {"path":str(path), "sha256":run.sha(path)}

            class Encoder(nn.Module):
                def __init__(self):
                    super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0))
                    self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))

            def source_contract(_seed):
                return checkpoint, manifest, {"steps":256}, pool, ids, source_sha

            def model_factory(seed, device):
                core = source_core()
                core.load_state_dict(source_state, strict=True)
                core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
                wrapped = PhaseStateIntegration(core, "phase")
                enc = Encoder().to(device)
                binder = run.NativeSpikeSlotBinder(seed=134).to(device)
                decoder = run.RelativeSlotRGBDecoder(seed=106).to(device)
                return wrapped, enc, None, None, None, 3., binder, decoder, pool, ids, source_sha

            def fake_encode(enc, *args):
                if enc.training:
                    enc.forward_count.add_(1)
                return enc.scale * raw_gamma

            raw_gamma = torch.randn(1, 8, 256) * 2.0
            image = (torch.rand(1, 3, 128, 128) * 255.).float()
            with patch.object(run, "source_contract", side_effect=source_contract), \
                 patch.object(run.sw130, "source_contract", side_effect=source_contract), \
                 patch.object(run, "load_models", side_effect=model_factory), \
                 patch.object(run, "implementation_fingerprint", return_value=fp), \
                 patch.object(run.source97, "EXPECTED_SOURCE_SHAS",
                              {**run.source97.EXPECTED_SOURCE_SHAS, 1: source_sha}), \
                 patch.object(run.sw130, "validate_rgb_assets", return_value=assets), \
                 patch.object(run.sw130, "encode", side_effect=fake_encode), \
                 patch.object(run.sw130, "read_batch", return_value=image):
                for arm in run.ARMS:
                    report = run._disposable_update(
                        1, arm, torch.device("cpu"), pool, np.zeros((1,), dtype=np.uint8),
                        0.1, warm_files, assets)
                    self.assertTrue(report["throwaway_only"])
                    self.assertGreater(report["head_changed_parameter_count"], 0)
                    if arm == "gate_joint":
                        for key in ("dendrite", "membrane", "a_d", "a_m", "b"):
                            self.assertEqual(report["rgb_gradient_norms_by_source_family"].get(key, 0.), 0.)
                    if arm == "actual_frozen":
                        self.assertTrue(report["source_parameters_unchanged"])
                        self.assertEqual(report["source_training_mode"], "eval")
                        self.assertEqual(report["joint_changed_parameter_count"], 0)
                    else:
                        self.assertGreater(report["joint_changed_parameter_count"], 0)
        finally:
            for item in fixture.iterdir():
                item.unlink()
            fixture.rmdir()
            torch.random.set_rng_state(rng_state)


if __name__ == "__main__":
    unittest.main()
