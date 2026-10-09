import unittest
import hashlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from model import AdaptiveTemporalRGBModel, normalized_patch_centers
from run import _param_families, normalized_rgb_loss, rollout, warmup_optimizer_step
import run as runner


class TinyCore(nn.Module):
    def __init__(self, batch=2):
        super().__init__()
        self.drive = nn.Parameter(torch.tensor(0.3))
        self.gamma_channel_proj = nn.Linear(1, 4, bias=False)
        self.gamma_phase_gain = nn.Parameter(torch.ones(1))
        self.graph_generator = nn.Linear(1, 1)
        self.dendric_layer = nn.Linear(1, 1)
        self.membrane_layer = nn.Linear(1, 1)
        self.batch = batch

    def forward(self, gamma, return_core_out=True, return_theta=True):
        b = gamma.shape[0]
        spikes = self.drive.expand(b, 4, 256, 64)
        self.last_component_spikes = spikes
        theta = torch.zeros(b, 64, 256, 4)
        return [], spikes.mean(dim=1), spikes.mean(dim=1), theta


class GateMembrane(nn.Module):
    def forward(self, h, gate):
        return h, gate


class GateCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.membrane_layer = GateMembrane()
        self.spike_per_component = True
        self.gate_parameter = nn.Parameter(torch.linspace(0.1, 0.9, 64))

    def forward(self, gamma, return_core_out=True, return_theta=True):
        b = gamma.shape[0]
        emitted = []
        for t in range(64):
            gate = self.gate_parameter[t].expand(b * 4, 256)
            _mem, spike = self.membrane_layer(torch.zeros_like(gate), gate)
            emitted.append(spike)
        self.last_component_spikes = torch.stack(emitted, dim=-1).reshape(b, 4, 256, 64)
        theta = torch.zeros(b, 64, 256, 4)
        return [], self.last_component_spikes.mean(dim=1), self.last_component_spikes.mean(dim=1), theta


class CaptureHead(nn.Module):
    def forward(self, traces, _coordinates):
        self.traces = traces
        return {"sum": traces.sum()}


class TrainingPathTests(unittest.TestCase):
    def test_load_calibration_returns_calibration_path_after_fingerprint_checks(self):
        archive = MagicMock(name="archive")
        calibration_path = MagicMock(name="calibration_path")
        archive.__truediv__.return_value = calibration_path
        calibration_path.is_file.return_value = True
        source, source_manifest = MagicMock(name="source"), MagicMock(name="source_manifest")
        gamma, gamma_manifest = MagicMock(name="gamma"), MagicMock(name="gamma_manifest")
        root, dependency = MagicMock(name="root"), MagicMock(name="dependency")
        root.__truediv__.return_value = dependency
        kappa = np.asarray([1, 2, 3, 4], dtype="<f4")
        rms = np.asarray([.5, .6, .7, .8], dtype="<f4")
        dependency_sha = "dependency-sha"
        record = {
            "status": "passed", "experiment": "SW0123", "seed": 0,
            "stage": "train_calibration_activity",
            "source_core_sha256": "source-sha", "source_manifest_sha256": "source-manifest-sha",
            "source_training_ids": list(range(4096)), "gamma_rows": list(range(4096)),
            "ground_truth_used": False, "optimizer_updates": 0,
            "encoder_sha256": "enc", "preprocessing_sha256": "stats",
            "implementation_fingerprint": {"dependency.py": dependency_sha},
            "gamma_train_sha256": "gamma-sha", "gamma_train_manifest_sha256": "gamma-manifest-sha",
            "gamma_manifest_status": "complete", "kappa": kappa.tolist(),
            "spike_rms": rms.tolist(),
            "kappa_sha256": hashlib.sha256(kappa.tobytes()).hexdigest(),
            "spike_rms_sha256": hashlib.sha256(rms.tobytes()).hexdigest(),
            "adaptive_event_activity": {
                "occupancy_by_component": [.1, .2, .3, .4],
                "mixed_image_patch_fraction_by_component": [.5] * 4,
            }
        }
        calibration_path.read_text.return_value = json.dumps(record)
        ids = np.arange(4096, dtype=np.int64)
        sha_by_id = {id(source): "source-sha", id(source_manifest): "source-manifest-sha",
                     id(gamma): "gamma-sha", id(gamma_manifest): "gamma-manifest-sha",
                     id(dependency): dependency_sha, id(calibration_path): "calibration-sha"}
        with patch.multiple(
            runner, ARCHIVE=archive, ROOT=root,
            source_contract=lambda _seed: (source, source_manifest, {}, ids, ids),
        ), patch.object(runner, "sha", side_effect=lambda value: sha_by_id[id(value)]), \
                patch.object(runner.base, "EXPECTED_ENCODER_SHA256", "enc"), \
                patch.object(runner.base, "EXPECTED_PREPROCESSING_SHA256", "stats"), \
                patch.object(runner.base, "GAMMA_TRAIN", gamma), \
                patch.object(runner.base, "GAMMA_TRAIN_MANIFEST", gamma_manifest), \
                patch.object(runner.base, "validate_gamma_cache",
                             return_value=(torch.empty(0), {"status": "complete"})):
            result_path, loaded, *_ = runner.load_calibration(0)
            self.assertEqual(runner.sha(result_path), "calibration-sha")
        self.assertIs(result_path, calibration_path)
        self.assertEqual(loaded["stage"], "train_calibration_activity")

    def test_oscillator_drive_is_separate_trainable_family(self):
        core, encoder, model = TinyCore(), nn.Linear(1, 1), AdaptiveTemporalRGBModel()
        families = _param_families(core, encoder, model)
        self.assertEqual(len(families["oscillator_drive"]), 2)
        self.assertEqual({id(p) for p in families["oscillator_drive"]},
                         {id(core.gamma_channel_proj.weight), id(core.gamma_phase_gain)})

    def test_head_warmup_updates_only_head_from_detached_core_trace(self):
        torch.manual_seed(9)
        core = TinyCore()
        model = AdaptiveTemporalRGBModel()
        model.assignment_head.spike_rms.fill_(0.5)
        optimizer = torch.optim.Adam(list(model.assignment_head.parameters()) +
                                     list(model.rgb_decoder.parameters()), lr=3e-4)
        before_core = core.drive.detach().clone()
        before_head = model.assignment_head.assignment.weight.detach().clone()
        gamma = torch.zeros(2, 8, 256)
        rgb = torch.rand(2, 3, 128, 128)
        loss, norm = warmup_optimizer_step(core, gamma, model, "legacy_full",
                                           normalized_patch_centers(), rgb, optimizer)
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(norm))
        self.assertGreater(float(norm), 0)
        self.assertTrue(torch.equal(core.drive, before_core))
        self.assertFalse(torch.equal(model.assignment_head.assignment.weight, before_head))

    def test_gate_control_uses_actual_folded_membrane_gate_hook(self):
        core = GateCore()
        model = CaptureHead()
        gamma = torch.zeros(2, 8, 256)
        rollout(core, gamma, model, "gate_only_control", torch.zeros(256, 2))
        self.assertEqual(tuple(model.traces.shape), (2, 4, 256, 32))
        expected = core.gate_parameter[32:64].view(1, 1, 1, 32).expand(2, 4, 256, 32)
        self.assertTrue(torch.equal(model.traces, expected))
        model.traces.sum().backward()
        self.assertIsNotNone(core.gate_parameter.grad)
        self.assertGreater(float(core.gate_parameter.grad[32:64].abs().sum()), 0)

    def test_full_resolution_rgb_loss_is_per_image_channel_variance_normalized(self):
        rgb = torch.zeros(2, 3, 8, 8)
        rgb[0, :, :, 4:] = 1
        rgb[1, 0, :, :] = 1
        reconstructed = torch.ones_like(rgb) * 0.5
        loss, per_image = normalized_rgb_loss(reconstructed, rgb)
        variance = rgb.var(dim=(2, 3), unbiased=False).mean(dim=1).clamp_min(1e-6)
        expected = (reconstructed - rgb).square().mean(dim=(1, 2, 3)) / variance
        torch.testing.assert_close(per_image, expected)
        torch.testing.assert_close(loss, expected.mean())


if __name__ == "__main__":
    unittest.main()
