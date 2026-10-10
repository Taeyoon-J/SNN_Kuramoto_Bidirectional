"""Focused SW0133 production preflight-path regression tests."""
from __future__ import annotations

import sys
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np
import torch

from collaborative_test.SW_0133_soft_partition_rgb import run
from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import RelativeRGBDecoder


class PairedPreflightPathTests(unittest.TestCase):
    def test_initial_arm_check_runs_detached_rgb_route_and_compares_soft_p(self):
        module = torch.nn.Linear(1, 1, bias=False)
        model = (module, module, None, None, None, None, module, [], [], "source")
        gamma = torch.tensor([[1.0, 2.0]])
        q = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
        labels = torch.tensor([[0, 1]])
        hard = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        result = (None, torch.tensor([1.0]), torch.tensor([2.0]),
                  torch.tensor([3.0]), torch.tensor([4.0]))
        target = torch.zeros(1, 2, 3)
        row = (gamma, result, q, labels, hard, [], target)
        assignment_modes = []
        p = torch.tensor([[0.8, 0.2], [0.1, 0.9]])

        def fake_rgb(_q, _hard, _gamma, _target, _decoder, *, assignment_live=True,
                     **_kwargs):
            assignment_modes.append(assignment_live)
            prediction = torch.ones(1, 2, 3)
            return prediction, torch.tensor(0.5), [{"P": p}]

        with mock.patch.object(run, "load_models", return_value=model), \
             mock.patch.object(run, "_batch_forward", return_value=row), \
             mock.patch.object(run, "_rgb_batch", side_effect=fake_rgb), \
             mock.patch.object(run, "_old_loss", return_value=torch.tensor(2.0)):
            report = run._paired_initial_arm_check(1, torch.device("cpu"),
                                                   torch.zeros(1, 3, 2, 2), model)
        self.assertEqual(assignment_modes, [True, True, False])
        self.assertTrue(report["initial_soft_p_rgb_equal"])

    def test_initial_arm_check_rejects_changed_p_even_when_rgb_mocks_match(self):
        module = torch.nn.Linear(1, 1, bias=False)
        model = (module, module, None, None, None, None, module, [], [], "source")
        row = (torch.ones(1, 2),
               (None, torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1)),
               torch.eye(2), torch.zeros(1, 2), [torch.eye(2)], [], torch.zeros(1, 2, 3))
        calls = 0

        def fake_rgb(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            p_value = torch.full((2, 2), 0.5) if calls < 3 else torch.eye(2)
            return torch.zeros(1, 2, 3), torch.tensor(0.0), [{"P": p_value}]

        with mock.patch.object(run, "load_models", return_value=model), \
             mock.patch.object(run, "_batch_forward", return_value=row), \
             mock.patch.object(run, "_rgb_batch", side_effect=fake_rgb), \
             mock.patch.object(run, "_old_loss", return_value=torch.tensor(2.0)):
            with self.assertRaisesRegex(AssertionError, "soft assignment P"):
                run._paired_initial_arm_check(1, torch.device("cpu"),
                                               torch.zeros(1, 3, 2, 2), model)

    def test_actual_core_detached_disposable_step_keeps_rgb_credit_out_of_q(self):
        """Exercise production forward/optimizer code with a source-shaped CPU core."""
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        warm_path = Path(tempfile.gettempdir()) / f"sw133_step_{uuid.uuid4().hex}.pt"
        try:
            torch.manual_seed(133)
            device = torch.device("cpu")
            core = run.sw130.source97.make_core("cpu", steps=64)
            source_state = {key: value.detach().clone() for key, value in core.state_dict().items()}
            core.load_state_dict(source_state, strict=True)
            core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
            wrapped = run.sw130.PhaseStateIntegration(core, "phase")

            class TinyEncoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 8, kernel_size=1)

                def forward(self, images):
                    return self.conv(images)

            encoder = TinyEncoder()
            patcher = run.sw130.FeaturePatchGammaInitializer(grid_size=16)
            decoder = RelativeRGBDecoder()
            decoder_parameters = list(decoder.parameters())
            warm_optimizer = torch.optim.Adam(decoder_parameters, lr=run.DECODER_LR)
            for _ in range(32):
                warm_optimizer.zero_grad(set_to_none=True)
                for parameter in decoder_parameters:
                    parameter.grad = torch.full_like(parameter, 1e-3)
                warm_optimizer.step()
            training_ids = list(range(4096))
            torch.save({
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": warm_optimizer.state_dict(),
                "source_core_sha256": "a" * 64,
                "training_ids_sha256": hashlib.sha256(
                    np.asarray(training_ids, dtype="<i8").tobytes()).hexdigest(),
                "asset_hashes": {"synthetic": "b" * 64},
                "warmup_updates": 32,
            }, warm_path)
            model = (wrapped, encoder, patcher, torch.zeros(1, 8, 1, 1),
                     torch.ones(1, 8, 1, 1), 3.0, decoder, np.arange(16),
                     training_ids, "a" * 64)
            images = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8).float()
            fake_cache = np.zeros((16, 128, 128, 3), dtype=np.uint8)
            with mock.patch.object(run, "load_models", return_value=model), \
                 mock.patch.object(run.sw130, "read_batch", return_value=images), \
                 mock.patch.object(run.np, "load", return_value=fake_cache):
                record = run._disposable_update(
                    1, "phase_detached", device, np.arange(16), 1.0, warm_path)

            self.assertEqual(record["arm"], "phase_detached")
            self.assertEqual(record["rgb_to_q_gradient_norm"], 0.0)
            self.assertEqual(record["rgb_gradient_norms_by_family"], {})
            self.assertTrue(all(record["rgb_gradient_unused_integration"].values()))
            self.assertTrue(all(all(value == 0.0 for value in values)
                                for values in record["integration_parameter_gradient_abs_by_component"].values()))
            self.assertTrue(record["joint_and_decoder_changed"])
            self.assertEqual(record["changed_decoder_parameter_count"], 6)
            self.assertEqual(record["optimizer_updates"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(record["old_loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(record["rgb_loss"])))
            self.assertTrue(all(torch.equal(value, core.state_dict()[key])
                                for key, value in source_state.items()) is False)
        finally:
            warm_path.unlink(missing_ok=True)
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
