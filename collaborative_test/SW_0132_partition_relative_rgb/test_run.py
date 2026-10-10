"""Focused preflight production-path smoke tests for SW0132."""
import sys
import unittest
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0110_xy_graph_route import run as source97
from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import SharedRGBDecoder
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0132_partition_relative_rgb import run


class _TinyEncoder(nn.Module):
    def forward(self, images):
        pooled = torch.nn.functional.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (16, 16))
        return pooled.repeat(1, 8, 1, 1)


class _PatchFlatten(nn.Module):
    def forward(self, features):
        return features.flatten(2)


class PreflightProductionPathTests(unittest.TestCase):
    def test_phase_and_constant_batch_paths_keep_old_phase_plv_and_rgb_shapes(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(132)
            native_template = source97.make_core("cpu", steps=run.STEPS)
            native_template._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            source_state = copy.deepcopy(native_template.state_dict())
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(106)
                decoder_state = SharedRGBDecoder().state_dict()
            gamma_encoder = _TinyEncoder()
            patcher = _PatchFlatten()
            images = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.float32)
            mean = torch.zeros(1, 8, 1, 1)
            std = torch.ones(1, 8, 1, 1)
            criterion = run.sw130.make_criterion()
            outputs = []
            for arm in run.ARMS:
                native = source97.make_core("cpu", steps=run.STEPS)
                native.load_state_dict(source_state, strict=True)
                native._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
                wrapped = PhaseStateIntegration(native, arm)
                wrapped.eval()
                result = run._batch_forward(wrapped, gamma_encoder, patcher, mean, std, 3.0, images)
                gamma, core_result, q, _labels, hard, _groups, target = result
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(106)
                    decoder = run.RelativeRGBDecoder()
                decoder.load_state_dict(decoder_state, strict=True)
                prediction, losses, details = run._rgb_batch(
                    q, hard, gamma, target, decoder, chunk_size=8192,
                    checkpoint_chunks=False)
                self.assertEqual(tuple(gamma.shape), (1, 8, 256))
                self.assertEqual(tuple(prediction.shape), (1, 128, 128, 3))
                self.assertEqual(tuple(losses.shape), (1,))
                self.assertTrue(torch.isfinite(losses).all())
                self.assertEqual(len(details), 1)
                self.assertTrue(torch.isfinite(core_result[3]).all())  # phase PLV in both arms
                outputs.append((core_result[3], q, prediction))
            for left, right in zip(outputs[0], outputs[1]):
                self.assertTrue(torch.equal(left, right))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
