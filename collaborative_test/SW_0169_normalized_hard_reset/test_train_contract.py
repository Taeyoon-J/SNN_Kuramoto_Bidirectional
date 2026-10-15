"""CPU contracts for the SW0169 trainer's registered loss and readout path."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import train


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class _Coupling:
    def __init__(self, matrix):
        self.matrix = matrix

    def prepare_coupling(self, *args, **kwargs):
        return self.matrix, self.matrix, self.matrix


class _Graph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_coupling_gain = torch.nn.Parameter(torch.zeros(()), requires_grad=False)


class _Core:
    def __init__(self, matrix):
        self.kuramoto = _Coupling(matrix)
        self.graph_generator = _Graph()


class _Wrapped(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.core = _Core(matrix)
        self.b = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


class _Foundation:
    def __init__(self, matrix):
        self.encoder = _Encoder()
        self.patcher = torch.nn.Identity()
        self.wrapped = _Wrapped(matrix)

    def encode(self, rgb):
        return rgb


class TrainContractTests(unittest.TestCase):
    def test_both_registered_arms_use_identical_maximin_objective(self):
        q = torch.tensor([[0.7, 0.6, 0.1, 0.1],
                          [0.6, 0.7, 0.3, 0.1],
                          [0.1, 0.3, 0.7, 0.5],
                          [0.1, 0.1, 0.5, 0.7]], requires_grad=True)
        target = np.asarray([[1, 1], [2, 2]], dtype=np.int32)
        support = np.ones((2, 2), dtype=bool)
        live = train._training_loss("history_live", q, target, support)
        detached = train._training_loss("history_detached", q, target, support)
        torch.testing.assert_close(live, detached, rtol=0, atol=0)
        g_live, = torch.autograd.grad(live, q, retain_graph=True)
        g_detached, = torch.autograd.grad(detached, q)
        torch.testing.assert_close(g_live, g_detached, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            train._training_loss("other", q, target, support)

    def test_registered_order_is_fixed_size_and_deterministic(self):
        a, b = train._epoch_orders(), train._epoch_orders()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)
        for epoch in a:
            self.assertEqual(len(epoch), train.COUNT)
            self.assertEqual(set(epoch), set(range(train.COUNT)))
        batches = [epoch[i:i + train.BATCH]
                   for epoch in a for i in range(0, train.COUNT, train.BATCH)]
        self.assertEqual(len(batches), train.UPDATES)
        self.assertTrue(all(len(batch) == train.BATCH for batch in batches))

    def test_diagnostic_uses_real_batched_coupling_and_sw164_solver_api(self):
        n, steps = 1024, 513
        matrix = torch.ones((1, n, n), dtype=torch.float32)
        found = _Foundation(matrix)
        q = torch.full((1, n, n), 0.2, dtype=torch.float32)
        q = (q + q.transpose(-1, -2)) * 0.5
        raw = torch.zeros((1, 4, n, steps), dtype=torch.float32)
        gates = torch.ones_like(raw)
        spikes = raw * gates
        trace = {
            "raw_events": raw,
            "component_gates": gates,
            "component_spikes": spikes,
            "component_membrane": torch.zeros_like(raw),
            "spikes": torch.zeros((1, n, steps), dtype=torch.float32),
            "component_membrane_unused": torch.zeros_like(raw),
        }
        # The actual source helper captures tuple[0] and returns a BxNxN tensor.
        def fake_rollout(*args, **kwargs):
            found.wrapped.core.kuramoto.prepare_coupling()
            return trace

        ids = np.arange(8, dtype=np.int64) + 100
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[0, 0] = 1
        data = {"rgb": np.zeros((8, 1, 1, 3), dtype=np.uint8),
                "supports": np.repeat(labels[None], 8, axis=0),
                "targets": np.repeat(labels[None], 8, axis=0)}
        caches = {"data": data, "source_ids": ids}
        evaluator = train._module("collaborative_test.SW_0136_native32_transfer.evaluate")
        classifier = train._module("snn_kuramoto_bidirectional.spike_classifier")
        native_objective = train._module("collaborative_test.SW_0155_supervised_spike_relations.objective")
        with (mock.patch.object(train.hard_reset, "rollout", side_effect=fake_rollout),
              mock.patch.object(evaluator, "_qcc_labels", return_value=np.repeat(labels[None], 1, axis=0)),
              mock.patch.object(classifier, "spike_synchrony_affinity", return_value=q),
              mock.patch.object(native_objective, "_component_correlations",
                                return_value=torch.zeros((1, 4, n, n)))):
            result = train._diagnose_eight_169(
                found, caches, torch.ones(4), arm="history_live", phase="contract")
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(result["selected_image_ids"], ids.tolist())
        self.assertEqual(result["rows"][0]["actual_a_sha256"],
                         train._sha_bytes(matrix[0].contiguous().numpy().tobytes()))
        self.assertEqual(result["rows"][0]["partition_foreground_patches"], 1)
        self.assertFalse(result["validation_data_read"])


if __name__ == "__main__":
    unittest.main()
