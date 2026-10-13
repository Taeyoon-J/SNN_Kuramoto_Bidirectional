from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0149_spike_graph_modularity import train


class TinyGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.15))


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = TinyGraph()
        self.native = nn.Parameter(torch.tensor(0.3))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(-0.1))


class TinyWrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = TinyCore()
        self.a_d = nn.Parameter(torch.tensor(0.0))
        self.a_m = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))


class TinyFoundation:
    def __init__(self):
        self.encoder = TinyEncoder()
        self.wrapped = TinyWrapped()


class ModularityTrainStepTests(unittest.TestCase):
    def test_real_logical_update_accumulates_four_microbatches_and_steps_adam(self):
        foundation = TinyFoundation()
        groups = {
            "encoder": [foundation.encoder.theta],
            "graph": [foundation.wrapped.core.graph_generator.weight],
            "native_core": [foundation.wrapped.core.native],
            "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m,
                                    foundation.wrapped.b],
        }
        optimizer = torch.optim.Adam([
            {"params": groups["encoder"], "lr": 1e-2, "name": "encoder"},
            {"params": groups["graph"], "lr": 1e-2, "name": "graph"},
        ])
        params = groups["encoder"] + groups["graph"]
        cache = np.zeros((64, 128, 128, 3), dtype=np.uint8)
        adjacency = torch.tensor([[[0., 1.], [1., 0.]]]).expand(4, -1, -1).clone()
        calls = []

        def teacher_adj(_teacher, _foundation, rgb):
            self.assertEqual(tuple(rgb.shape), (4, 128, 128, 3))
            return adjacency

        def student_q(model, rgb):
            value = torch.sigmoid(model.encoder.theta +
                                  model.wrapped.core.graph_generator.weight)
            row0 = torch.stack((value * 0.0, value)).expand(rgb.shape[0], -1)
            row1 = torch.stack((value, value * 0.0)).expand(rgb.shape[0], -1)
            return torch.stack((row0, row1), dim=1), None

        initial = [param.detach().clone() for param in params]
        with mock.patch.object(train, "_teacher_adjacency", side_effect=teacher_adj), \
             mock.patch.object(train, "_student_q", side_effect=student_q):
            record = train.logical_update(
                "encoder_graph", 0, foundation=foundation, teacher=object(),
                optimizer=optimizer, groups=groups, params=params, cache=cache,
                ordered_rows=np.arange(16, dtype=np.int64),
                ordered_ids=list(range(16)), commit=lambda row: calls.append(row))
        self.assertEqual(len(record["microbatches"]), 4)
        self.assertEqual(record["image_ids"], list(range(16)))
        self.assertGreater(record["gradient_norms"]["encoder"], 0.0)
        self.assertGreater(record["gradient_norms"]["graph"], 0.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["update"], 1)
        self.assertTrue(all(not torch.equal(before, after)
                            for before, after in zip(initial, params)))
        self.assertEqual(train._optimizer_step_range(optimizer), [1, 1])


if __name__ == "__main__":
    unittest.main()
