"""Exercise the real SW0168 B1 adapter against the production QCC return schema."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import preflight_v2


class ProductionB1ApiContractTests(unittest.TestCase):
    def test_numpy_batch_qcc_and_actual_batched_q_a_shapes_reach_solver(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:, 16:] = 1
        production_qcc_return = labels[None, ...]
        q = torch.eye(1024, dtype=torch.float32)[None, ...]
        a = torch.ones((1, 1024, 1024), dtype=torch.float32)
        gamma = torch.ones((1, 8, 1024), dtype=torch.float32)
        trace = {"spikes": torch.zeros((1, 1024, 1024))}
        found = SimpleNamespace(
            encoder=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(()))]),
            wrapped=SimpleNamespace(core=SimpleNamespace(graph_generator=SimpleNamespace(
                log_coupling_gain=torch.tensor(0.0)))))

        def solver(q64, a64, gain, qcc, *, grid_shape):
            self.assertEqual(q64.shape, (1024, 1024))
            self.assertEqual(a64.shape, (1024, 1024))
            self.assertEqual(qcc.shape, (32, 32))
            self.assertEqual(grid_shape, (32, 32))
            return {"labels": qcc.copy(), "merge_log": [], "merge_count": 0}

        with (mock.patch.object(preflight_v2, "_capture_trace", return_value=(gamma, trace, a)),
              mock.patch.object(preflight_v2, "_q", return_value=q),
              mock.patch("collaborative_test.SW_0136_native32_transfer.evaluate._qcc_labels",
                         return_value=production_qcc_return),
              mock.patch("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed.foreground_induced_merge",
                         side_effect=solver)):
            report = preflight_v2._b1_source_readout(found, np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(report["qcc_labels"].shape, (32, 32))
        self.assertEqual(report["merge_count"], 0)


if __name__ == "__main__":
    unittest.main()
