"""Regression for the production NumPy-returning B1 QCC API."""
from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import preflight_v2


class B1ReadoutApiTests(unittest.TestCase):
    def test_numpy_qcc_output_is_consumed_without_tensor_detach_assumption(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:, :16] = 1
        q = torch.eye(1024, dtype=torch.float32)
        a = torch.ones((1, 1024, 1024), dtype=torch.float32)
        gamma = torch.ones((1, 8, 1024), dtype=torch.float32)
        trace = {"spikes": torch.zeros((1, 1024, 1024))}
        parameter = torch.nn.Parameter(torch.zeros(()))
        found = SimpleNamespace(
            encoder=torch.nn.ParameterList([parameter]),
            wrapped=SimpleNamespace(core=SimpleNamespace(graph_generator=SimpleNamespace(
                log_coupling_gain=torch.tensor(0.0)))))
        fake_result = {"labels": labels.copy(), "merge_log": [], "merge_count": 0}
        with (mock.patch.object(preflight_v2, "_capture_trace", return_value=(gamma, trace, a)),
              mock.patch.object(preflight_v2, "_q", return_value=q),
              mock.patch("collaborative_test.SW_0136_native32_transfer.evaluate._qcc_labels",
                         return_value=[labels]),
              mock.patch("collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed.foreground_induced_merge",
                         return_value=fake_result)):
            result = preflight_v2._b1_source_readout(found, np.zeros((128, 128, 3), dtype=np.uint8))
        expected = hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest()
        self.assertEqual(result["qcc_sha256"], expected)
        self.assertEqual(result["merge_count"], 0)
        self.assertEqual(result["e0_labels_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
