"""CPU API regressions for the public SW0162 source preflight helpers."""
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0162_connected_signed_partition import preflight
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto


class FakeCache:
    shape = (70000, 128, 128, 3)
    ndim = 4
    dtype = np.dtype(np.uint8)

    def __getitem__(self, rows):
        values = np.asarray(rows)
        out = np.zeros((len(values), 128, 128, 3), dtype=np.uint8)
        for index, value in enumerate(values):
            out[index].fill(int(value) % 251)
        return out


class PreflightTests(unittest.TestCase):
    def test_capture_uses_actual_prepare_tuple_and_restores_instance_descriptor(self):
        layer = graphVectorKuramoto(4, device="cpu", kuramoto_backend="factorized")
        adjacency = torch.tensor([[[1., .2, .0, .1], [.2, 1., .3, .0],
                                   [.0, .3, 1., .4], [.1, .0, .4, 1.]]])
        original = layer.prepare_coupling
        returned = []

        def rollout():
            result = layer.prepare_coupling(adjacency, batch_size=1, num_units=4, device="cpu")
            returned.append(result)
            return result

        result, actual_a = preflight.capture_source_coupling(layer, rollout)
        self.assertIs(result, returned[0])
        self.assertEqual(len(result), 3)
        torch.testing.assert_close(actual_a, result[0], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_a, actual_a.transpose(1, 2)))
        self.assertTrue(torch.all(actual_a > 0))
        self.assertNotIn("prepare_coupling", layer.__dict__)
        self.assertEqual(layer.prepare_coupling.__func__, original.__func__)

    def test_capture_restores_original_callable_when_rollout_raises(self):
        layer = graphVectorKuramoto(3, device="cpu", kuramoto_backend="factorized")
        original = layer.prepare_coupling
        adjacency = torch.eye(3).unsqueeze(0)

        def broken_rollout():
            layer.prepare_coupling(adjacency, batch_size=1, num_units=3, device="cpu")
            raise RuntimeError("injected rollout failure")

        with self.assertRaisesRegex(RuntimeError, "injected rollout failure"):
            preflight.capture_source_coupling(layer, broken_rollout)
        self.assertNotIn("prepare_coupling", layer.__dict__)
        self.assertEqual(layer.prepare_coupling.__func__, original.__func__)

    def test_fixed_source_train_row_order_and_rgb_bytes(self):
        ids = list(range(90000, 94096))
        rows = np.arange(4096, dtype=np.int64)
        rgb, selected_rows, selected_ids = preflight.select_train_rows(FakeCache(), rows, ids)
        self.assertEqual(tuple(selected_rows), tuple(preflight.POSITIONS))
        self.assertEqual(selected_ids, [ids[index] for index in preflight.POSITIONS])
        self.assertEqual(rgb.shape, (8, 128, 128, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual([int(rgb[i, 0, 0, 0]) for i in range(8)],
                         [position % 251 for position in preflight.POSITIONS])

    def test_production_fingerprint_resolves_actual_source_dependencies(self):
        fingerprint = preflight.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0162_connected_signed_partition/preflight.py", fingerprint)
        self.assertIn("collaborative_test/SW_0162_connected_signed_partition/connected_signed.py", fingerprint)
        self.assertIn("collaborative_test/SW_0136_native32_transfer/evaluate.py", fingerprint)
        self.assertIn("snn_kuramoto_bidirectional/graph_generator.py", fingerprint)

    def test_public_run_validates_cpu_inputs_before_cuda_probe(self):
        prepared = {"cpu_validation_complete": True}
        with mock.patch.object(preflight, "prepare_preflight_cpu", return_value=prepared) as cpu_prepare, \
             mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires the registered owner-held CUDA device"):
                preflight.run_preflight(device="cuda:0",
                    output=Path(__file__).with_name(f".must-not-exist-{id(self)}.json"))
        cpu_prepare.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
