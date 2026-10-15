"""CPU producer/consumer checks for SW0168 prediction rows and support routing."""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0168_end_to_end_connectivity import predict_evaluate as evaluate

import torch


class PredictionContractTests(unittest.TestCase):
    def test_public_predictor_routes_conditional_and_standalone_support_separately(self):
        source_f = np.zeros((32, 32), dtype=np.int32); source_f.flat[[0, 1]] = [1, 1]
        native_f = np.zeros((32, 32), dtype=np.int32); native_f.flat[[12, 13, 14]] = [1, 1, 1]
        q = torch.eye(1024, dtype=torch.float32).unsqueeze(0)
        a = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        trace = {name: torch.ones((1,), dtype=torch.float32) for name in evaluate.capture.TRACE_FIELDS}
        captured_supports = []
        def merge(q_value, a_value, gain, labels, *, grid_shape):
            captured_supports.append(np.asarray(labels).copy())
            return {"labels": np.asarray(labels, dtype=np.int32), "merge_count": 0}
        found = SimpleNamespace(encode=lambda image: torch.zeros((1, 1)),
            wrapped=SimpleNamespace(core=SimpleNamespace(
                kuramoto=object(), graph_generator=SimpleNamespace(log_coupling_gain=torch.tensor(np.log(2.0))))))
        with (mock.patch.object(evaluate.capture, "capture_source_coupling", return_value=(trace, a)),
              mock.patch.object(evaluate.capture, "_trace_check"),
              mock.patch.object(evaluate.capture, "_validate_actual_A"),
              mock.patch.object(evaluate.capture, "_actual_q", return_value=q),
              mock.patch.object(evaluate.sw136, "_qcc_labels", return_value=native_f[None]),
              mock.patch.object(evaluate.solver, "foreground_induced_merge", side_effect=merge)):
            result = evaluate._predict_one(found, torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                {"source_T": {"labels": np.stack([source_f])}}, 0)
        self.assertEqual(len(captured_supports), 2)
        np.testing.assert_array_equal(captured_supports[0], source_f)
        np.testing.assert_array_equal(captured_supports[1], native_f)
        np.testing.assert_array_equal(result["labels"]["conditional"] > 0, source_f > 0)
        np.testing.assert_array_equal(result["labels"]["standalone"] > 0, native_f > 0)

    def test_atomic_row_producer_roundtrips_through_real_reader_and_detects_tamper(self):
        temp_root = Path(__file__).resolve().parents[1] / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / f"sw168-pred-row-{os.getpid()}"
        root.mkdir(exist_ok=False)
        try:
            labels = np.zeros((32, 32), dtype=np.int32); labels.flat[:3] = [1, 1, 2]
            support = np.zeros((32, 32), dtype=np.int32); support.flat[:3] = 1
            traces = {name: hashlib.sha256(name.encode()).hexdigest() for name in evaluate.capture.TRACE_FIELDS}
            result = {"labels": {endpoint: labels.copy() for endpoint in evaluate.ENDPOINTS},
                "standalone_support": support,
                "q_sha256": "a"*64, "actual_a_sha256": "b"*64,
                "standalone_qcc_sha256": hashlib.sha256(support.astype("<i4", copy=False).tobytes()).hexdigest(),
                "trace_sha256": traces, "gain": 1.25,
                "details": {endpoint: {"merge_count": 1, "support_foreground_patches": 3}
                            for endpoint in evaluate.ENDPOINTS}}
            evaluate._commit_prediction_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64,
                result, rollout_seconds=.5, cuda_peak_allocated_bytes=4096)
            loaded, record = evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64)
            self.assertEqual(record["cuda_peak_allocated_bytes"], 4096)
            for endpoint in evaluate.ENDPOINTS:
                np.testing.assert_array_equal(loaded[endpoint], labels)
            self.assertEqual(record["standalone_qcc_sha256"], result["standalone_qcc_sha256"])
            archive = root / "rows" / "0000" / "row.npz"
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64)
        finally:
            row_dir = root / "rows" / "0000"
            if row_dir.exists():
                for file in row_dir.iterdir():
                    file.unlink()
                row_dir.rmdir()
                row_dir.parent.rmdir()
            root.rmdir()

    def test_bootstrap_is_fixed_seed_and_requires_complete_pair(self):
        x = np.linspace(0, 1, evaluate.COUNT)
        a = evaluate._bootstrap(x, np.zeros_like(x))
        b = evaluate._bootstrap(x, np.zeros_like(x))
        self.assertEqual(a, b)
        self.assertEqual(a["seed"], 168)
        self.assertEqual(a["replicates"], 10_000)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(x[:-1], np.zeros_like(x[:-1]))


if __name__ == "__main__":
    unittest.main()
