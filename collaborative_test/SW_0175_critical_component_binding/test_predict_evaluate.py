"""CPU producer/consumer contracts for SW0175 fixed-320 inference."""
from __future__ import annotations

import hashlib
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0175_critical_component_binding import predict_evaluate as evaluate


class PredictionContractTests(unittest.TestCase):
    def test_public_predictor_uses_separate_registered_supports_and_accepts_capacity_1024(self):
        source = np.arange(1, 1025, dtype=np.int32).reshape(32, 32)
        native = np.arange(1024, 0, -1, dtype=np.int32).reshape(32, 32)
        q = torch.eye(1024, dtype=torch.float32).unsqueeze(0)
        a = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        trace = {name: torch.ones((1,), dtype=torch.float32) for name in evaluate.capture.TRACE_FIELDS}
        observed = []

        def merge(q_value, a_value, gain, labels, *, grid_shape):
            observed.append(np.asarray(labels, dtype=np.int32).copy())
            return {"labels": np.asarray(labels, dtype=np.int32), "merge_count": 0}

        found = SimpleNamespace(encode=lambda image: torch.zeros((1, 1)),
            wrapped=SimpleNamespace(core=SimpleNamespace(
                kuramoto=object(), graph_generator=SimpleNamespace(
                    log_coupling_gain=torch.tensor(np.log(2.0))))))
        with (mock.patch.object(evaluate.capture, "capture_source_coupling", return_value=(trace, a)),
              mock.patch.object(evaluate.capture, "_trace_check"),
              mock.patch.object(evaluate.capture, "_validate_actual_A"),
              mock.patch.object(evaluate.capture, "_actual_q", return_value=q),
              mock.patch.object(evaluate.sw136, "_qcc_labels", return_value=native[None]),
              mock.patch.object(evaluate.solver, "foreground_induced_merge", side_effect=merge)):
            result = evaluate._predict_one(found, torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                {"source_T": {"labels": source[None]}}, 0)
        self.assertEqual(len(observed), 2)
        np.testing.assert_array_equal(observed[0], source)
        np.testing.assert_array_equal(observed[1], native)
        self.assertEqual(int(result["labels"]["conditional"].max()), 1024)
        self.assertEqual(int(result["labels"]["standalone"].max()), 1024)

    def test_row_writer_reader_roundtrip_and_integrity(self):
        central_temp = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        central_temp.mkdir(parents=True, exist_ok=True)
        root = central_temp / f"sw175-row-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            labels = np.arange(1, 1025, dtype=np.int32).reshape(32, 32)
            support = np.ones((32, 32), dtype=np.int32)
            traces = {name: hashlib.sha256(name.encode()).hexdigest()
                      for name in evaluate.capture.TRACE_FIELDS}
            result = {"labels": {endpoint: labels.copy() for endpoint in evaluate.ENDPOINTS},
                "standalone_support": support,
                "q_sha256": "a" * 64, "actual_a_sha256": "b" * 64,
                "standalone_qcc_sha256": hashlib.sha256(
                    support.astype("<i4", copy=False).tobytes()).hexdigest(),
                "trace_sha256": traces, "gain": 1.25,
                "details": {endpoint: {"merge_count": 1, "support_foreground_patches": 1024}
                            for endpoint in evaluate.ENDPOINTS}}
            evaluate._commit_prediction_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64,
                result, rollout_seconds=.5, cuda_peak_allocated_bytes=4096)
            loaded, _record = evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64)
            for endpoint in evaluate.ENDPOINTS:
                np.testing.assert_array_equal(loaded[endpoint], labels)
            archive = root / "rows" / "0000" / "row.npz"
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64)
        finally:
            row = root / "rows" / "0000"
            if row.exists():
                for item in row.iterdir():
                    item.unlink()
                row.rmdir()
                (root / "rows").rmdir()
            if root.exists():
                root.rmdir()

    def test_fixed_bootstrap_uses_seed175_and_rejects_partial_vectors(self):
        values = np.linspace(0, 1, evaluate.COUNT)
        first = evaluate._bootstrap(values, np.zeros_like(values))
        self.assertEqual(first, evaluate._bootstrap(values, np.zeros_like(values)))
        self.assertEqual(first["seed"], 175)
        self.assertEqual(first["replicates"], 10_000)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(values[:-1], np.zeros_like(values[:-1]))


if __name__ == "__main__":
    unittest.main()
