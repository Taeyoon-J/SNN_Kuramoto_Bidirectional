"""SW0169 fixed320 prediction-row, support and scoring contracts."""
from __future__ import annotations

import hashlib
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import predict_evaluate as evaluate


def _central_temp():
    root = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
    root.mkdir(parents=True, exist_ok=True)
    owned = root / f"sw169-eval-{os.getpid()}-{time.time_ns()}"
    owned.mkdir(exist_ok=False)
    return owned


class PredictionContractTests(unittest.TestCase):
    def test_public_predictor_routes_source_and_native_supports(self):
        source = np.zeros((32, 32), dtype=np.int32)
        source.flat[:2] = [1, 1]
        native = np.zeros((32, 32), dtype=np.int32)
        native.flat[12:15] = [1, 1, 1]
        q = torch.eye(1024, dtype=torch.float32).unsqueeze(0)
        a = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        trace = {name: torch.zeros((1,), dtype=torch.float32)
                 for name in evaluate.capture.TRACE_FIELDS}
        trace["raw_events"] = torch.zeros((1, 4, 1024, 513), dtype=torch.float32)
        trace["component_gates"] = torch.ones((1, 4, 1024, 513), dtype=torch.float32)
        trace["component_spikes"] = torch.zeros((1, 4, 1024, 513), dtype=torch.float32)
        supports = []

        def merge(_q, _a, gain, labels, *, grid_shape):
            self.assertEqual(grid_shape, (32, 32))
            self.assertAlmostEqual(gain, 2.0, places=6)
            supports.append(np.asarray(labels).copy())
            return {"labels": np.asarray(labels, dtype=np.int32), "merge_count": 0}

        found = SimpleNamespace(encode=lambda _image: torch.zeros((1, 1)), wrapped=SimpleNamespace(
            core=SimpleNamespace(kuramoto=object(), graph_generator=SimpleNamespace(
                log_coupling_gain=torch.tensor(np.log(2.0))))))
        with (mock.patch.object(evaluate.capture, "capture_source_coupling", return_value=(trace, a)),
              mock.patch.object(evaluate.capture, "_trace_check"),
              mock.patch.object(evaluate.capture, "_validate_actual_A"),
              mock.patch.object(evaluate.capture, "_actual_q", return_value=q),
              mock.patch.object(evaluate.sw136, "_qcc_labels", return_value=native[None]),
              mock.patch.object(evaluate.solver, "foreground_induced_merge", side_effect=merge)):
            out = evaluate._predict_one(found, torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                {"source_T": {"labels": source[None]}}, 0, "history_live", torch.ones(4))
        self.assertEqual(len(supports), 2)
        np.testing.assert_array_equal(supports[0], source)
        np.testing.assert_array_equal(supports[1], native)
        np.testing.assert_array_equal(out["labels"]["conditional"] > 0, source > 0)
        np.testing.assert_array_equal(out["labels"]["standalone"] > 0, native > 0)
        self.assertEqual(set(out["trace_sha256"]), set(evaluate.capture.TRACE_FIELDS))

    def test_atomic_row_roundtrip_tamper_and_capacity(self):
        root = _central_temp()
        try:
            labels = np.zeros((32, 32), dtype=np.int32)
            labels.flat[:3] = [1, 1, 2]
            support = np.zeros((32, 32), dtype=np.int32)
            support.flat[:3] = 1
            traces = {name: hashlib.sha256(name.encode()).hexdigest()
                      for name in evaluate.capture.TRACE_FIELDS}
            result = {"labels": {endpoint: labels.copy() for endpoint in evaluate.ENDPOINTS},
                "standalone_support": support,
                "q_sha256": "a"*64, "actual_a_sha256": "b"*64,
                "standalone_qcc_sha256": hashlib.sha256(support.astype("<i4").tobytes()).hexdigest(),
                "trace_sha256": traces, "gain": 1.25,
                "details": {endpoint: {"merge_count": 0, "support_foreground_patches": 3}
                            for endpoint in evaluate.ENDPOINTS},
                "event_diagnostics": {"raw_on_fraction": .2, "active_gate_fraction": .8,
                                       "raw_on_when_gate_positive": .25}}
            evaluate._commit_prediction_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64,
                result, rollout_seconds=.5, cuda_peak_allocated_bytes=4096)
            loaded, record = evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64)
            self.assertEqual(record["cuda_peak_allocated_bytes"], 4096)
            for endpoint in evaluate.ENDPOINTS:
                np.testing.assert_array_equal(loaded[endpoint], labels)
            row = root / "rows" / "0000" / "row.npz"
            row.write_bytes(row.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d"*64, "e"*64)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_bootstrap_seed_and_shape_are_fixed(self):
        values = np.linspace(0, 1, evaluate.COUNT)
        one = evaluate._bootstrap(values, np.zeros_like(values))
        two = evaluate._bootstrap(values, np.zeros_like(values))
        self.assertEqual(one, two)
        self.assertEqual(one["seed"], 169)
        self.assertEqual(one["replicates"], 10_000)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(values[:-1], np.zeros_like(values[:-1]))

    def test_gate_requires_both_history_credit_and_endpoint_iou(self):
        scores = {}
        for name, fg, fiou, oiou in (
            ("SW0164", .50, .4, .3), ("Source-T", .45, .4, .3), ("Slot", .48, .4, .3),
            ("history_live_conditional", .8, .4, .4), ("history_detached_conditional", .6, .4, .35),
            ("history_live_standalone", .8, .5, .4), ("history_detached_standalone", .6, .45, .35)):
            scores[name] = {"fg_ari": {"mean": fg, "per_image": np.full(evaluate.COUNT, fg)},
                "foreground_iou": {"mean": fiou, "per_image": np.full(evaluate.COUNT, fiou)},
                "matched_object_iou": {"mean": oiou, "per_image": np.full(evaluate.COUNT, oiou)}}
        _paired, gates = evaluate._score_gate(scores)
        self.assertTrue(gates["co_primary_gate_passed"])
        scores["history_live_conditional"]["matched_object_iou"]["mean"] = .1
        _paired, gates = evaluate._score_gate(scores)
        self.assertFalse(gates["co_primary_gate_passed"])


if __name__ == "__main__":
    unittest.main()
