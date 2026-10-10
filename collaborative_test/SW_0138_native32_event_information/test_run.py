from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import unittest
import uuid
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0138_native32_event_information import run


class EventEquationContracts(unittest.TestCase):
    def test_direct_threshold_events_reconstruct_emission_even_at_zero_gate(self):
        membrane = torch.tensor([[[[0.75, 0.25]], [[0.60, 0.90]],
                                  [[0.40, 0.80]], [[0.70, 0.20]]]])
        threshold = torch.full((4, 1), 0.5)
        gates = torch.tensor([[[[0.0, 0.4]], [[0.5, 0.0]],
                               [[1.0, 0.2]], [[0.3, 1.0]]]])
        events = run.native_events_from_equation(membrane, threshold)
        emitted = events * gates
        self.assertTrue(torch.equal(events, torch.tensor([[[[1.0, 0.0]], [[1.0, 1.0]],
                                                           [[0.0, 1.0]], [[1.0, 0.0]]]])))
        self.assertEqual(float(events[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(gates[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(emitted[0, 0, 0, 0]), 0.0)
        self.assertTrue(run.assert_spike_gate_identity(emitted, events, gates))

    def test_spike_gate_identity_rejects_division_recovery_or_changed_emission(self):
        events = torch.tensor([[[[1.0, 0.0]]]])
        gates = torch.tensor([[[[0.0, 0.5]]]])
        emitted = events * gates
        self.assertTrue(run.assert_spike_gate_identity(emitted, events, gates))
        with self.assertRaises(AssertionError):
            run.assert_spike_gate_identity(torch.tensor([[[[0.2, 0.0]]]]), events, gates)

    def test_trajectory_parity_requires_all_fields_exact(self):
        fields = ("component_membrane", "component_spikes", "component_gates",
                  "membrane", "spikes", "theta")
        left = {name: torch.arange(3) for name in fields}
        right = {name: value.clone() for name, value in left.items()}
        self.assertTrue(run.assert_late_rollout_parity(left, right))
        right["theta"][0] += 1
        with self.assertRaises(AssertionError):
            run.assert_late_rollout_parity(left, right)

    def test_actual_and_gate_readouts_use_native_numpy_label_contract(self):
        gates = torch.full((2, 4, 8, 4), 0.25)
        spikes = torch.zeros_like(gates)
        trace = {"component_spikes": spikes, "spikes": spikes.mean(dim=1)}
        actual = np.zeros((2, 32, 32), dtype=np.int64)
        gate = np.ones((2, 32, 32), dtype=np.int64)
        with mock.patch.object(run.sw137.sw136_eval, "_qcc_labels", side_effect=[actual, gate]) as qcc:
            got_actual, got_gate = run.qcc_pair(trace, gates)
        self.assertEqual(got_actual.dtype, np.int64)
        self.assertEqual(got_gate.dtype, np.int64)
        args0 = qcc.call_args_list[0].args[0]
        args1 = qcc.call_args_list[1].args[0]
        self.assertIs(args0, trace)
        self.assertTrue(torch.equal(args1["component_spikes"], gates))
        self.assertTrue(torch.equal(args1["spikes"], gates.mean(dim=1)))

    def test_real_frozen_qcc_helper_accepts_actual_and_gate_trace_arrays(self):
        generator = torch.Generator().manual_seed(138)
        component = torch.randint(0, 2, (1, 4, 4, 514), generator=generator).float()
        gates = torch.rand((1, 4, 4, 514), generator=generator)
        actual, gate = run.qcc_pair({"component_spikes": component,
                                     "spikes": component.mean(dim=1)}, gates)
        self.assertEqual(actual.shape, (1, 32, 32))
        self.assertEqual(gate.shape, (1, 32, 32))
        self.assertEqual(actual.dtype, np.int64)
        self.assertEqual(gate.dtype, np.int64)

    def test_threshold_uses_registered_wrapped_b_expression(self):
        from types import SimpleNamespace
        wrapped = SimpleNamespace(b=torch.tensor([0.0, 0.1, -0.2, 0.3]))
        threshold = run._native_threshold(wrapped, batch=2, nodes=1024)
        expected = (run.BASE_THRESHOLD * torch.exp(wrapped.b))[None, :, None, None]
        expected = expected.expand(2, 4, 1024, 1)
        self.assertTrue(torch.equal(threshold, expected))
        with self.assertRaises(ValueError):
            run._native_threshold(SimpleNamespace(b=torch.zeros(3)), batch=1, nodes=2)

    def test_prediction_validator_binds_manifest_assets_and_first_train_preflight(self):
        root = ROOT / f"sw138_validate_{uuid.uuid4().hex[:8]}"
        output = root / "seed0"
        output.mkdir(parents=True)
        try:
            manifest = root / "manifest.json"
            manifest.write_text('{"source":"97"}\n', encoding="utf-8")
            core_sha = "core97"
            pool = np.arange(4096, dtype=np.int64)
            ids = np.arange(1000, 5096, dtype=np.int64)
            assets = {"encoder_source_sha256": "enc", "feature_preprocessing_sha256": "stats",
                      "rgb_cache_sha256": "rgbcache", "rgb_cache_manifest_sha256": "rgbmanifest"}
            ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            source_record = {"source_core_sha256": core_sha,
                "source_manifest_sha256": run.sha256_file(manifest),
                "source_native32_state_sha256": "state", "encoder_sha256": "enc",
                "feature_preprocessing_sha256": "stats", "rgb_asset_validation": assets,
                "source_training_ids_sha256": ids_sha}
            preflight = {"split": "TRAIN", "image_id": 1000, "pool_row": 0,
                "gamma_shape": [1, 8, 1024], "qcc_shape": [1, 32, 32],
                "event_identity_exact": True, "ground_truth_used": False}
            preflight_record = {**source_record, "preflight": preflight,
                "implementation": {"frozen": "run"},
                "status": "source_event_preflight_complete"}
            (output / "preflight.json").write_text(json.dumps(preflight_record), encoding="utf-8")
            labels = np.zeros((320, 32, 32), dtype=np.int64)
            with (output / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, image_ids=np.arange(1320, 1640),
                    actual_qcc_labels=labels, gate_qcc_labels=labels)
            ref_protocol = {"source_native32_state_sha256": "state",
                "source_manifest_sha256": source_record["source_manifest_sha256"],
                "encoder_sha256": "enc", "feature_preprocessing_sha256": "stats",
                "rgb_assets": assets}
            ref = {"labels": labels, "prediction_sha256": "137pred",
                   "protocol_sha256": "137protocol", "protocol": ref_protocol}
            protocol = {"experiment": "SW0138_native32_event_information", "status": "complete",
                "seed": 0, "image_ids": [1320, 1639], "count": 320,
                "grid_size": [32, 32], "prediction_shape": [320, 32, 32],
                "steps": 1024, "settle": 512, "microbatch_size": 1,
                "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
                "source": source_record, "sw0137_reference": {"prediction_sha256": "137pred",
                    "protocol_sha256": "137protocol", "labels_exactly_reproduced": True},
                "preflight": preflight,
                "preflight_sha256": run.sha256_file(output / "preflight.json"),
                "implementation": {"frozen": "run"},
                "prediction_sha256": run.sha256_file(output / "predictions.npz")}
            (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            source_contract = (Path("core"), manifest, {}, pool, ids, core_sha)
            with mock.patch.object(run.sw130, "source_contract", return_value=source_contract), \
                 mock.patch.object(run.sw137, "_validated_rgb_assets", return_value=assets), \
                 mock.patch.object(run.sw137, "validate_prediction", return_value=ref), \
                 mock.patch.object(run, "implementation_fingerprint", return_value={"frozen": "run"}):
                loaded = run.validate_prediction(output, seed=0, sw137_root=root)
                self.assertEqual(loaded["actual"].shape, (320, 32, 32))
                protocol["source"]["source_manifest_sha256"] = "wrong"
                (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaises(ValueError):
                    run.validate_prediction(output, seed=0, sw137_root=root)
                protocol["source"]["source_manifest_sha256"] = run.sha256_file(manifest)
                protocol["preflight"]["pool_row"] = 1
                (output / "preflight.json").write_text(
                    json.dumps({**preflight_record, "preflight": protocol["preflight"]}), encoding="utf-8")
                protocol["preflight_sha256"] = run.sha256_file(output / "preflight.json")
                (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaises(ValueError):
                    run.validate_prediction(output, seed=0, sw137_root=root)
        finally:
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_registered_trace_validator_rejects_missing_or_wrong_geometry(self):
        with self.assertRaises(ValueError):
            run._validate_trace({"spikes": torch.zeros(1, 1024, 1024)}, batch=1)
        invalid = {key: torch.zeros(shape) for key, shape in {
            "component_membrane": (1, 4, 1024, 1024),
            "component_spikes": (1, 4, 1024, 1024),
            "component_gates": (1, 4, 1024, 1024),
            "membrane": (1, 1024, 1024), "spikes": (1, 1024, 1024),
            "theta": (1, 1024, 1024, 4)}.items()}
        invalid["theta"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            run._validate_trace(invalid, batch=1)


if __name__ == "__main__":
    unittest.main()
