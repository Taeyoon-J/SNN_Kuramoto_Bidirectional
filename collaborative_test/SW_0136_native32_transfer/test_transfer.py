from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder, RelativeSlotRGBDecoder
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file
from collaborative_test.SW_0136_native32_transfer import evaluate, transfer


class TransferContracts(unittest.TestCase):
    def test_wrapped_state_splits_only_registered_core_and_integration(self):
        core, integ = {"weight": torch.ones(2, 2)}, {k: torch.ones(4) for k in transfer.INTEGRATION_KEYS}
        wrapped = {f"core.{k}": v for k, v in core.items()} | integ
        got_core, got_integ = transfer.split_wrapped_state(wrapped)
        self.assertEqual(set(got_core), set(core))
        self.assertEqual(set(got_integ), transfer.INTEGRATION_KEYS)
        for invalid in (wrapped | {"unregistered": torch.ones(1)},
                        {k: v for k, v in wrapped.items() if k != "b"},
                        wrapped | {"a_d": torch.ones(3)}):
            with self.assertRaises(ValueError):
                transfer.split_wrapped_state(invalid)

    def test_native_wrapper_mapping_must_cover_exact_target_schema(self):
        core = {"theta": torch.zeros(2), "graph": torch.ones(2)}
        integ = {k: torch.full((4,), i, dtype=torch.float32) for i, k in enumerate(sorted(transfer.INTEGRATION_KEYS))}
        template = {f"core.{k}": v.clone() for k, v in core.items()} | {k: v.clone() for k, v in integ.items()}
        result = transfer.compose_native_wrapper_state(core, integ, template)
        self.assertEqual(set(result), set(template))
        torch.testing.assert_close(result["a_m"], integ["a_m"])
        with self.assertRaises(ValueError):
            transfer.compose_native_wrapper_state(core, integ, {"core.theta": torch.zeros(2)})

    def test_strict_head_transfer_preserves_persistent_noise_and_all_weights(self):
        source_binder = NativeSpikeSlotBinder(seed=135)
        source_decoder = RelativeSlotRGBDecoder(seed=106)
        with torch.no_grad():
            source_binder.slot_mu.add_(0.125)
            source_decoder.net[-2].bias.add_(0.05)
        payload = {"binder_state_dict": copy.deepcopy(source_binder.state_dict()),
                   "decoder_state_dict": copy.deepcopy(source_decoder.state_dict())}
        target_binder = NativeSpikeSlotBinder(seed=135)
        target_decoder = RelativeSlotRGBDecoder(seed=106)
        transfer.strict_load_heads(target_binder, target_decoder, payload)
        self.assertTrue(torch.equal(target_binder.initial_noise, source_binder.initial_noise))
        self.assertTrue(torch.equal(target_binder.slot_mu, source_binder.slot_mu))
        self.assertTrue(torch.equal(target_decoder.net[-2].bias, source_decoder.net[-2].bias))
        payload["binder_state_dict"]["initial_noise"] = torch.zeros_like(source_binder.initial_noise)
        altered = NativeSpikeSlotBinder(seed=135)
        transfer.strict_load_heads(altered, RelativeSlotRGBDecoder(seed=106), payload)
        self.assertTrue(torch.equal(altered.initial_noise, torch.zeros_like(altered.initial_noise)))

    def test_preflight_uses_first_registered_training_pool_row(self):
        pool = np.arange(4096, dtype=np.int64) + 17
        ids = np.arange(4096, dtype=np.int64) + 1000
        with mock.patch.object(evaluate.sw130, "source_contract",
                               return_value=(Path("core"), Path("manifest"), {}, pool, ids, "sha")):
            row, image_id = evaluate._first_registered_train_image()
        self.assertEqual((row, image_id), (17, 1000))

    def test_native_trace_contract_checks_every_component_and_aggregate_field(self):
        trace = {
            "component_spikes": torch.zeros(1, 4, 1024, 1024),
            "component_membrane": torch.zeros(1, 4, 1024, 1024),
            "component_gates": torch.ones(1, 4, 1024, 1024),
            "membrane": torch.zeros(1, 1024, 1024),
            "spikes": torch.zeros(1, 1024, 1024),
            "theta": torch.zeros(1, 1024, 1024, 4),
        }
        evaluate._validate_native32_trace(trace, 1)
        trace["component_gates"][0, 3, 1, 8] = float("nan")
        with self.assertRaises(FloatingPointError):
            evaluate._validate_native32_trace(trace, 1)
        trace["component_gates"] = torch.ones(1, 3, 1024, 1024)
        with self.assertRaises(ValueError):
            evaluate._validate_native32_trace(trace, 1)

    def test_prediction_validator_detects_mutated_protocol_and_label_geometry(self):
        root = Path(__import__("tempfile").gettempdir()) / f"sw136_contract_{__import__('uuid').uuid4().hex[:10]}"
        root.mkdir()
        try:
            directory = root / "arm"; directory.mkdir()
            npz = directory / "predictions.npz"
            with npz.open("xb") as stream:
                np.savez_compressed(stream, image_ids=np.arange(1320, 1640),
                                    labels=np.zeros((320, 32, 32), dtype=np.int64),
                                    qcc_labels=np.ones((320, 32, 32), dtype=np.int64))
            protocol = {"experiment": "SW0136_native32_transfer", "status": "complete",
                        "arm": "actual_joint", "seed": 1, "image_ids": [1320, 1639],
                        "count": 320, "grid_size": [32, 32], "prediction_shape": [320, 32, 32],
                        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
                        "transfer": {"arm": "actual_joint", "seed": 1,
                                     "checkpoint_sha256": transfer.EXPECTED_CHECKPOINTS["actual_joint"],
                                     "source_core_sha256": evaluate.sw130.source97.EXPECTED_SOURCE_SHAS[1],
                                     "strict_transfer": True, "ground_truth_used": False,
                                     "optimizer_updates": 0},
                        "implementation": evaluate.implementation_fingerprint(),
                        "prediction_sha256": sha256_file(npz)}
            protocol_path = directory / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            loaded = evaluate.validate_prediction(directory, arm="actual_joint")
            self.assertEqual(loaded["labels"].shape, (320, 32, 32))
            protocol["ground_truth_used_for_prediction"] = True
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.validate_prediction(directory, arm="actual_joint")
        finally:
            for path in root.rglob("*"):
                if path.is_file(): path.unlink()
            (root / "arm").rmdir(); root.rmdir()

    def test_score_refuses_missing_prediction_before_opening_ground_truth(self):
        with mock.patch.object(evaluate, "validate_prediction", side_effect=FileNotFoundError("missing prediction")), \
             mock.patch.object(evaluate.h5py, "File", side_effect=AssertionError("GT opened too early")):
            with self.assertRaises(FileNotFoundError):
                evaluate.score_transfer(output_root=Path("not-used"), dataset=Path("must-not-open"))


if __name__ == "__main__":
    unittest.main()
