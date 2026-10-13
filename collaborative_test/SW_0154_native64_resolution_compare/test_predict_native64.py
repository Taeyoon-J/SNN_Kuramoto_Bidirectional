from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0154_native64_resolution_compare import predict_native64 as predict
from collaborative_test.SW_0154_native64_resolution_compare.native64_model import GRID, NODES


class Native64PredictionTests(unittest.TestCase):
    def test_resume_record_binds_source_image_seeds_and_readout_horizon(self):
        record = {"seed": 1, "image_id": 1320, "source_checkpoint_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64, "rgb_asset_validation": {"rgb": "fixed"},
            "implementation_fingerprint": {"run.py": "c" * 64},
            "input_rgb_sha256": "d" * 64, "model_initialization_seed": predict.MODEL_INIT_SEED,
            "saved_native32_parity_sha256": "e" * 64,
            "inference_seed": predict.INFERENCE_SEED, "steps": 1024, "settle": 512,
            "live_tail_steps": 0, "grid_size": [64, 64],
            "ground_truth_used_for_prediction": False}
        labels = np.ones((GRID, GRID), dtype=np.int64)

        class Archive:
            files = ["labels", "record_json"]
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __getitem__(self, key):
                return labels if key == "labels" else np.asarray(json.dumps(record))

        kwargs = {"seed": 1, "image_id": 1320, "source_sha": "a" * 64,
            "manifest_sha": "b" * 64, "assets": {"rgb": "fixed"},
            "fingerprint": {"run.py": "c" * 64}, "rgb_sha": "d" * 64,
            "parity_sha": "e" * 64}
        with mock.patch.object(predict.np, "load", return_value=Archive()):
            got, got_record = predict._read_resume_row(Path("ledger.npz"), **kwargs)
        self.assertTrue(np.array_equal(got, labels))
        self.assertEqual(got_record["image_id"], 1320)
        for field, value in (("steps", 1000), ("inference_seed", 77),
                             ("saved_native32_parity_sha256", "f" * 64)):
            record[field] = value
            with mock.patch.object(predict.np, "load", return_value=Archive()):
                with self.assertRaisesRegex(ValueError, "lineage/schema"):
                    predict._read_resume_row(Path("ledger.npz"), **kwargs)
            record[field] = {"steps": 1024, "inference_seed": predict.INFERENCE_SEED,
                             "saved_native32_parity_sha256": "e" * 64}[field]

    def test_prediction_uses_registered_native64_actual_spike_readout(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        seen = {}
        trace = {key: torch.ones(1) for key in (
            "component_spikes", "component_membrane", "component_gates",
            "spikes", "membrane", "theta")}
        labels = torch.full((1, GRID, GRID), 3, dtype=torch.int64)

        def classify(*args, **kwargs):
            seen.update(kwargs)
            return object()

        with mock.patch.object(predict, "late_rollout", return_value=trace) as rollout, \
             mock.patch.object(predict, "spike_synchrony_components", side_effect=classify), \
             mock.patch.object(predict, "spatial_components_to_patch_labels", return_value=labels):
            result = predict._predict_one(None,
                lambda x: torch.zeros(1, 8, 126, 126),
                lambda x: torch.ones(1, 8, NODES),
                torch.zeros(1, 8, 1, 1), torch.ones(1, 8, 1, 1), 3.0,
                image, torch.device("cpu"))
        self.assertEqual(result.shape, (64, 64))
        self.assertTrue(np.all(result == 3))
        self.assertEqual(seen["synchrony_threshold"], .50)
        self.assertEqual(seen["min_group_size"], 32)
        self.assertEqual(seen["settle"], 512)
        self.assertEqual(seen["spatial_grid_size"], 64)
        self.assertEqual(rollout.call_args.args[0], None)
        self.assertEqual(rollout.call_args.kwargs,
            {"total_steps": 1024, "live_tail_steps": 0})


if __name__ == "__main__":
    unittest.main()
