from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for import_root in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from collaborative_test.SW_0151_source_preserving_event_history import actual_temporal_probe as driver


class _RowCache:
    shape = (70000, 128, 128, 3)
    dtype = np.dtype(np.uint8)

    def __getitem__(self, row):
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        image[0, 0, 0] = row % 256
        return image


class _EncoderSpy:
    def __init__(self):
        self.calls = 0

    def encode(self, value):
        self.calls += 1
        if value.dtype != torch.uint8 or tuple(value.shape) != (1, 128, 128, 3):
            raise AssertionError("encoder must receive original uint8 NHWC once")
        if not torch.all(value[0, 0, 0] == torch.tensor([17, 255, 255], dtype=torch.uint8)):
            raise AssertionError("input was transformed before the registered encoder")
        return torch.zeros((1, 8, 1024), dtype=torch.float32, device=value.device)


class ActualTemporalProbeTests(unittest.TestCase):
    def test_preflight_requires_strict_validator_and_fixed_positive_scale(self):
        report = {
            "status": "passed_kappa_zero_actual_source_parity",
            "training_admission": False,
            "ground_truth_used": False,
            "optimizer_updates": 0,
            "calibration": {"fixed_per_component_scale": [0.1, 0.2, 0.3, 0.4]},
        }
        path = HERE / "unused_preflight_fixture.json"
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_text", side_effect=lambda *args, **kwargs: json.dumps(report)), \
                patch.object(driver, "sha256_file", return_value="a" * 64):
            loaded, digest = driver.validated_preflight(path, validator=lambda p: True)
            self.assertEqual(loaded, report)
            self.assertEqual(digest, "a" * 64)
            with self.assertRaisesRegex(ValueError, "strict preflight validator"):
                driver.validated_preflight(path, validator=lambda p: False)
            report["calibration"]["fixed_per_component_scale"][2] = 0.0
            with self.assertRaisesRegex(ValueError, "fixed positive"):
                driver.validated_preflight(path, validator=lambda p: True)

    def test_selects_only_first_registered_row_and_rejects_reordered_contract(self):
        ids = list(range(1000, 5096))
        rows = np.arange(1000, 5096, dtype=np.int64)
        image = driver.validate_first_row(
            _RowCache(), expected_id=ids[0], expected_row=int(rows[0]),
            registered_ids=ids, registered_rows=rows)
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image[0, 0, 0]), 232)
        with self.assertRaisesRegex(ValueError, "does not match"):
            driver.validate_first_row(
                _RowCache(), expected_id=ids[1], expected_row=int(rows[0]),
                registered_ids=ids, registered_rows=rows)

    def test_uint8_rgb_is_sent_to_registered_encoder_exactly_once(self):
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        image[0, 0, 0] = 17
        encoder = _EncoderSpy()
        gamma = driver.encode_once(encoder, image, torch.device("cpu"))
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(tuple(gamma.shape), (1, 8, 1024))
        self.assertFalse(gamma.requires_grad)
        with self.assertRaisesRegex(ValueError, "uint8"):
            driver.encode_once(encoder, image.astype(np.float32) / 255.0, torch.device("cpu"))
        self.assertEqual(encoder.calls, 1)

    def test_fingerprint_is_bound_to_current_probe_and_preflight_code(self):
        fingerprint = driver.implementation_fingerprint()
        self.assertEqual(fingerprint["collaborative_test/SW_0151_source_preserving_event_history/actual_temporal_probe.py"],
                         hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest())
        self.assertIn("collaborative_test/SW_0151_source_preserving_event_history/temporal_credit_probe.py",
                      fingerprint)
        self.assertIn("collaborative_test/SW_0151_source_preserving_event_history/preflight.py",
                      fingerprint)


if __name__ == "__main__":
    unittest.main()
