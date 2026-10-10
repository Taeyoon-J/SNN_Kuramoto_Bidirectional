from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import uuid
from unittest import mock

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0140_native32_causal_screen import screen
from collaborative_test.SW_0140_native32_causal_screen.causal_rollout import CONDITIONS, TRACE_FIELDS


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.kuramoto = type("Kuramoto", (), {"K": 0.7})()


class FakeWrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeCore()


class FakeFoundation:
    def __init__(self, pool, ids):
        self.wrapped = FakeWrapped()
        self.encoder = nn.Linear(1, 1)
        self.pool_indices = np.asarray(pool, dtype=np.int64)
        self.image_ids = list(ids)
        self.provenance = {
            "source_core_sha256": "source-core",
            "encoder_sha256": "encoder-file",
            "feature_preprocessing_sha256": "stats-file",
            "rgb_asset_validation": {"train_cache_sha256": "rgb-cache"},
            "implementation_fingerprint": {"base_dependency": "sha"},
        }

    def encode(self, rgb):
        return torch.ones((rgb.shape[0], 8, 1024), dtype=torch.float32, device=rgb.device)


class TinyCache:
    shape = (70000, 128, 128, 3)
    dtype = np.dtype(np.uint8)

    def __getitem__(self, item):
        return np.zeros((1, 128, 128, 3), dtype=np.uint8)


def _trace():
    result = {name: torch.zeros((1, 1)) for name in TRACE_FIELDS}
    result.update({name: torch.zeros((1, 1)) for name in (
        "component_carrier", "component_events", "component_natural_threshold_events")})
    result["diagnostics"] = {
        "zero_gate_positions": 0, "zero_gate_hold_positions": 0,
        "event_transitions": 0, "event_occupancy": 0.0,
        "natural_threshold_event_occupancy": 0.0, "emitted_spike_occupancy": 0.0,
    }
    return result


class SW0140ScreenTests(unittest.TestCase):
    def test_source_contract_list_ids_reach_write_once_report(self):
        root = ROOT / f"_sw0140_screen_test_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            checkpoint, manifest_path = root / "core.pt", root / "manifest.json"
            checkpoint.write_bytes(b"checkpoint")
            manifest_path.write_text("{}", encoding="utf-8")
            pool = np.arange(100, 4196, dtype=np.int64)
            ids = list(range(1000, 5096))  # Exact source_contract list[int] API.
            foundation = FakeFoundation(pool, ids)
            source_row = (checkpoint, manifest_path, {"steps": 256}, pool, ids, "source-core")
            fake = _trace()
            with (mock.patch.object(screen.sw130, "source_contract", return_value=source_row),
                  mock.patch.object(screen, "late_rollout",
                                    return_value={name: fake[name] for name in TRACE_FIELDS}),
                  mock.patch.object(screen, "causal_rollout", return_value=fake)):
                output = root / "screen.json"
                result = screen.run_screen(
                    0, "cpu", output,
                    foundation_loader=lambda *_a, **_kw: foundation,
                    train_cache=TinyCache(), _test_only_allow_cpu=True)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["image_ids"], [1000, 1001, 1002, 1003])
            self.assertEqual(loaded["pool_rows"], [100, 101, 102, 103])
            self.assertEqual([row["image_id"] for row in loaded["batches"]], loaded["image_ids"])
            self.assertEqual(result["status"], "screen_complete")
            self.assertEqual(set(CONDITIONS), set(loaded["conditions"]))
            self.assertFalse(foundation.wrapped.training)
            self.assertFalse(foundation.encoder.training)
            self.assertTrue(all(not p.requires_grad for p in foundation.wrapped.parameters()))
            with self.assertRaises(FileExistsError):
                screen.run_screen(0, "cpu", output, foundation_loader=lambda *_a, **_kw: foundation,
                                  train_cache=TinyCache(), _test_only_allow_cpu=True)
        finally:
            for path in (root / "core.pt", root / "manifest.json", root / "screen.json"):
                if path.exists():
                    path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
