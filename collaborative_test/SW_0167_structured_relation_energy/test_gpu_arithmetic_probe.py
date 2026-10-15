from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("gpu_arithmetic_probe.py")
SPEC = importlib.util.spec_from_file_location("sw0167_gpu_arithmetic_probe_tested", MODULE_PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ArithmeticDeltaTests(unittest.TestCase):
    def test_delta_separates_float_difference_from_threshold_edge_changes(self):
        cached = np.asarray([[0.500001, 0.499999], [0.2, 0.8]], dtype=np.float32)
        actual = np.asarray([[0.499999, 0.499998], [0.2, 0.8]], dtype=np.float32)
        result = probe._delta(cached, actual)
        self.assertGreater(result["max_abs"], 0.0)
        self.assertEqual(result["exact_entry_differences"], 2)
        self.assertEqual(result["threshold_edge_disagreements_at_0_5"], 1)

    def test_delta_requires_same_geometry(self):
        with self.assertRaises(ValueError):
            probe._delta(np.zeros((2, 2), np.float32), np.zeros((3, 3), np.float32))

    def test_source_proof_normalizes_actual_audit_selected_ids_schema(self):
        # _validate_upstream_and_rows emits selected_ids, not ordered_image_ids.
        prepared = {
            "selected_ids": [68304, 49032, 70041, 7829, 1000, 1001, 1002, 1003],
            "source_checkpoint_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "source_cache_manifest_sha256": "c" * 64,
            "context_cache_manifest_sha256": "d" * 64,
            "source_state_sha256": "e" * 64,
            "gain": 7.623262882232666,
            "gain_receipt_sha256": "f" * 64,
        }
        receipt = probe._source_proof(prepared)
        self.assertEqual(receipt["ordered_image_ids"], prepared["selected_ids"])
        self.assertEqual(probe._source_proof(prepared), receipt)
        self.assertNotIn("selected_ids", receipt)

    def test_cached_b4_q_is_an_explicit_feature_and_energy_input(self):
        cached = np.asarray([[0.2, 0.5], [0.5, 0.2]], dtype=np.float32)
        recomputed_b4 = cached.copy()
        recomputed_b1 = np.asarray([[0.2, 0.4], [0.4, 0.2]], dtype=np.float32)
        modes = probe._q_modes(cached, recomputed_b4, recomputed_b1)
        self.assertEqual([name for name, _ in modes], ["cached_gpu_b4", "gpu_b4", "gpu_b1"])
        self.assertIs(modes[0][1], cached)
        self.assertIs(modes[1][1], recomputed_b4)
        self.assertIs(modes[2][1], recomputed_b1)


if __name__ == "__main__":
    unittest.main()
