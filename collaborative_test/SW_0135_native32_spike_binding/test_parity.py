from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0135_native32_spike_binding.parity import (
    _compare_records, _validate_gamma, registered_batch, write_once,
)


class Native32ParityTests(unittest.TestCase):
    def test_registered_batch_keeps_cache_rows_distinct_from_global_ids(self):
        rows, ids = registered_batch(np.asarray([14, 3, 28, 9]),
                                     [1320, 1321, 1322, 1323], 4)
        self.assertEqual(rows.tolist(), [14, 3, 28, 9])
        self.assertEqual(ids, [1320, 1321, 1322, 1323])
        with self.assertRaisesRegex(ValueError, "unique"):
            registered_batch(np.asarray([2, 2]), [1320, 1321], 2)

    def test_gamma_shape_and_finiteness_contract(self):
        _validate_gamma(torch.zeros(2, 8, 1024), 2)
        with self.assertRaisesRegex(ValueError, "gamma"):
            _validate_gamma(torch.zeros(2, 8, 256), 2)
        bad = torch.zeros(1, 8, 1024)
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "gamma"):
            _validate_gamma(bad, 1)

    def test_native_adapter_comparison_reports_exactness_and_detects_mutation(self):
        native = {
            "theta": torch.tensor([[[1.0, 2.0]]]),
            "mean_membrane": torch.tensor([[[.1, .2]]]),
            "mean_spikes": torch.tensor([[[0., 1.]]]),
            "component_membrane": torch.tensor([[[[.1, .2]]]]),
            "component_spikes": torch.tensor([[[[0., 1.]]]]),
            "q": torch.tensor([[1.0]]),
            "losses": {key: torch.tensor(value) for key, value in (
                ("phase_plv", .2), ("phase_loss", .3),
                ("positive_q_loss", .4), ("old_objective", 2.3))},
        }
        exact = _compare_records(native, native)
        self.assertTrue(all(item["exact"] and item["max_abs"] == 0
                            for item in exact.values()))
        changed = {**native, "theta": native["theta"] + 1e-6}
        diff = _compare_records(native, changed)
        self.assertFalse(diff["theta"]["exact"])
        self.assertGreater(diff["theta"]["max_abs"], 0)
        self.assertEqual(diff["theta"]["atol"], 0.0)
        self.assertEqual(diff["theta"]["rtol"], 0.0)
        zero_native = {**native, "q": torch.tensor([[0.]])}
        nonzero_adapter = {**native, "q": torch.tensor([[1e-38]])}
        tiny = _compare_records(zero_native, nonzero_adapter)["q"]
        self.assertFalse(tiny["exact"])
        self.assertAlmostEqual(tiny["zero_reference_max_abs"], 1e-38, delta=1e-45)
        self.assertTrue(np.isfinite(tiny["max_relative_nonzero_reference"]))

    def test_parity_report_is_create_once_and_preserves_existing_record(self):
        folder = HERE / f"tmp_parity_{uuid.uuid4().hex[:8]}"
        folder.mkdir()
        target = folder / "record.json"
        try:
            write_once(target, {"status": "passed", "ground_truth_used": False})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_once(target, {"status": "replacement"})
            self.assertEqual(target.read_bytes(), before)
        finally:
            if target.exists():
                target.unlink()
            folder.rmdir()


if __name__ == "__main__":
    unittest.main()
