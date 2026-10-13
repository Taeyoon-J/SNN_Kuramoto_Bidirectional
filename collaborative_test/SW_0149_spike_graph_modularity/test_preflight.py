from __future__ import annotations

import copy
import json
import unittest

from collaborative_test.SW_0149_spike_graph_modularity import preflight


class PreflightContractTests(unittest.TestCase):
    def test_registered_saved_sw0135_parity_receipt_is_accepted(self):
        self.assertEqual(
            preflight.PARITY_PATH,
            preflight.ROOT / "collaborative_test" / "SW_0135_native32_spike_binding"
            / "results_archive" / "native32_zero_adapter_parity_seed1_b4.json",
        )
        row = json.loads(preflight.PARITY_PATH.read_text(encoding="utf-8"))
        checked = preflight.validate_existing_parity_record(
            row,
            source_core_sha256=row["source_core_sha256"],
            source_manifest_sha256=row["source_manifest_sha256"],
        )
        self.assertEqual(len(checked["exact_arm_horizons"]), 4)
        self.assertTrue(all(exact for _, exact in checked["exact_arm_horizons"]))

    def test_saved_parity_rejects_changed_source_binding_and_nonexact_trace(self):
        row = json.loads(preflight.PARITY_PATH.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "stale or invalid"):
            preflight.validate_existing_parity_record(
                row, source_core_sha256="0" * 64,
                source_manifest_sha256=row["source_manifest_sha256"],
            )
        altered = copy.deepcopy(row)
        altered["arms"]["phase"][0]["exact_parity"] = False
        with self.assertRaisesRegex(ValueError, "exact native32 horizons"):
            preflight.validate_existing_parity_record(
                altered, source_core_sha256=row["source_core_sha256"],
                source_manifest_sha256=row["source_manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
