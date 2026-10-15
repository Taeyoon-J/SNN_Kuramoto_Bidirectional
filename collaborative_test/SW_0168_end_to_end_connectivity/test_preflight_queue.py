"""Focused queue receipt binding checks for SW0168."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0168_end_to_end_connectivity import preflight_queue


class QueueReceiptTests(unittest.TestCase):
    def test_child_deadline_only_stops_owned_child_after_deadline(self):
        child = object()
        stopped = []
        owner = SimpleNamespace(_stop_owned=lambda value: stopped.append(value))
        preflight_queue._check_deadline(child, 10.0, owner, 10.0)
        self.assertEqual(stopped, [])
        with self.assertRaisesRegex(TimeoutError, "1800 seconds"):
            preflight_queue._check_deadline(child, 10.0, owner, 10.01)
        self.assertEqual(stopped, [child])

    def test_queue_validator_binds_both_queue_and_current_preflight_code(self):
        current = {"preflight.py": "a" * 64}
        report = {"queue_fingerprint_sha256": "b" * 64,
                  "implementation_fingerprint": current}
        delegated = []
        prep = SimpleNamespace(
            implementation_fingerprint=lambda: current,
            validate_report=lambda value, **kwargs: delegated.append(kwargs) or True)
        with mock.patch.object(preflight_queue, "_load_preflight", return_value=prep):
            self.assertTrue(preflight_queue.validate_report(
                report, source_proof={"selected_ids": list(range(8))},
                fingerprint="b" * 64))
            self.assertEqual(delegated[0]["expected_fingerprint"], current)
            report["implementation_fingerprint"] = {"preflight.py": "c" * 64}
            with self.assertRaisesRegex(ValueError, "implementation fingerprint"):
                preflight_queue.validate_report(
                    report, source_proof={"selected_ids": list(range(8))},
                    fingerprint="b" * 64)
            report["implementation_fingerprint"] = current
            with self.assertRaisesRegex(ValueError, "queue fingerprint"):
                preflight_queue.validate_report(
                    report, source_proof={"selected_ids": list(range(8))},
                    fingerprint="d" * 64)


if __name__ == "__main__":
    unittest.main()
