import unittest
from pathlib import Path
from unittest.mock import patch

from scheduler_contract import ORDER, validate_markers


class SchedulerContractTest(unittest.TestCase):
    def test_ordered_prefix_and_complete_state(self):
        present = set(ORDER[:3])
        def is_file(path): return path.name in present
        with patch.object(Path, "is_file", is_file):
            root = Path("/fake/state")
            self.assertEqual(validate_markers(root), list(ORDER[:3]))
            with self.assertRaises(ValueError):
                validate_markers(root, require_complete=True)
            present.update(ORDER[3:])
            self.assertEqual(validate_markers(root, require_complete=True), list(ORDER))

    def test_skipped_phase_is_rejected(self):
        present = {"WAITING_FOR_SW0055", "SEED2_STARTED"}
        with patch.object(Path, "is_file", lambda path: path.name in present):
            root = Path("/fake/state")
            with self.assertRaises(ValueError):
                validate_markers(root)


if __name__ == "__main__":
    unittest.main()
