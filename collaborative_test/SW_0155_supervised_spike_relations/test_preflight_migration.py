from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from collaborative_test.SW_0155_supervised_spike_relations import preflight_migration, train


class SW0155PreflightMigrationTests(unittest.TestCase):
    def test_actual_preflight_migrates_only_the_four_serialization_literals(self):
        archived_candidates = (
            train.HERE / "results_archive" / "prior_preflight_20261013" / "train.py",
            train.HERE / "results_archive" / "serialization_review_20261013" / "old_train.py",
        )
        archived_old = next((path for path in archived_candidates if path.is_file()), None)
        if archived_old is not None:
            old_source = archived_old.read_bytes()
        else:
            # The source archive is included in deployments. Keep a git fallback
            # for minimal local checkouts, but never infer a different old version.
            relative = "collaborative_test/SW_0155_supervised_spike_relations/train.py"
            old_source = subprocess.check_output(
                ["git", "-C", str(train.ROOT), "show", f"HEAD:{relative}"])
        new_source = (train.HERE / "train.py").read_bytes()
        old_report_path = train.HERE / "results_archive" / "preflight_completed_20261013" / "preflight_seed1.json"
        old_report = json.loads(old_report_path.read_text(encoding="utf-8-sig"))
        migrated = preflight_migration.create_migrated_report(old_report, old_source, new_source)
        self.assertTrue(preflight_migration.validate_migrated_report(old_report, migrated))
        altered = json.loads(json.dumps(migrated))
        altered["implementation_fingerprint"][preflight_migration.TRAIN_KEY] = "0" * 64
        self.assertFalse(preflight_migration.validate_migrated_report(old_report, altered))
        self.assertEqual(migrated["serialization_migration"]["ast_proof"]["changed_lines"],
                         [361, 362, 376, 377])


if __name__ == "__main__":
    unittest.main()
