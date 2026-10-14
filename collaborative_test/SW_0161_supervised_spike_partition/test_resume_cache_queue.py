from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest import mock

import numpy as np
from collaborative_test.SW_0161_supervised_spike_partition import resume_cache_queue as retry


class CacheRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(retry.HERE) / ("retry_queue_test_" + uuid.uuid4().hex)
        self.old = self.root / "old"
        self.new = self.root / "new"
        self.cache = self.root / "cache"
        self.old.mkdir(parents=True)
        (self.old / "cache_attempt01.log").write_text(
            "AttributeError: module '...objective' has no attribute 'SLOTS'\n", encoding="utf-8")
        state = {"experiment": "SW0161_source_cache_and_head_preflight_queue",
                 "status": "failed_cache_stage", "supervisor_pid": retry.EXPECTED_OLD_SUPERVISOR,
                 "stages": {"cache": {"status": "failed_child_or_artifact_validation",
                                       "child_pid": retry.EXPECTED_OLD_CHILD,
                                       "child_returncode": 1}}}
        (self.old / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def tearDown(self):
        if self.root.exists():
            resolved = self.root.resolve()
            if resolved.parent != Path(retry.HERE).resolve():
                raise AssertionError("refusing cleanup outside owned fixture")
            shutil.rmtree(resolved)

    def test_only_reviewed_failed_attempt_with_no_outputs_is_retryable(self):
        with mock.patch.object(retry, "_pid_dead", return_value=True):
            report = retry.validate_retry_prerequisites(self.old / "state.json", self.cache, self.new)
        self.assertEqual(report["missing_symbol"], "SLOTS")
        self.assertTrue(report["cache_root_absent"])
        self.assertFalse(self.new.exists())

    def test_unexpected_error_live_owner_or_existing_cache_is_fail_closed(self):
        log = self.old / "cache_attempt01.log"
        with mock.patch.object(retry, "_pid_dead", return_value=True):
            log.write_text("RuntimeError: numerical failure", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the reviewed"):
                retry.validate_retry_prerequisites(self.old / "state.json", self.cache, self.new)
            log.write_text("AttributeError: objective has no attribute SLOTS", encoding="utf-8")
            with mock.patch.object(retry, "_pid_dead", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "still live"):
                    retry.validate_retry_prerequisites(self.old / "state.json", self.cache, self.new)
            self.cache.mkdir()
            with self.assertRaisesRegex(FileExistsError, "preserve unexpected"):
                retry.validate_retry_prerequisites(self.old / "state.json", self.cache, self.new)

    def test_retry_module_fingerprint_binds_changed_cache_and_new_wrapper(self):
        fp = retry.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0161_supervised_spike_partition/resume_cache_queue.py", fp)
        self.assertIn("collaborative_test/SW_0161_supervised_spike_partition/cache.py", fp)
        self.assertEqual(fp["collaborative_test/SW_0161_supervised_spike_partition/cache.py"],
                         retry.sha256_file(retry.HERE / "cache.py"))


if __name__ == "__main__":
    unittest.main()
