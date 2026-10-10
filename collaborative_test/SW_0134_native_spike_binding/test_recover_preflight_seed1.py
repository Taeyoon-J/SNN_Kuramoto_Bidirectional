"""Safety checks for the single interrupted seed-1 preflight recovery."""
from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0134_native_spike_binding import recover_preflight_seed1 as recovery


class RecoverySafetyTests(unittest.TestCase):
    def test_only_interrupted_foreign_owner_with_empty_outputs_is_recoverable(self):
        root = Path(tempfile.gettempdir()) / f"sw134_recovery_{uuid.uuid4().hex[:8]}"
        root.mkdir()
        try:
            log = root / "original.log"
            log.write_bytes(b"")
            state = {"status": "scientific_preflight_failed", "supervisor_pid": 43210,
                     "tasks": {
                         "seed0": {"task": {"seed": 0}, "status": "passed"},
                         "seed1": {"task": {"seed": 1},
                                   "status": "interrupted_foreign_gpu_owner",
                                   "foreign_owner_pids": [43211], "child_pid": 43212,
                                   "artifact_valid": False},
                         "seed2": {"task": {"seed": 2}, "status": "passed"}}}
            original_exists = Path.exists

            def proc_absent(path):
                if str(path).startswith("/proc/"):
                    return False
                return original_exists(path)

            with mock.patch.object(Path, "exists", proc_absent):
                self.assertTrue(recovery.source_failure_is_recoverable(
                    state, seed1_log=log, canonical_outputs=[root / "warm.pt", root / "result.json"]))
                state["tasks"]["seed1"]["status"] = "failed"
                self.assertFalse(recovery.source_failure_is_recoverable(
                    state, seed1_log=log, canonical_outputs=[]))
                state["tasks"]["seed1"]["status"] = "interrupted_foreign_gpu_owner"
                state["tasks"]["seed1"]["foreign_owner_pids"] = []
                self.assertFalse(recovery.source_failure_is_recoverable(
                    state, seed1_log=log, canonical_outputs=[]))
                state["tasks"]["seed1"]["foreign_owner_pids"] = [43211]
                warm = root / "warm.pt"
                warm.write_bytes(b"partial output")
                self.assertFalse(recovery.source_failure_is_recoverable(
                    state, seed1_log=log, canonical_outputs=[warm]))
                warm.unlink()
            log.write_bytes(b"preserved nonempty failure output")
            with mock.patch.object(Path, "exists", proc_absent):
                self.assertFalse(recovery.source_failure_is_recoverable(
                    state, seed1_log=log, canonical_outputs=[]))
        finally:
            if log.exists():
                log.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
