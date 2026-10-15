"""Regression for preserving a completed GPU receipt before a post-run CPU check."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0168_end_to_end_connectivity import preflight_queue_v4 as queue

import torch


class QueueV4Tests(unittest.TestCase):
    def test_raw_gpu_report_survives_failing_fresh_cpu_revalidation(self):
        temp_root = Path(__file__).resolve().parents[1] / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = temp_root / f"sw168-v4-queue-{__import__('os').getpid()}"
        temp.mkdir(exist_ok=False)
        try:
            report_path = temp / "preflight.json"
            proof = {"selected_ids": [101, 102], "source_checkpoint_sha256": "a" * 64}
            changed = {**proof, "source_checkpoint_sha256": "b" * 64}
            prep = SimpleNamespace(
                run_two_update_preflight=lambda prepared, device: {"training_admission": True},
                implementation_fingerprint=lambda: {"preflight": "c" * 64},
                _atomic_json=queue._load_preflight()._atomic_json,
                validate_report=lambda *args, **kwargs: self.fail("must fail on CPU proof mismatch first"),
            )
            result = SimpleNamespace(stdout=json.dumps(changed) + "\n", stderr="")
            import os
            old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            try:
                with (mock.patch.object(queue, "_load_owner_preserving_visibility", return_value=None),
                      mock.patch.object(queue, "_source_admission", return_value=(prep, {}, proof)),
                      mock.patch.object(queue, "fingerprint_sha256", return_value="d" * 64),
                      mock.patch.object(torch.cuda, "is_initialized", return_value=False),
                      mock.patch.object(torch.cuda, "device_count", return_value=1),
                      mock.patch.object(torch.cuda, "set_device"),
                      mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
                      mock.patch.object(torch, "set_num_threads"),
                      mock.patch.object(queue.subprocess, "run", return_value=result) as run_cpu):
                    with self.assertRaisesRegex(RuntimeError, "changed during preflight"):
                        queue._worker(1, report_path)
                self.assertEqual(run_cpu.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            finally:
                if old_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
            self.assertTrue(report_path.is_file())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["source"], proof)
            self.assertEqual(saved["queue_fingerprint_sha256"], "d" * 64)
            report_path.unlink()
        finally:
            temp.rmdir()


if __name__ == "__main__":
    unittest.main()
