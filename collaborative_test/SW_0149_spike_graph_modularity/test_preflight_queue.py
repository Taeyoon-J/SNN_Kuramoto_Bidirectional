from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4
from unittest import mock

from collaborative_test.SW_0149_spike_graph_modularity import preflight_queue as queue


class PreflightQueueTests(unittest.TestCase):
    def test_single_preflight_dag_and_cpu_parent_does_not_initialize_cuda(self):
        rows = queue.task_plan()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "preflight")
        self.assertEqual(rows[0]["depends_on"], [])
        self.assertEqual(rows[0]["seed"], 1)
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_result_validator_checks_complete_source_evidence(self):
        report = {
            "status": "preflight_complete", "experiment": queue.preflight.NAME,
            "seed": 1, "training_admission": False, "ground_truth_used": False,
            "implementation_sha256": "a" * 64,
            "implementation_fingerprint": {"x": "b" * 64},
            "disposable_optimizer_updates_per_arm": 2,
            "existing_native32_parity": {"sha256": "c" * 64,
                                  "exact_arm_horizons": [{}, {}, {}, {}]},
            "first64_modularity_credit": {"images": 64,
                "positive_M_offdiagonal_entries": 1, "negative_M_offdiagonal_entries": 1,
                "per_b4_batch": [{} for _ in range(16)]},
            "arms": [{"arm": name,
                "fresh_source_parity": {"actual_spike_q_exact": True,
                                        "source_graph_exact": True},
                "updates": [{}, {}]} for name in ("encoder_graph", "joint")],
        }
        tmp = Path(gettempdir()) / f"sw149_valid_result_{uuid4().hex}.json"
        try:
            tmp.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(queue.preflight, "implementation_sha", return_value="a" * 64), \
                 mock.patch.object(queue.preflight, "fingerprint", return_value={"x": "b" * 64}):
                self.assertTrue(queue.valid_result(tmp))
                report["ground_truth_used"] = True
                tmp.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.valid_result(tmp))
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
