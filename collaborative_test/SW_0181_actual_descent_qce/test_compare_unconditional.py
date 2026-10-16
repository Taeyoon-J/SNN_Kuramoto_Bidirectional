"""Pure-CPU contracts for supplemental same-seed unconditional QCE comparison."""
from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0181_actual_descent_qce import compare_unconditional as comparison


def report(experiment, offset):
    groups = {}
    for endpoint in comparison.ENDPOINTS:
        metrics = {}
        for metric_index, metric in enumerate(comparison.METRICS):
            rows = {str(seed): np.full(comparison.COUNT,
                offset + seed * 0.01 + metric_index * 0.001, dtype=np.float64).tolist()
                for seed in comparison.SEEDS}
            metrics[metric] = {"per_seed_per_image": rows,
                "per_seed_mean": {seed: float(np.mean(values)) for seed, values in rows.items()}}
        groups[f"candidate_{endpoint}"] = metrics
    return {"experiment": experiment, "status": "complete",
        "seed_scope": list(comparison.SEEDS), "scores": groups}


class UnconditionalComparisonTests(unittest.TestCase):
    def test_shared_image_three_seed_comparison_recomputes_paired_delta(self):
        candidate = report("SW0181_actual_descent_qce_three_seed_score", 0.7)
        control = report("SW0179_qce_seed_replication_three_seed_score", 0.6)
        result = comparison.compare_reports(candidate, control,
            control_sha256=comparison.SW0179_REPORT_SHA256)
        for endpoint in comparison.ENDPOINTS:
            for metric in comparison.METRICS:
                row = result["paired_results"][endpoint][metric]
                self.assertAlmostEqual(row["mean"], 0.1, places=12)
                self.assertEqual(row["bootstrap_seed"], 178)
                self.assertEqual(row["replicates"], 10_000)
        self.assertFalse(result["validation_data_read"])

    def test_wrong_control_digest_or_missing_seed_is_rejected(self):
        candidate = report("SW0181_actual_descent_qce_three_seed_score", 0.7)
        control = report("SW0179_qce_seed_replication_three_seed_score", 0.6)
        with self.assertRaises(ValueError):
            comparison.compare_reports(candidate, control, control_sha256="0" * 64)
        del control["scores"]["candidate_standalone"]["fg_ari"]["per_seed_per_image"]["2"]
        with self.assertRaises(ValueError):
            comparison.compare_reports(candidate, control,
                control_sha256=comparison.SW0179_REPORT_SHA256)


if __name__ == "__main__": unittest.main()
