"""Receipt and source-contract checks for the SW0168 disposable preflight."""
from __future__ import annotations

import unittest

from collaborative_test.SW_0168_end_to_end_connectivity import preflight


def _sha(ch):
    return ch * 64


def _valid_report():
    source = {
        "selected_ids": list(range(8)),
        "source_checkpoint_sha256": _sha("a"),
        "source_manifest_sha256": _sha("b"),
        "source_cache_manifest_sha256": _sha("c"),
        "context_cache_manifest_sha256": _sha("d"),
        "source_state_sha256": _sha("e"),
        "hdf5_path": "canonical_train.h5",
        "hdf5_size_bytes": 123,
        "hdf5_mtime_ns": 456,
    }
    traces = {name: _sha("f") for name in
              ("theta", "membrane", "spikes", "component_membrane",
               "component_spikes", "component_gates")}
    rows = [{"index": i, "image_id": i, "gamma_sha256": _sha("1"),
             "trace_sha256": traces, "q_sha256": _sha("2"),
             "actual_a_sha256": _sha("3"),
             "cached_component_spikes_exact": True, "cached_actual_a_exact": True}
            for i in range(8)]
    step = {"update": 1, "mean_loss": 0.5, "q_gradient_norm_mean": 0.1,
            "live_tail_component_spike_gradient_norm_mean": 0.2,
            "preclip_global_gradient_norm": 2.0, "clip_scale": 0.5,
            "elapsed_seconds": 1.0,
            "family_gradient_norms": {"encoder": 1.0, "graph": 1.0,
                                      "native_core": 1.0, "integration": 0.2},
            "family_parameter_delta_l2": {"encoder": 0.01, "graph": 0.01,
                                           "native_core": 0.01, "integration": 0.002}}
    arms = {arm: {"updates": 2, "initial_source_parity_exact": True,
                  "initial_b1_readout_exact": True,
                  "initial_model_state_sha256": source["source_state_sha256"],
                  "steps": [dict(step, update=1), dict(step, update=2)]}
            for arm in preflight.ARMS}
    return source, {
        "experiment": "SW0168_actual_spike_maximin_preflight", "status": "complete_admitted",
        "implementation_fingerprint": {"code.py": _sha("4")}, "source": source,
        "selected_image_ids": list(range(8)), "source_model_state_sha256": source["source_state_sha256"],
        "source_cache_unchanged": True, "source_cache_file_receipt_before": {"m": _sha("5")},
        "source_cache_file_receipt_after": {"m": _sha("5")}, "optimizer_updates": 2,
        "trace_replay_exact": {name: True for name in traces},
        "initial_b4_trace_q_a_parity": {arm: True for arm in preflight.ARMS},
        "source_b4_reference_rows": rows, "arms": arms,
        "source_b1_readout_reference_rows": [
            {"index": i, "image_id": i, "qcc_sha256": _sha("1"),
             "e0_labels_sha256": _sha("2"), "merge_log_sha256": _sha("3"),
             "merge_count": 7, "q_sha256": _sha("4"), "actual_a_sha256": _sha("5")}
            for i in range(8)],
        "validation_data_read": False,
    }


class PreflightReceiptTests(unittest.TestCase):
    def test_actual_producer_shaped_receipt_roundtrips(self):
        source, report = _valid_report()
        self.assertTrue(preflight.validate_report(
            report, expected_source=source, expected_fingerprint=report["implementation_fingerprint"]))

    def test_receipt_rejects_missing_live_tail_credit_and_source_drift(self):
        source, report = _valid_report()
        report["arms"]["constrained_maximin"]["steps"][1][
            "live_tail_component_spike_gradient_norm_mean"] = 0.0
        with self.assertRaisesRegex(ValueError, "credit"):
            preflight.validate_report(report, expected_source=source,
                                     expected_fingerprint=report["implementation_fingerprint"])

    def test_receipt_requires_b1_qcc_e0_and_merge_replay(self):
        source, report = _valid_report()
        report["arms"]["matched_local_edge"]["initial_b1_readout_exact"] = False
        with self.assertRaisesRegex(ValueError, "update record"):
            preflight.validate_report(report, expected_source=source,
                                     expected_fingerprint=report["implementation_fingerprint"])
        source, report = _valid_report()
        report["source_b1_readout_reference_rows"][0]["merge_log_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "B1 Source-T/E0"):
            preflight.validate_report(report, expected_source=source,
                                     expected_fingerprint=report["implementation_fingerprint"])
        source, report = _valid_report()
        report["source"]["source_state_sha256"] = _sha("9")
        with self.assertRaisesRegex(ValueError, "provenance"):
            preflight.validate_report(report, expected_source=source,
                                     expected_fingerprint=report["implementation_fingerprint"])

    def test_source_identity_contains_real_hdf_and_cache_provenance(self):
        values = {"selected_ids": list(range(8)),
                  "source_checkpoint_sha256": _sha("a"),
                  "source_manifest_sha256": _sha("b"),
                  "source_cache_manifest_sha256": _sha("c"),
                  "context_cache_manifest_sha256": _sha("d"),
                  "source_state_sha256": _sha("e"), "hdf5_path": "/canonical/train.h5",
                  "hdf5_size_bytes": 999, "hdf5_mtime_ns": 1000,
                  "rgb8": object(), "cache": object(), "targets": object()}
        identity = preflight._source_identity(values)
        self.assertEqual(identity["hdf5_path"], "/canonical/train.h5")
        self.assertEqual(identity["selected_ids"], list(range(8)))
        self.assertNotIn("rgb8", identity)
        self.assertNotIn("targets", identity)


if __name__ == "__main__":
    unittest.main()
