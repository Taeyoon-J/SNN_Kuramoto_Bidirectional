from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np  # Load the repository's NumPy/OpenMP runtime before torch.
import torch

from collaborative_test.SW_0174_event_gradient_isolation import audit, event_gradient


class EventGradientTests(unittest.TestCase):
    def test_detach_preserves_values_and_removes_event_gradient(self):
        class Rollout:
            @staticmethod
            def act_fun_adp(value):
                return torch.sigmoid(value)

        module = Rollout()
        original = module.act_fun_adp
        x = torch.tensor([-.5, .25, 1.0], requires_grad=True)
        native = module.act_fun_adp(x)
        native_grad = torch.autograd.grad(native.sum(), x)[0]
        with event_gradient.detach_event_outputs(module) as scope:
            stopped = module.act_fun_adp(x)
            self.assertTrue(torch.equal(native, stopped))
            self.assertFalse(stopped.requires_grad)
            self.assertEqual(scope.calls, 0)  # finalized on context exit
        self.assertIs(module.act_fun_adp, original)
        self.assertEqual(scope.calls, 1)
        self.assertTrue(torch.all(native_grad > 0))

    def test_alias_restored_after_exception(self):
        class Rollout:
            @staticmethod
            def act_fun_adp(value):
                return value + 1

        module = Rollout()
        original = module.act_fun_adp
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with event_gradient.detach_event_outputs(module):
                module.act_fun_adp(torch.ones(1, requires_grad=True))
                raise RuntimeError("sentinel")
        self.assertEqual(module.act_fun_adp, original)

    def test_actual_native_event_autograd_is_removed_without_changing_events(self):
        from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp

        module = SimpleNamespace(act_fun_adp=act_fun_adp)
        x = torch.tensor([-.2, -.01, .01, .2], requires_grad=True)
        native_events = module.act_fun_adp(x)
        native_grad = torch.autograd.grad(native_events.sum(), x)[0]
        self.assertTrue(torch.isfinite(native_grad).all())
        self.assertGreater(float(native_grad.abs().sum()), 0.0)
        with event_gradient.detach_event_outputs(module) as scope:
            stopped_events = module.act_fun_adp(x)
            self.assertTrue(torch.equal(native_events, stopped_events))
            self.assertFalse(stopped_events.requires_grad)
        self.assertIs(module.act_fun_adp, act_fun_adp)
        self.assertEqual(scope.calls, 1)

    def test_paired_forward_restores_rng_and_checks_same_forward(self):
        class Rollout:
            @staticmethod
            def act_fun_adp(value):
                return torch.sigmoid(value)

        module = Rollout()

        def forward():
            x = torch.rand(4, requires_grad=True)
            events = module.act_fun_adp(x)
            q = events[:, None] * events[None, :]
            trace = {"spikes": events[None, None, :],
                     "component_spikes": events[None, None, None, :]}
            return torch.zeros(1, 4), trace, torch.eye(4)[None], torch.ones(1, 4, 4), q[None]

        with event_gradient.detach_event_outputs(module) as scope:
            isolated, native_snap, isolated_snap, checks = audit._paired_forward(
                lambda *_args: forward(), scope, (), ("spikes", "component_spikes"))
            self.assertTrue(checks["all_forward_values_exact"])
            self.assertTrue(torch.equal(native_snap["q"], isolated_snap["q"]))
            self.assertFalse(isolated[4].requires_grad)
        self.assertEqual(scope.calls, 1)

    def test_scoped_audit_wrapper_reports_final_call_count_and_restores_forward(self):
        class Rollout:
            @staticmethod
            def act_fun_adp(value):
                return torch.sigmoid(value)

        rollout = Rollout()
        owner = SimpleNamespace(_forward=lambda: None)
        original_forward = owner._forward
        original_event = rollout.act_fun_adp

        def run(scope):
            def forward():
                x = torch.tensor([-.4, .2], requires_grad=True)
                event = rollout.act_fun_adp(x)
                trace = {"spikes": event[None, None, :],
                         "component_spikes": event[None, None, None, :]}
                return torch.zeros(1, 2), trace, torch.eye(2)[None], torch.ones(1, 2, 4), torch.eye(2)[None]

            owner._forward = lambda: audit._paired_forward(
                lambda *_: forward(), scope, (), ("spikes", "component_spikes"))
            paired = owner._forward()
            return paired[3]["all_forward_values_exact"]

        value, event_calls = audit._run_scoped_intervention(rollout, owner, run)
        self.assertTrue(value)
        self.assertEqual(event_calls, 1)
        self.assertIs(owner._forward, original_forward)
        self.assertIs(rollout.act_fun_adp, original_event)

    def test_snapshot_comparison_covers_all_registered_outputs(self):
        trace_fields = ("spikes", "component_spikes")
        value = (torch.zeros(1, 8),
                 {"spikes": torch.ones(1, 8, 4),
                  "component_spikes": torch.zeros(1, 4, 8, 4)},
                 torch.eye(8)[None], torch.ones(1, 8, 4), torch.eye(8)[None])
        snap = event_gradient.snapshot_forward(value, trace_fields)
        checks = event_gradient.compare_forward_snapshots(snap, snap)
        self.assertTrue(checks["all_forward_values_exact"])
        changed = dict(snap)
        changed["q"] = snap["q"].clone()
        changed["q"][0, 0, 0] = 0.0
        self.assertFalse(event_gradient.compare_forward_snapshots(snap, changed)["all_forward_values_exact"])

    def test_primary_gate_requires_both_route_losses_and_both_readouts(self):
        def source_rows(source_f, native_qcc):
            return [{"native_loss": 1.0,
                     "metrics": {"source_F": {"fg_ari": source_f},
                                 "native_QCC": {"fg_ari": native_qcc}}}
                    for _ in range(2)]

        def build_report(cohort_change=-.1, transfer_change=-.1,
                         source_f=.8, native_qcc=.7):
            trial = {}
            for arm, change in (("cohort", cohort_change), ("transfer", transfer_change)):
                images = [{"metrics": {name: {"fg_ari": score}
                                        for name, score in (("source_F", source_f),
                                                            ("native_QCC", native_qcc))}}
                          for _ in range(2)]
                trial[arm] = [{"scale": 1.0, "actual_native_change": change,
                               "trial_native_mean_loss": .9,
                               "source_native_mean_loss": .9, "images": images}]
            return {"source_baseline": {"cohort": source_rows(.75, .65),
                                         "transfer": source_rows(.75, .65)},
                    "trials": {"full1024": trial}}

        ref = build_report(source_f=.75, native_qcc=.65)
        self.assertTrue(audit._primary_decision(build_report(), ref)["registered_primary_pass"])
        self.assertFalse(audit._primary_decision(build_report(cohort_change=0.0), ref)
                         ["registered_primary_pass"])
        self.assertFalse(audit._primary_decision(build_report(source_f=.74), ref)
                         ["registered_primary_pass"])
        self.assertFalse(audit._primary_decision(build_report(native_qcc=.64), ref)
                         ["registered_primary_pass"])

    def test_reference_guard_rejects_status_fingerprint_and_source_changes(self):
        expected_fp = {"sealed": "fp"}
        source = {"ids": [1, 2]}
        report = {"status": "complete_gradient_audit", "optimizer_updates": 0,
                  "validation_data_read": False, "reserve_data_read": False,
                  "model_state_restored": True, "source_model_unchanged": True,
                  "implementation_fingerprint": expected_fp, "source_proof": source,
                  "same_rng_parity_rows": [{}], "source_baseline": {}}
        records = {"state": {"status": "complete_gradient_audit", "returncode": 0,
                             "report_sha256": audit.REFERENCE_SHA256},
                   "report_file_sha256": audit.REFERENCE_SHA256, "report": report}
        self.assertIs(audit._validate_reference_payload(report, records, expected_fp, source), report)
        for mutate in (
                lambda r, s: s["state"].update(status="failed"),
                lambda r, s: r.update(implementation_fingerprint={"sealed": "changed"}),
                lambda r, s: r.update(source_proof={"ids": [1, 3]})):
            import copy
            bad_report, bad_records = copy.deepcopy(report), copy.deepcopy(records)
            mutate(bad_report, bad_records)
            bad_records["report"] = bad_report
            with self.assertRaises(ValueError):
                audit._validate_reference_payload(bad_report, bad_records, expected_fp, source)


if __name__ == "__main__":
    unittest.main()
