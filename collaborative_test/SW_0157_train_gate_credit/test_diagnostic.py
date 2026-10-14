import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0155_supervised_spike_relations import objective, train
from collaborative_test.SW_0157_train_gate_credit import diagnostic


class GateCreditDiagnosticTests(unittest.TestCase):
    def test_upstream_payload_normalizes_independently_built_numpy_rows(self):
        image_ids = list(range(10000, 14096))
        assets = {"train_cache_sha256": "a" * 64}

        def source_contract():
            return {"image_ids": list(image_ids), "rows": np.arange(4096, dtype=np.int64),
                    "source_sha256": "b" * 64, "manifest_path": Path("source-manifest.json")}

        t_ref = {"arm": "T", "ids": np.asarray(diagnostic.sw143.contract.IMAGE_IDS),
                 "prediction_sha256": "c" * 64, "protocol_sha256": "d" * 64}
        stat = SimpleNamespace(st_size=123, st_mtime_ns=456)
        with mock.patch.object(diagnostic, "_source_and_assets",
                               side_effect=lambda: (source_contract(), assets, stat)), \
             mock.patch.object(diagnostic.sw143, "validate_arm_prediction", return_value=t_ref), \
             mock.patch.object(diagnostic, "sha256_file", return_value="e" * 64), \
             mock.patch.object(train, "TRAIN_ROOT", Path("fixture-train-root")), \
             mock.patch.object(Path, "is_file", autospec=True, return_value=True), \
             mock.patch.object(Path, "read_text", autospec=True,
                               return_value=json.dumps({"implementation_fingerprint": {"test": "fp"}})), \
             mock.patch.object(diagnostic.artifactvalidator, "validate_training_artifact", return_value=True):
            first = diagnostic.validate_upstream()
            second = diagnostic.validate_upstream()
        self.assertIsInstance(first["source_rows"], list)
        self.assertEqual(first["source_rows"], list(range(4096)))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_event_margin_occupancy_gate_support_and_spike_identity(self):
        # Each component shares the same g; E is reconstructed from M and b.
        base = torch.tensor([.05, .07, .08, .04]).reshape(1, 1, 1, 4)
        membrane = base.expand(1, 4, 2, 4).clone()
        gates = torch.tensor([0., .5, 1., 1.]).reshape(1, 1, 1, 4).expand_as(membrane).clone()
        events = membrane > .06  # same fixed .06 native threshold
        spikes = events.to(gates.dtype) * gates
        original = membrane.clone()
        report = diagnostic.event_diagnostics(membrane, spikes, gates, torch.zeros(4), settle=2)
        self.assertTrue(report["shared_gate_exact"])
        self.assertTrue(report["emitted_spike_equals_event_times_gate_exact"])
        self.assertAlmostEqual(report["full_window_per_component"][0]["event_on_fraction"], .5)
        self.assertAlmostEqual(report["full_window_per_component"][0][
            "event_on_fraction_when_gate_positive"], 2.0 / 3.0)
        self.assertAlmostEqual(report["per_component"][0]["event_on_fraction"], .5)
        self.assertAlmostEqual(report["per_component"][0][
            "event_on_fraction_when_gate_positive"], .5)
        self.assertEqual(report["per_component"][0]["gate_positive_support_count"], 4)
        self.assertTrue(torch.equal(membrane, original))

    def test_parameter_groups_accept_source_style_standalone_gamma_parameter(self):
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_generator = torch.nn.Linear(2, 2)
                self.gamma_to_drive = torch.nn.Linear(2, 2)
                self.gamma_channel_proj = torch.nn.Linear(2, 2)
                self.gamma_phase_gain = torch.nn.Parameter(torch.ones(4))
                self.kuramoto = torch.nn.Linear(2, 2)
                self.dendric_layer = torch.nn.Linear(2, 2)
                self.membrane_layer = torch.nn.Linear(2, 2)

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = Core()
                self.a_d = torch.nn.Parameter(torch.ones(4))
                self.a_m = torch.nn.Parameter(torch.ones(4))
                self.b = torch.nn.Parameter(torch.ones(4))

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_encoder = torch.nn.Linear(2, 2)

        foundation = type("Foundation", (), {})()
        foundation.encoder = Encoder()
        foundation.patcher = torch.nn.Linear(2, 2)
        foundation.wrapped = Wrapped()
        groups, frozen = diagnostic._named_parameter_groups(foundation)
        phase_gain = foundation.wrapped.core.gamma_phase_gain
        self.assertTrue(any(parameter is phase_gain for parameter in groups["gamma_drive"]))
        self.assertEqual(len({id(p) for values in groups.values() for p in values}),
                         sum(map(len, groups.values())))
        self.assertTrue(all(not p.requires_grad for p in frozen))

    def test_wrong_shared_gate_or_event_spike_product_is_visible(self):
        mem = torch.ones(1, 4, 2, 3)
        gates = torch.ones_like(mem)
        gates[:, 1, :, 1] = 0
        spikes = (mem > .06).float() * gates
        record = diagnostic.event_diagnostics(mem, spikes, gates, torch.zeros(4), settle=1)
        self.assertFalse(record["shared_gate_exact"])
        spikes = spikes.clone(); spikes[:, 0, 0, 0] = 0
        record = diagnostic.event_diagnostics(mem, spikes, gates, torch.zeros(4), settle=1)
        self.assertFalse(record["emitted_spike_equals_event_times_gate_exact"])

    def test_actual_sw0155_objective_has_finite_component_spike_gradient(self):
        torch.manual_seed(157)
        components = torch.rand(2, 4, 4, 10, requires_grad=True)
        trace = {"component_spikes": components, "spikes": components.mean(dim=1)}
        labels = torch.tensor([[0, 1, 1, 2], [0, 0, 3, 4]], dtype=torch.int64)
        loss, details = objective.supervised_relation_loss(trace, labels, settle=2)
        grad, = torch.autograd.grad(loss, components)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0)
        self.assertEqual(tuple(details["q"].shape), (2, 4, 4))

    def test_parameter_family_gradient_summary_reports_unused_without_claim(self):
        a = torch.nn.Parameter(torch.tensor(2.0))
        b = torch.nn.Parameter(torch.tensor(3.0))
        x = torch.tensor(1.0, requires_grad=True)
        loss = (a * x + b).square()
        groups = {name: [] for name in diagnostic.PARAMETER_GROUPS}
        groups["spatial_residual"] = [a]
        groups["graph"] = [b]
        report, x_report = diagnostic._gradient_summary(loss, groups, extra=x)
        self.assertTrue(report["spatial_residual"]["finite"])
        self.assertEqual(report["spatial_residual"]["l2_norm"], 10.0)
        self.assertEqual(report["kuramoto"]["parameter_count"], 0)
        self.assertEqual(report["kuramoto"]["l2_norm"], 0.0)
        self.assertEqual(x_report["l2_norm"], 20.0)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def _valid_report(self):
        ids = list(range(5000, 9096))
        arms = {}
        lineage = {}
        for arm in train.ARMS:
            arms[arm] = {"manifest_sha256": arm[0].lower() * 64,
                         "checkpoint_sha256": arm[0].upper() * 64,
                         "manifest": {"implementation_fingerprint": {"train": arm}}}
            lineage[arm] = {"manifest_sha256": arms[arm]["manifest_sha256"],
                            "checkpoint_sha256": arms[arm]["checkpoint_sha256"],
                            "implementation_fingerprint": arms[arm]["manifest"]["implementation_fingerprint"]}
        upstream = {"source_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "source_training_ids_sha256": "c" * 64, "source_training_ids": ids,
            "source_rows": list(range(4096)), "rgb_asset_validation": {"train_cache_sha256": "d" * 64},
            "sw0143_T_prediction_sha256": "e" * 64, "sw0143_T_protocol_sha256": "f" * 64,
            "sw0155_arms": arms, "hdf5_path": "/registered/train.h5",
            "hdf5_size_bytes": 1000, "hdf5_mtime_ns": 1234}
        model_rows = {}
        expected_ids = [ids[pos] for pos in diagnostic.POSITIONS]
        for model in diagnostic.MODEL_ORDER:
            batches = []
            for index, positions in enumerate(diagnostic.BATCH_POSITIONS):
                gradients = {key: {"parameter_count": 1, "parameters_with_gradient": 1,
                    "unused_parameter_count": 0, "finite": True, "l2_norm": .2}
                    for key in diagnostic.PARAMETER_GROUPS}
                component_records = [{"margin_quantiles": {str(q): .1 for q in (.01,.1,.5,.9,.99)},
                    "margin_mean": .1, "margin_std": .2, "margin_rms": .3,
                    "event_on_fraction": .5, "event_transition_fraction": .2,
                    "gate_positive_support_count": 10,
                    "event_on_fraction_when_gate_positive": .5,
                    "weighted_spike_mean": .4} for _ in range(4)]
                events = {"shared_gate_exact": True,
                    "emitted_spike_equals_event_times_gate_exact": True,
                    "settled_window_start": diagnostic.SETTLE,
                    "per_component": component_records,
                    "full_window_per_component": component_records}
                batches.append({"source_positions": list(positions),
                    "image_ids": expected_ids[index*4:(index+1)*4],
                    "rng_seed": diagnostic.ROLLOUT_SEED + index,
                    "mean_actual_spike_relation_loss": .4,
                    "mean_membrane_proxy_relation_loss": .5,
                    "actual_spike_parameter_gradients": gradients,
                    "membrane_proxy_parameter_gradients": gradients,
                    "actual_component_spikes_input_gradient": {"finite": True, "l2_norm": .2},
                    "component_membrane_proxy_input_gradient": {"finite": True, "l2_norm": .3},
                    "event_dynamics": events})
            model_rows[model] = {"model_state_sha256_before": "1"*64,
                "model_state_sha256_after": "1"*64, "model_state_unchanged": True,
                "optimizer_updates": 0, "frozen_rgb_base_sha256": "2"*64,
                "batches": batches}
        report = {"experiment": diagnostic.EXPERIMENT, "status": "complete", "seed": 1,
            "implementation_fingerprint": {"code": "frozen"},
            "upstream": {key: value for key, value in upstream.items()
                         if key not in ("source_training_ids", "source_rows", "sw0155_arms")},
            "sw0155_arm_lineage": lineage, "source_order_sha256": upstream["source_training_ids_sha256"],
            "source_training_ids": expected_ids, "source_positions": list(diagnostic.POSITIONS),
            "batch_positions": [list(x) for x in diagnostic.BATCH_POSITIONS],
            "models": model_rows, "rng": {"matched_seed_per_batch": [157001, 157002],
                "caller_rng_restored": True}, "steps": 1024, "settle": 512,
            "live_tail_steps": 64, "optimizer_updates": 0, "ground_truth_used": True,
            "ground_truth_scope": "the eight registered source TRAIN masks only",
            "validation_or_test_ground_truth_used": False}
        return report, upstream

    def test_strict_report_validator_binds_models_rows_gt_scope_and_full_lineage(self):
        report, upstream = self._valid_report()
        payload = [json.dumps(report)]
        with mock.patch.object(Path, "read_text", autospec=True,
                               side_effect=lambda _path, **_kwargs: payload[0]):
            self.assertTrue(diagnostic.validate_report("ignored", expected_fingerprint={"code":"frozen"},
                                                        upstream=upstream))
            bad = json.loads(payload[0]); bad["ground_truth_scope"] = "validation masks"
            payload[0] = json.dumps(bad)
            self.assertFalse(diagnostic.validate_report("ignored", expected_fingerprint={"code":"frozen"},
                                                         upstream=upstream))
            bad = json.loads(json.dumps(report)); bad["models"]["SW0155_joint"][
                "model_state_sha256_after"] = "3"*64
            payload[0] = json.dumps(bad)
            self.assertFalse(diagnostic.validate_report("ignored", expected_fingerprint={"code":"frozen"},
                                                         upstream=upstream))
            bad = json.loads(json.dumps(report)); bad["models"]["source97"]["batches"][0][
                "actual_component_spikes_input_gradient"]["finite"] = False
            payload[0] = json.dumps(bad)
            self.assertFalse(diagnostic.validate_report("ignored", expected_fingerprint={"code":"frozen"},
                                                         upstream=upstream))
            bad = json.loads(json.dumps(report)); bad["models"]["source97"]["batches"][0][
                "event_dynamics"]["per_component"][0]["event_on_fraction"] = 1.2
            payload[0] = json.dumps(bad)
            self.assertFalse(diagnostic.validate_report("ignored", expected_fingerprint={"code":"frozen"},
                                                         upstream=upstream))


if __name__ == "__main__":
    unittest.main()
