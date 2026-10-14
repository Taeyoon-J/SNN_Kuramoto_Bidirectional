"""CPU regression tests for SW0159 calibration, source loader, and native rollout."""
import numpy as np  # Load NumPy before Torch on Windows' bundled OpenMP runtimes.
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import torch

from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
from collaborative_test.SW_0110_xy_graph_route import run as source97
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0159_one_shot_event_regime import calibration
from collaborative_test.SW_0159_one_shot_event_regime import preflight
from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp


class OneShotCalibrationTests(unittest.TestCase):
    def test_pinned_margin_rms_maps_to_physical_threshold_and_existing_b(self):
        rms = calibration.source_margin_rms()
        self.assertEqual(rms, calibration.SOURCE_MARGIN_RMS)
        result = calibration.one_shot_bias(rms)
        self.assertEqual(result["thresholds"], tuple(0.5 * x for x in rms))
        for threshold, bias in zip(result["thresholds"], result["bias"]):
            self.assertAlmostEqual(0.06 * np.exp(bias), threshold, places=12)
        self.assertAlmostEqual(result["bias"][0], 3.115, places=2)

    def test_unregistered_calibration_file_rejected(self):
        with self.assertRaises(ValueError):
            calibration.source_margin_rms(Path(__file__))
        for invalid in ((1, 2, 3), (1, 2, 3, 0), (1, 2, np.nan, 4)):
            with self.assertRaises(ValueError):
                calibration.one_shot_bias(invalid)

    def test_source_contract_requires_actual_ids_list_and_rows_array_types(self):
        source = {"image_ids": list(range(4096)), "rows": np.arange(4096, dtype=np.int64)}
        self.assertTrue(preflight.validate_source_contract_types(source))
        with self.assertRaises(ValueError):
            preflight.validate_source_contract_types({"image_ids": tuple(range(4096)), "rows": source["rows"]})
        with self.assertRaises(ValueError):
            preflight.validate_source_contract_types({"image_ids": source["image_ids"], "rows": list(range(4096))})

    def test_complete_scientific_rejection_is_valid_but_not_admitted(self):
        trace_fields = {name: True for name in preflight.sw143.TRACE_FIELDS}
        record = {"experiment": preflight.EXPERIMENT, "status": "complete", "seed": 1,
            "optimizer_updates": 2, "source_positions": list(preflight.POSITIONS),
            "preflight_logical_batch": 4, "preflight_microbatch": 1,
            "image_ids": [10, 11, 12, 13, 20, 21, 22, 23], "ground_truth_used": True,
            "ground_truth_scope": "the eight registered source TRAIN masks only",
            "validation_or_test_ground_truth_used": False, "training_admission": False,
            "admission_reasons": ["initial_candidate_events_match_source", "no_gate_positive_settled_event_transitions",
                "update_1_existing_b_gradient_zero", "update_1_membrane_gradient_zero", "update_1_existing_b_unchanged",
                "update_1_membrane_parameters_unchanged",
                "update_1_selected_family_unchanged", "update_2_existing_b_gradient_zero",
                "update_2_membrane_gradient_zero", "update_2_existing_b_unchanged",
                "update_2_membrane_parameters_unchanged", "update_2_selected_family_unchanged"],
            "control_source_T_parity": {"six_trace_exact": True, "production_Q_exact": True,
                "production_QCC_exact": True, "trace_fields": trace_fields},
            "initialization": {"fraction": .5, "margin_rms": [2., 2., 2., 2.],
                "thresholds": [1., 1., 1., 1.], "bias": [float(np.log(1/.06))]*4},
            "thresholds_after_init": [1., 1., 1., 1.],
            "candidate_event_changed_from_source": False,
            "candidate_temporal_event_transitions_gate_positive": 0,
            "candidate_initial_production_Q_mean": .5,
            "candidate_initial_production_Q_sha256": "a"*64,
            "candidate_initial_QCC_label_count_sum": 8,
            "source_event_regime_by_batch": [{}, {}],
            "candidate_event_regime_by_batch": [
                {"settled_event_on_fraction": 1., "settled_gate_positive_count": 10,
                 "settled_event_differs_from_source": False, "settled_on_off_transitions_gate_positive": 0,
                 "per_component_settled_on_fraction": [1.]*4},
                {"settled_event_on_fraction": 1., "settled_gate_positive_count": 10,
                 "settled_event_differs_from_source": False, "settled_on_off_transitions_gate_positive": 0,
                 "per_component_settled_on_fraction": [1.]*4}],
            "candidate_updates": []}
        for index in (1, 2):
            record["candidate_updates"].append({"update": index, "mean_loss": 1.,
                "family_gradient_norms": {"native_core": 0., "integration_tensors": 0.},
                "b_gradient_norm": 0., "membrane_gradient_norm": 0., "b_parameter_change_l2": 0.,
                "membrane_parameter_change_l2": 0.,
                "family_parameter_change_l2": {"native_core": 0., "integration_tensors": 0.}})
        self.assertTrue(preflight.validate_report_record(record))
        record["training_admission"] = True
        self.assertFalse(preflight.validate_report_record(record))

    def test_admission_requires_actual_membrane_parameter_change(self):
        base = {"experiment": preflight.EXPERIMENT, "status": "complete", "seed": 1,
            "optimizer_updates": 2, "source_positions": list(preflight.POSITIONS),
            "preflight_logical_batch": 4, "preflight_microbatch": 1,
            "image_ids": [10, 11, 12, 13, 20, 21, 22, 23], "ground_truth_used": True,
            "ground_truth_scope": "the eight registered source TRAIN masks only",
            "validation_or_test_ground_truth_used": False, "training_admission": False,
            "admission_reasons": ["update_1_membrane_parameters_unchanged",
                                  "update_2_membrane_parameters_unchanged"],
            "control_source_T_parity": {"six_trace_exact": True, "production_Q_exact": True,
                "production_QCC_exact": True,
                "trace_fields": {name: True for name in preflight.sw143.TRACE_FIELDS}},
            "initialization": {"fraction": .5, "margin_rms": [2.]*4,
                "thresholds": [1.]*4, "bias": [float(np.log(1/.06))]*4},
            "thresholds_after_init": [1.]*4,
            "candidate_event_changed_from_source": True,
            "candidate_temporal_event_transitions_gate_positive": 2,
            "candidate_initial_production_Q_mean": .5,
            "candidate_initial_production_Q_sha256": "a"*64,
            "candidate_initial_QCC_label_count_sum": 8,
            "source_event_regime_by_batch": [{}, {}],
            "candidate_event_regime_by_batch": [
                {"settled_event_on_fraction": .5, "settled_gate_positive_count": 10,
                 "settled_event_differs_from_source": True, "settled_on_off_transitions_gate_positive": 1,
                 "per_component_settled_on_fraction": [.5]*4},
                {"settled_event_on_fraction": .5, "settled_gate_positive_count": 10,
                 "settled_event_differs_from_source": True, "settled_on_off_transitions_gate_positive": 1,
                 "per_component_settled_on_fraction": [.5]*4}],
            "candidate_updates": []}
        for index in (1, 2):
            base["candidate_updates"].append({"update": index, "mean_loss": 1.,
                "family_gradient_norms": {"native_core": .2, "integration_tensors": .1},
                "b_gradient_norm": .3, "membrane_gradient_norm": .4,
                "b_parameter_change_l2": .01, "membrane_parameter_change_l2": 0.,
                "family_parameter_change_l2": {"native_core": .02, "integration_tensors": .03}})
        self.assertTrue(preflight.validate_report_record(base))
        for row in base["candidate_updates"]:
            row["membrane_parameter_change_l2"] = .01
        base["training_admission"] = True
        base["admission_reasons"] = []
        self.assertTrue(preflight.validate_report_record(base))
        base["training_admission"] = False
        self.assertFalse(preflight.validate_report_record(base))


class NativeRolloutTests(unittest.TestCase):
    @staticmethod
    def tiny_wrapped():
        hp = hparams("raw")
        hp.num_regions = 4
        hp.num_time_steps = 4
        hp.spike_per_component = True
        hp.graph_mode = "learned"
        hp.graph_top_k = 2
        hp.spike_spatial_grid_size = 2
        hp.geodesic_steps = 0
        torch.manual_seed(8031)
        core = source97.S2NetCore(hp.validate(), device="cpu")
        return PhaseStateIntegration(core, "phase")

    def test_original_native_rollout_reset_gate_and_existing_b_gradient(self):
        wrapped = self.tiny_wrapped()
        gamma = torch.linspace(-1, 1, 32).reshape(1, 8, 4)
        dense_previous = []
        original = wrapped._dense_dendrite

        def observe(layer, carrier, previous_spike):
            dense_previous.append(previous_spike.detach().clone())
            return original(layer, carrier, previous_spike)

        wrapped._dense_dendrite = observe
        trace = late_rollout(wrapped, gamma, total_steps=3, live_tail_steps=3)
        self.assertEqual(trace["component_spikes"].shape, (1, 4, 4, 3))
        self.assertTrue(torch.equal(dense_previous[0], torch.zeros_like(dense_previous[0])))
        self.assertTrue(torch.equal(dense_previous[1], trace["component_spikes"][0, :, :, 0]))
        native_events = act_fun_adp(trace["component_membrane"] - 0.06)
        expected = native_events * trace["component_gates"]
        self.assertTrue(torch.equal(trace["component_spikes"], expected))

        loss = trace["component_spikes"].sum()
        loss.backward()
        self.assertIsNotNone(wrapped.b.grad)
        self.assertTrue(torch.isfinite(wrapped.b.grad).all())
        self.assertGreater(float(wrapped.b.grad.norm()), 0.0)

    def test_one_shot_bias_receives_two_native_surrogate_updates(self):
        wrapped = self.tiny_wrapped()
        initial = calibration.one_shot_bias(calibration.source_margin_rms())
        calibration.install_bias(wrapped, initial["bias"])
        gamma = torch.linspace(-1, 1, 32).reshape(1, 8, 4)
        optimizer = torch.optim.Adam([wrapped.b], lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            trace = late_rollout(wrapped, gamma, total_steps=3, live_tail_steps=3)
            loss = trace["component_spikes"].sum()
            loss.backward()
            self.assertTrue(torch.isfinite(wrapped.b.grad).all())
            self.assertGreater(float(wrapped.b.grad.norm()), 0.0)
            before = wrapped.b.detach().clone()
            optimizer.step()
            self.assertGreater(float((wrapped.b.detach() - before).norm()), 0.0)
            del trace, loss

    def test_actual_sw155_sorted_loader_restores_list_ids_and_numpy_rows(self):
        # Mirrors source_contract's concrete types: ordered IDs are a Python
        # list, pool rows are a NumPy integer array, cache rows are uint8 RGB.
        ids = [41, 8, 29]
        row_ids = np.asarray([410, 80, 290], dtype=np.int64)
        cache = np.arange(3 * 128 * 128 * 3, dtype=np.uint8).reshape(3, 128, 128, 3)
        sorted_order = np.argsort(ids)
        sorted_ids = np.asarray(ids)[sorted_order]
        images = cache[sorted_order]
        masks = np.stack([np.full((128, 128), i, np.uint8) for i in sorted_order])
        aligned_images, aligned_masks = sw155.restore_source_order(ids, sorted_ids, images, masks, cache)
        self.assertTrue(np.array_equal(aligned_images.numpy(), cache))
        self.assertEqual(aligned_masks.shape, (3, 128, 128))
        self.assertEqual(row_ids.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
