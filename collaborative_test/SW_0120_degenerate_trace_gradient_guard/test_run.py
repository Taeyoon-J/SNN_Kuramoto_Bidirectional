import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0120_degenerate_trace_gradient_guard import run as sw120


class ScalarCriterion:
    plv_target_density = 0.1

    def __call__(self, plv, theta=None):
        return plv.mean(), {}


class SyntheticCore:
    last_component_spikes = None


class SW0120RunnerTests(unittest.TestCase):
    def test_actual_update3_proof_contract_accepts_and_rejects_mutation(self):
        proof = json.loads(sw120.PROOF_PATH.read_text(encoding="utf-8"))
        sw120.validate_proof_record(
            proof, proof["source_core_sha256"], proof["source_manifest_sha256"],
            proof["candidate_history_sha256"], proof["lambda_sha256"])
        bad = json.loads(json.dumps(proof))
        bad["update_3_guarded_diagnostic"]["component_gradient_guard"]["5_positive_Q"][
            "valid_trace_gradient_exactly_matches_legacy"] = False
        with self.assertRaises(AssertionError):
            sw120.validate_proof_record(
                bad, bad["source_core_sha256"], bad["source_manifest_sha256"],
                bad["candidate_history_sha256"], bad["lambda_sha256"])

    def test_refuses_existing_artifact_without_modifying_it(self):
        path = Path(__file__)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            sw120.require_vacant(path, "test output")
        self.assertEqual(path.read_bytes(), before)

    def test_synthetic_guard_objective_matches_legacy_forward_exactly(self):
        torch.manual_seed(1203)
        device = "cpu"
        core = SyntheticCore()
        components = torch.randn(2, 4, 256, 64, requires_grad=True)
        # Include a constant component trace so the synthetic rollout exercises
        # both ordinary and degenerate normalization paths.
        with torch.no_grad():
            components[0, 0, 0] = 0.0
            components[1, 1, 1] = 4.0
        spikes = components.mean(dim=1)
        theta = torch.randn(2, 256, 64, requires_grad=True)
        core_out = spikes * 0.5

        def fake_forward(_core, _gamma, _criterion, _settle, _source, _combine):
            _core.last_component_spikes = components
            return None, spikes, core_out, theta, theta

        labels = torch.zeros(2, 256, dtype=torch.long)
        hard = [torch.ones(256, 1), torch.ones(256, 1)]
        rgb_patches = torch.rand(2, 256, 3)
        with (mock.patch.object(sw117, "_forward_with_plv", side_effect=fake_forward),
              mock.patch.object(sw120, "production_partition", return_value=(labels, hard))):
            legacy = sw117.objective_parts(core, torch.zeros(2, 8, 256), rgb_patches,
                                           ScalarCriterion())
            guarded = sw120.guarded_parts_from_legacy(legacy, rgb_patches, ScalarCriterion())
        for idx in (0, 1, 2, 3, 4, 5, 9, 10, 11, 12):
            self.assertTrue(torch.equal(legacy[idx], guarded[idx]), f"objective field {idx}")
        self.assertTrue(torch.equal(legacy[7], guarded[7]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(legacy[6], guarded[6])))


if __name__ == "__main__":
    unittest.main()
