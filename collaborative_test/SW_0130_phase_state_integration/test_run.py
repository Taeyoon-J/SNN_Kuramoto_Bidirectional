"""Focused SW0130 preflight API and optimizer-membership contracts."""
import sys
import unittest
from pathlib import Path

import numpy as np  # import before torch in the local Windows environment
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'snn_kuramoto_bidirectional'),
                str(ROOT / 'collaborative_test')]

from collaborative_test.SW_0130_phase_state_integration import run
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0130_phase_state_integration.test_model import make_core


class PreflightContracts(unittest.TestCase):
    def test_all_optimizer_parameters_are_in_gradient_union_once(self):
        wrapped = PhaseStateIntegration(make_core(64), 'phase')
        encoder = nn.Linear(5, 3)
        named, groups = run.joint_parameters(wrapped, encoder)
        optimizer_params = [parameter for group in groups for parameter in group['params']]
        optimizer_ids = [id(parameter) for parameter in optimizer_params]
        gradient_ids = [id(parameter) for _, parameter in named]

        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
        self.assertEqual(set(optimizer_ids), set(gradient_ids))
        for name in ('a_d', 'a_m', 'b'):
            parameter = getattr(wrapped, name)
            self.assertEqual(optimizer_ids.count(id(parameter)), 1)
            self.assertIn(id(parameter), gradient_ids)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[1]['params'], [wrapped.a_d, wrapped.a_m, wrapped.b])

    def test_training_gamma_manifest_comes_from_registered_source_module(self):
        self.assertIs(run.GAMMA_TRAIN_MANIFEST, run.source97.GAMMA_TRAIN_MANIFEST)


if __name__ == '__main__':
    unittest.main()
