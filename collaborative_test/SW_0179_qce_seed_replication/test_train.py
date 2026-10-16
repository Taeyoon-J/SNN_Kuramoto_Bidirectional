import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import torch
import unittest

from collaborative_test.SW_0179_qce_seed_replication import train
from collaborative_test.SW_0178_temporal_class_competition.class_competition import (
    leave_one_out_class_loss,
)


class SW0179TrainingContracts(unittest.TestCase):
    def test_repository_root_and_seed_specific_foundation_sources_exist(self):
        self.assertTrue((train.ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py").is_file())
        self.assertTrue((train.ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py").is_file())
        self.assertTrue((train.ROOT / "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md").is_file())
        self.assertEqual(train.ROOT, train.HERE.parents[1])

    def test_order_is_two_consecutive_fixed_numpy_permutations(self):
        rng = np.random.RandomState(171001)
        expected = []
        for _ in range(2):
            order = rng.permutation(512).astype(np.int64)
            expected.extend(order[i:i + 16].tolist() for i in range(0, 512, 16))
        self.assertEqual(train._expected_batches(), expected)
        self.assertEqual(len(expected), 64)
        self.assertEqual(len({index for batch in expected[:32] for index in batch}), 512)
        self.assertEqual(len({index for batch in expected[32:] for index in batch}), 512)

    def test_seed_mapping_binds_fresh_source_and_context_without_reusing_seed1_proof(self):
        self.assertEqual(train.CONTEXT_SEED_BASE + 0, 171000)
        self.assertEqual(train.CONTEXT_SEED_BASE + 1, 171001)
        self.assertEqual(train.CONTEXT_SEED_BASE + 2, 171002)
        self.assertEqual([train.MODEL_SEED_BASE + seed + 0 for seed in (0, 1, 2)],
                         [171000, 171001, 171002])
        source = {"seed": 2, "source_state_sha256": "a" * 64,
            "source_checkpoint_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64,
            "source_manifest_path": "seed2.json", "source_steps": 256,
            "source_core_path": "seed2.pt", "encoder_sha256": "d" * 64,
            "feature_preprocessing_sha256": "e" * 64}
        inputs = {"dataset_producer_proof": {"producer_source_seed": 1,
                    "scope": "shared dataset provenance only"},
            "dataset_manifest_sha256": "f" * 64, "dataset_arrays_sha256": "1" * 64,
            "ordered_train_ids": list(range(512))}
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        families = {name: [(name + ".weight", parameter.clone().detach().requires_grad_())]
                    for name in ("context_encoder", "graph", "native_core", "integration")}
        binding = train._bindings("product_q_ce", 2, inputs, source, "2" * 64,
                                  "3" * 64, families)
        self.assertEqual(binding["seed"], 2)
        self.assertEqual(binding["source_binding"]["seed"], 2)
        self.assertEqual(binding["context_seed"], 171002)
        self.assertEqual(binding["dataset_producer_proof"]["producer_source_seed"], 1)
        self.assertNotIn("source_state_sha256", binding["dataset_producer_proof"])
        self.assertFalse(binding["unsupervised_leaderboard_eligible"])
        self.assertTrue(binding["mask_supervised"])

    def test_vectorized_product_q_loss_matches_frozen_reference_value_and_gradient(self):
        generator = torch.Generator().manual_seed(17921)
        raw = torch.randn((2, 9, 9), generator=generator, dtype=torch.float64)
        q = ((raw + raw.transpose(1, 2)) * .5).requires_grad_(True)
        labels = torch.tensor([[0, 0, 0, 3, 3, 8, 8, 8, 15],
                               [4, 4, 4, 4, 4, 4, 4, 4, 4]], dtype=torch.int64)
        vectorized = torch.stack([
            train._loss_for_image("product_q_ce", q[i], None, labels[i])
            for i in range(q.shape[0])]).mean()
        reference = leave_one_out_class_loss(q, labels, temperature=0.1)
        self.assertTrue(torch.allclose(vectorized, reference, atol=1e-11, rtol=1e-11))
        gv, = torch.autograd.grad(vectorized, q, retain_graph=True)
        gr, = torch.autograd.grad(reference, q)
        self.assertTrue(torch.allclose(gv, gr, atol=1e-10, rtol=1e-10))
        self.assertTrue(bool(torch.isfinite(gv).all()))

    def test_first_two_update_gate_requires_emitted_s_and_all_four_families(self):
        family_names = ("context_encoder", "graph", "native_core", "integration")
        valid = {"update": 1, "loss_to_emitted_component_spikes_gradient_l2": .2,
            "gradient_norm_before_clip": .3,
            "family_gradient_norms": {name: .01 for name in family_names},
            "family_parameter_delta_l2": {name: .001 for name in family_names}}
        self.assertEqual(train._credit_failures(valid), [])
        bad = dict(valid)
        bad["family_parameter_delta_l2"] = dict(valid["family_parameter_delta_l2"], graph=0.0)
        failures = train._credit_failures(bad)
        self.assertTrue(any("graph" in value for value in failures))
        bad = dict(valid, loss_to_emitted_component_spikes_gradient_l2=0.0)
        self.assertTrue(any("emitted-S" in value for value in train._credit_failures(bad)))


if __name__ == "__main__":
    unittest.main()
