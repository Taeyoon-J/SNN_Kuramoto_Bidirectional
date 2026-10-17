"""CPU contract tests for SW0187 common-prefix schedule and exact forks."""
import numpy as np
import torch
import unittest
from pathlib import Path
import uuid

from collaborative_test.SW_0187_joint_encoder_unique_scaling import train
from collaborative_test.SW_0182_unique1024_actual_descent import schedule as schedule1024
from collaborative_test.SW_0179_qce_seed_replication import train as control
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint


class SharedPrefixTests(unittest.TestCase):
    def test_schedule_has_exact_shared_prefix_and_disjoint_second_branch(self):
        old = control._expected_batches()
        new = schedule1024.expected_batches_1024()
        self.assertEqual(old[:32], new[:32])
        self.assertEqual(train._batches("shared_prefix"), old[:32])
        self.assertEqual(train._batches("control512"), old)
        self.assertEqual(train._batches("candidate1024"), new)
        self.assertTrue(all(i < 512 for row in old for i in row))
        self.assertTrue(all(i >= 512 for row in new[32:] for i in row))
        self.assertEqual(len({i for row in new for i in row}), 1024)

    def test_history_validator_checks_branch_mapping_and_accept_count(self):
        batch = train._batches("shared_prefix")[0]
        before, after = 1.0, 0.9
        accepted, threshold = train.prior._acceptance(before, after)
        row = {"proposal": 1, "proposal_seed": 171000, "image_indices": batch,
            "accepted": accepted, "before_loss": before, "after_loss": after,
            "acceptance_threshold": threshold, "accepted_update_count": 1}
        train._validate_history("shared_prefix", 1, 1, [row], 0)
        bad = dict(row, image_indices=list(train._batches("candidate1024")[32]))
        with self.assertRaises(ValueError):
            train._validate_history("shared_prefix", 1, 1, [bad], 0)
        with self.assertRaises(ValueError):
            train._validate_history("shared_prefix", 1, 0, [row], 0)

    def test_rng_digest_uses_content_not_storage_identity(self):
        state_a = {"torch": torch.arange(8, dtype=torch.uint8),
            "numpy": np.arange(4, dtype=np.int64), "nested": (1, "x")}
        state_b = {"torch": state_a["torch"].clone(),
            "numpy": state_a["numpy"].copy(), "nested": (1, "x")}
        self.assertEqual(train._rng_sha(state_a), train._rng_sha(state_b))
        state_b["torch"][0] = 9
        self.assertNotEqual(train._rng_sha(state_a), train._rng_sha(state_b))

    def test_archived_control_rows_are_reported_not_an_execution_gate(self):
        current = {"image_indices": [0, 1], "accepted": True,
            "start_transaction_sha256": "new-start", "proposal_transaction_sha256": "new-proposal",
            "before_loss": 1.0, "after_loss": 0.8}
        archived = dict(current, start_transaction_sha256="old-start",
                         proposal_transaction_sha256="old-proposal")
        row = train._archived_prefix_comparison(current, archived, 1)
        self.assertFalse(row["start_state_sha_match"])
        self.assertFalse(row["proposal_state_sha_match"])
        self.assertTrue(row["batch_match"])
        self.assertEqual(row["proposal"], 2)

    def test_real_adam_model_fork_restores_parameters_moments_and_rng(self):
        torch.manual_seed(381001)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        loss = model(torch.tensor([[1., 2., 3.]])).square().sum()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        modules = {"wrapped": model}
        snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
        expected_state = tx.transaction_digest(snapshot)
        expected_rng = train._rng_sha(snapshot["rng"])

        # Candidate branch mutates a real model and populated Adam state.
        loss = model(torch.tensor([[-2., .5, 1.]])).square().sum()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.assertNotEqual(tx.transaction_digest(
            tx.capture_transaction(modules, optimizer, offload_to_cpu=True)), expected_state)

        tx.restore_transaction(snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
        fork_a = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
        self.assertEqual(tx.transaction_digest(fork_a), expected_state)
        self.assertEqual(train._rng_sha(fork_a["rng"]), expected_rng)

        # A second branch restores the same copied boundary, then diverges independently.
        branch_b = torch.nn.Linear(3, 2)
        optimizer_b = torch.optim.Adam(branch_b.parameters(), lr=3e-4)
        tx.restore_transaction(snapshot, {"wrapped": branch_b}, optimizer_b,
                               restore_rng=True, clear_grad=True)
        fork_b = tx.capture_transaction({"wrapped": branch_b}, optimizer_b, offload_to_cpu=True)
        self.assertEqual(tx.transaction_digest(fork_b), expected_state)
        self.assertEqual(train._rng_sha(fork_b["rng"]), expected_rng)
        loss = branch_b(torch.tensor([[3., 1., -1.]])).square().sum()
        loss.backward(); optimizer_b.step()
        self.assertNotEqual(tx.transaction_digest(
            tx.capture_transaction({"wrapped": branch_b}, optimizer_b, offload_to_cpu=True)), expected_state)
        self.assertEqual(tx.transaction_digest(
            tx.capture_transaction(modules, optimizer, offload_to_cpu=True)), expected_state)

    def test_real_148_checkpoint_roundtrip_preserves_populated_adam_and_rng(self):
        central = train.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        root = central / f"sw0187-boundary-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(183021)
        wrapped = torch.nn.Linear(2, 4)
        encoder = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam([
            {"params": list(encoder.parameters()), "lr": 3e-4, "name": "context"},
            {"params": list(wrapped.parameters()), "lr": 3e-5, "name": "native_core"}])
        x = torch.randn(4, 3)
        optimizer.zero_grad(set_to_none=True)
        (wrapped(encoder(x)).square().mean()).backward(); optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history = [{"proposal": 1, "loss": 1.0}]
        bindings = {"experiment": "SW0187-test", "seed": 1}
        runtime = checkpoint.runtime_signature("cpu")
        checkpoint.save_boundary(root, completed_updates=1, wrapped=wrapped,
            encoder=encoder, optimizer=optimizer, history=history, bindings=bindings,
            runtime=runtime, device="cpu")
        expected = tx.capture_transaction({"wrapped": wrapped, "encoder": encoder},
            optimizer, offload_to_cpu=True)
        expected_state, expected_rng = tx.transaction_digest(expected), train._rng_sha(expected["rng"])

        torch.manual_seed(9001)
        restored_wrapped = torch.nn.Linear(2, 4)
        restored_encoder = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.Adam([
            {"params": list(restored_encoder.parameters()), "lr": 3e-4, "name": "context"},
            {"params": list(restored_wrapped.parameters()), "lr": 3e-5, "name": "native_core"}])
        loaded = checkpoint.load_latest(root, wrapped=restored_wrapped,
            encoder=restored_encoder, optimizer=restored_optimizer,
            expected_bindings=bindings, expected_runtime=runtime, device="cpu", max_updates=4)
        actual = tx.capture_transaction({"wrapped": restored_wrapped, "encoder": restored_encoder},
            restored_optimizer, offload_to_cpu=True)
        self.assertEqual(loaded["completed_updates"], 1)
        self.assertEqual(loaded["history"], history)
        self.assertEqual(tx.transaction_digest(actual), expected_state)
        self.assertEqual(train._rng_sha(actual["rng"]), expected_rng)

    def test_paths_keep_shared_and_branch_workspaces_separate(self):
        prefix = train._paths("/tmp/s0183", "shared_prefix", 1)
        left = train._paths("/tmp/s0183", "control512", 1)
        right = train._paths("/tmp/s0183", "candidate1024", 1)
        self.assertEqual(prefix[0], left[0].parent / "shared_prefix_seed1")
        self.assertNotEqual(left[0], right[0])
        self.assertNotEqual(left[1], right[1])


class EncoderArtifactGuards(unittest.TestCase):
    def fixture(self):
        tensor_state={'base_encoder.weight':torch.tensor([[.4,.8]])}
        final=train.source._base_encoder_sha_from_state(tensor_state)
        manifest=dict(source_base_trainable=True,source_base_encoder_changed=True,
            source_base_encoder_initial_sha256='a'*64,source_base_encoder_final_sha256=final,
            bindings={'base_encoder_sha256':'a'*64},accepted_updates=2,
            optimizer_step_summary={'source_base':dict(unused_parameter_count=0,updated_parameter_count=1,steps=[2])})
        return manifest,tensor_state

    def test_actual_encoder_tensor_and_committed_Adam_evidence_are_required(self):
        manifest,state=self.fixture()
        train._assert_trained_encoder_manifest(manifest,state)
        altered={name:value+1 for name,value in state.items()}
        with self.assertRaises(ValueError):train._assert_trained_encoder_manifest(manifest,altered)

    def test_frozen_encoder_or_wrong_initial_lineage_cannot_be_scored(self):
        manifest,state=self.fixture()
        for change in ({'source_base_encoder_changed':False},{'source_base_trainable':False},
                       {'bindings':{'base_encoder_sha256':'b'*64}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                train._assert_trained_encoder_manifest({**manifest,**change},state)

    def test_missing_or_mismatched_encoder_Adam_steps_are_rejected(self):
        manifest,state=self.fixture()
        for group in (dict(unused_parameter_count=1,updated_parameter_count=1,steps=[2]),
                      dict(unused_parameter_count=0,updated_parameter_count=0,steps=[2]),
                      dict(unused_parameter_count=0,updated_parameter_count=1,steps=[1])):
            with self.subTest(group=group),self.assertRaises(ValueError):
                train._assert_trained_encoder_manifest({**manifest,'optimizer_step_summary':{'source_base':group}},state)


if __name__ == "__main__":
    unittest.main()
