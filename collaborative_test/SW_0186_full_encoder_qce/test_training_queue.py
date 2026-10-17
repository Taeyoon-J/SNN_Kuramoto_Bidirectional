"""CPU contracts for SW0186 source-preflight and exclusive-worker queue wiring."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0186_full_encoder_qce import training_queue as queue
from collaborative_test.SW_0186_full_encoder_qce import train


class TrainingQueueTests(unittest.TestCase):
    def test_worker_command_binds_seed_physical_gpu_and_module(self):
        expected = [queue.sys.executable, "-m",
            "collaborative_test.SW_0186_full_encoder_qce.training_queue",
            "--worker", "--source-seed", "2", "--physical-gpu", "3"]
        self.assertEqual(queue.worker_command(2, 3), expected)
        for seed, gpu in ((3, 1), (1, 4)):
            with self.assertRaises(ValueError):
                queue.worker_command(seed, gpu)

    def test_cli_help_and_parent_dispatch_use_registered_seed_choices(self):
        with self.assertRaises(SystemExit) as help_exit:
            queue.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        with mock.patch.object(queue, "run", return_value={"status": "waiting"}) as run:
            result = queue.main(["--source-seed", "0"])
        self.assertEqual(result, 0)
        run.assert_called_once_with(0)

    def test_cpu_admission_keeps_dataset_proof_distinct_from_model_source(self):
        inputs = {"dataset_manifest_sha256": "a" * 64,
            "dataset_arrays_sha256": "b" * 64,
            "dataset_producer_proof": {"source_seed": 1, "scope": "data only"},
            "ordered_train_ids": list(range(512))}
        proof = {"seed": 2, "source_state_sha256": "c" * 64,
            "initial_model_state_sha256": "d" * 64, "cuda_initialized": False}
        trainer = SimpleNamespace(prepare_training_inputs_cpu=lambda: inputs,
            prepare_fresh_training_state_cpu=lambda seed, prepared: {"proof": proof})
        receipt = {"report_sha256": "e" * 64, "queue_fingerprint_sha256": "f" * 64}
        got_inputs, identity = queue._cpu_admission(2, trainer, receipt)
        self.assertIs(got_inputs, inputs)
        self.assertEqual(identity["source_factory_proof"], proof)
        self.assertEqual(identity["source_preflight_report_sha256"], "e" * 64)
        self.assertEqual(identity["source_preflight_queue_fingerprint_sha256"], "f" * 64)
        self.assertNotIn("source_state_sha256", identity["dataset_producer_proof"])

    def test_queue_fingerprint_binds_own_source_and_training_bytes(self):
        identity = {"source_factory_proof": {"seed": 1,
            "initial_model_state_sha256": "a" * 64}, "ordered_train_ids": list(range(512))}
        payload, digest = queue._queue_fingerprint(train, 1, identity)
        self.assertEqual(payload["experiment"], "SW0186_training_queue")
        self.assertEqual(payload["seed"], 1)
        self.assertEqual(payload["input_identity"], identity)
        self.assertEqual(len(payload["queue_sha256"]), 64)
        self.assertEqual(len(payload["trainer_sha256"]), 64)
        self.assertEqual(len(payload["trainer_fingerprint_sha256"]), 64)
        self.assertEqual(len(digest), 64)

    def test_state_writer_persists_terminal_counts_without_duplicate_seed_fields(self):
        with mock.patch.object(queue, "RUNTIME", Path("unused")):
            temp_root = queue.ROOT / "collaborative_testing" / "temporary"
            temp_root.mkdir(parents=True, exist_ok=True)
            state_path = temp_root / f"sw186-queue-{os.getpid()}-state.json"
            try:
                with mock.patch.object(queue, "_state_path", return_value=state_path):
                    state = queue._write_state(1, "complete_rejected",
                        proposal_cursor=2, accepted_updates=1, validation_data_read=False)
                self.assertEqual(state["seed"], 1)
                self.assertEqual(state["status"], "complete_rejected")
                self.assertEqual(state["proposal_cursor"], 2)
                self.assertEqual(json.loads(state_path.read_text())["accepted_updates"], 1)
            finally:
                if state_path.is_file():
                    state_path.unlink()
                if temp_root.is_dir() and not any(temp_root.iterdir()):
                    temp_root.rmdir()

    def test_pending_preflight_waits_until_a_real_complete_receipt(self):
        class Wait:
            def __init__(self): self.n = 0
            def is_set(self): return False
            def wait(self, _seconds): self.n += 1
        fake_queue = SimpleNamespace(STATE=Path("state.json"), REPORT=Path("report.json"))
        trainer = SimpleNamespace(_preflight_receipt=lambda _seed: {"status": "complete",
            "report_sha256": "c" * 64})
        stopper = Wait()
        def exists(self):
            return stopper.n >= 1
        def read_text(self, **_kwargs):
            return json.dumps({"status": "complete"})
        with mock.patch.object(queue, "_import_preserving_visibility", return_value=fake_queue), \
             mock.patch.object(Path, "is_file", exists), \
             mock.patch.object(Path, "read_text", read_text), \
             mock.patch.object(queue, "_sha", return_value="c" * 64):
            result = queue._wait_preflight(trainer, stopper, poll_seconds=0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(stopper.n, 1)

    def test_scientific_rejection_is_terminal_and_validates_dict_fingerprint(self):
        temp_root = queue.ROOT / "collaborative_testing" / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        work = temp_root / f"sw186-reject-{os.getpid()}"
        work.mkdir(exist_ok=False)
        try:
            trainer = SimpleNamespace(EXPERIMENT="SW0186_full_encoder_qce",
                implementation_fingerprint=lambda: {"experiment": "SW0186_full_encoder_qce",
                    "files": {"train.py": "a" * 64}})
            payload = {"experiment": trainer.EXPERIMENT, "status": "scientific_rejection",
                "arm": "product_q_ce", "seed": 2, "proposal_cursor": 1,
                "accepted_updates": 1, "bindings": {"training_fingerprint":
                    trainer.implementation_fingerprint()}, "rejection_reasons": ["family credit missing"],
                "validation_data_read": False}
            (work / "credit_rejection.json").write_text(json.dumps(payload), encoding="utf-8")
            result = queue._validate_credit_rejection(trainer, work, 2)
            self.assertEqual(result["status"], "complete_rejected")
            self.assertEqual(result["proposal_cursor"], 1)
            self.assertEqual(len(result["rejection_sha256"]), 64)
            payload["bindings"]["training_fingerprint"] = "a" * 64
            (work / "credit_rejection.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                queue._validate_credit_rejection(trainer, work, 2)
        finally:
            for item in work.iterdir():
                item.unlink()
            work.rmdir()
            if temp_root.is_dir() and not any(temp_root.iterdir()):
                temp_root.rmdir()


if __name__ == "__main__":
    unittest.main()
