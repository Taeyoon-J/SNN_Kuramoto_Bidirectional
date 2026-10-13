from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0154_native64_resolution_compare import (
    predict_native64_recovery as recovery, recovery_queue as queue,
)


class Native64RecoveryQueueTests(unittest.TestCase):
    def test_plan_and_argv_are_limited_to_missing_seeds0_and2(self):
        rows = queue.task_plan()
        self.assertEqual([row["seed"] for row in rows], [0, 2])
        self.assertEqual([row["task_id"] for row in rows], ["recover_seed0", "recover_seed2"])
        argv = queue.command(rows[1], physical_gpu=3)
        self.assertIn("predict_native64_recovery", argv[2])
        self.assertEqual(argv[argv.index("--seed") + 1], "2")
        with self.assertRaises(ValueError):
            queue.command({**rows[1], "seed": 1}, physical_gpu=3)

    def test_recovery_validator_scopes_legacy_seed_and_restores_it(self):
        protocol = {"seed_specific_parity_recovery": {
                "status": "seed_specific_legacy_validator_override",
                "recovery_entrypoint_sha256": recovery.implementation_sha256(),
                "frozen_predictor_sha256": recovery.EXPECTED_PREDICTOR_SHA256,
                "legacy_validator_module": "SW0150_spike_spatial_mixture.preflight",
                "legacy_default_seed": 1, "scoped_seed": 2,
                "source_checkpoint_sha256": recovery.EXPECTED_SOURCE_SHA256[2]}}
        before = queue.legacy_preflight.SEED
        seen = []

        def fake_validate(seed, **_kwargs):
            seen.append((seed, queue.legacy_preflight.SEED))
            return {"seed": seed}

        with mock.patch.object(Path, "read_text", return_value=json.dumps(protocol)), \
             mock.patch.object(queue, "FROZEN_VALIDATE_PREDICTION", side_effect=fake_validate):
            self.assertEqual(queue.validate_prediction(2), {"seed": 2})
        self.assertEqual(seen, [(2, 2)])
        self.assertEqual(queue.legacy_preflight.SEED, before)

    def test_wrong_recovery_seed_or_repair_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            queue.validate_prediction(1)
        with mock.patch.object(Path, "read_text", return_value=json.dumps({
                "seed_specific_parity_recovery": {"scoped_seed": 0}})):
            with self.assertRaises(ValueError):
                queue.validate_prediction(0)

    def test_unstable_reservation_releases_then_reacquires_before_child(self):
        row = queue.task_plan()[0]
        instance = object.__new__(queue.SeedRecoveryQueue)
        instance.stop_event = queue.threading.Event()
        instance.guard = queue.threading.RLock()
        instance.state = {"status": "queued", "tasks": {row["task_id"]: {"task": row}}}
        instance._verify = mock.Mock()
        leases = [mock.Mock(), mock.Mock()]
        for index, lease in enumerate(leases):
            lease.fileno.return_value = 70 + index
        child = mock.Mock()
        child.pid, child.poll.return_value = 8181, 0
        child.wait.return_value = 0
        fp = SimpleNamespace(LOCK_UN=8, flock=mock.Mock())
        with mock.patch.object(queue, "fcntl", fp), \
             mock.patch.object(queue, "_atomic_json"), \
             mock.patch.object(queue, "implementation_fingerprint"), \
             mock.patch.object(queue.predictor, "implementation_fingerprint"), \
             mock.patch.object(queue, "valid_prediction", side_effect=[False, False, True]), \
             mock.patch.object(queue.owner, "_acquire", side_effect=[(0, "uuid-a", 5, leases[0]),
                 (0, "uuid-a", 5, leases[1])]) as acquire, \
             mock.patch.object(queue.owner, "_stable_idle", side_effect=[False, True]), \
             mock.patch.object(queue.subprocess, "Popen", return_value=child) as popen, \
             mock.patch.object(Path, "open", return_value=mock.mock_open()()), \
             mock.patch.object(queue, "RUNTIME", Path("unused-runtime")):
            result = instance._execute_seed(row)
        self.assertTrue(result)
        self.assertEqual(acquire.call_count, 2)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual([lease.close.call_count for lease in leases], [1, 1])
        self.assertEqual(fp.flock.call_count, 2)

    def test_original_queue_must_be_terminal_and_child_free_before_recovery(self):
        terminal = {"experiment": queue.EXPERIMENT, "status": "predictions_failed",
                    "supervisor_pid": 71, "tasks": {}}
        with mock.patch.object(Path, "read_text", return_value=json.dumps(terminal)):
            returned = queue.wait_original_queue_clear(process_reader=lambda _pid: None)
        self.assertEqual(returned["status"], "predictions_failed")
        pending = dict(terminal, status="predictions_running")
        sleep = mock.Mock()
        with mock.patch.object(Path, "read_text", side_effect=[json.dumps(pending), json.dumps(terminal)]):
            returned = queue.wait_original_queue_clear(process_reader=lambda _pid: None, sleep=sleep)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(returned["status"], "predictions_failed")

        sleep = mock.Mock()
        realistic = {"pid": 71, "state": "S", "argv": ["python", "-m",
            "collaborative_test.SW_0154_native64_resolution_compare.prediction_queue"]}
        with mock.patch.object(Path, "read_text", return_value=json.dumps(terminal)):
            returned = queue.wait_original_queue_clear(
                process_reader=mock.Mock(side_effect=[realistic, None]), sleep=sleep)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(returned["status"], "predictions_failed")

    def test_frozen_score_runs_only_after_all_seed_predictions_validate(self):
        output = queue.HERE / "results_archive" / f"recovery_score_test_{os.getpid()}.json"
        if output.exists():
            self.fail("preserve pre-existing score fixture output")
        events = []
        report = {"experiment": queue.EXPERIMENT, "status": "complete",
            "seed_scope": [0, 1, 2],
            "ground_truth_opened_after_all_six_predictions_validated": True,
            "implementation_fingerprint": queue.evaluate.implementation_fingerprint()}

        def frozen_score():
            for seed in (0, 1, 2):
                queue.evaluate.validate_native_prediction(seed)
            events.append("gt_and_metrics")
            return report

        original_path = queue.original_queue.EVALUATION_OUTPUT
        try:
            queue.original_queue.EVALUATION_OUTPUT = output
            with mock.patch.object(queue, "validate_prediction", side_effect=lambda seed, **_kw:
                    events.append(f"candidate_{seed}") or {"seed": seed}), \
                 mock.patch.object(queue, "FROZEN_VALIDATE_PREDICTION",
                    side_effect=lambda seed, **_kw: events.append(f"source_{seed}") or {"seed": seed}), \
                 mock.patch.object(queue.evaluate, "score_all", side_effect=frozen_score):
                result = queue.score_after_predictions()
            self.assertEqual(result["status"], "complete")
            self.assertLess(max(events.index("candidate_0"), events.index("candidate_2")),
                            events.index("gt_and_metrics"))
            self.assertTrue(queue.valid_evaluation())
        finally:
            queue.original_queue.EVALUATION_OUTPUT = original_path
            if output.exists():
                output.unlink()

    def test_actual_frozen_score_all_validates_seed_scoped_predictions_before_hdf5(self):
        ids = np.asarray(queue.evaluate.IMAGE_IDS, dtype=np.int64)
        labels = np.zeros((queue.evaluate.COUNT, 64, 64), dtype=np.int64)
        target_mask = np.zeros((queue.evaluate.COUNT, 128, 128, 1), dtype=np.uint8)
        target_mask[:, 16:56, 16:56, 0] = 1
        target_mask[:, 16:56, 56:96, 0] = 2
        dataset = Path("synthetic-canonical-mask.h5")
        stat = SimpleNamespace(st_size=1234, st_mtime_ns=5678)
        fake_rows = {}
        for seed in (0, 1, 2):
            proto = {"canonical_dataset": {"path": str(dataset.resolve()),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "source_checkpoint_sha256": f"{seed}" * 64,
                "model_initialization_seed": queue.predictor.MODEL_INIT_SEED}
            fake_rows[seed] = {"seed": seed, "ids": ids, "labels": labels.copy(),
                "npz_path": Path(f"candidate{seed}.npz"),
                "protocol_path": Path(f"candidate{seed}.json"),
                "prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64,
                "protocol": proto}
        slot_rows = {seed: {"seed": seed, "ids": ids, "labels64": labels.copy(),
            "npz_path": Path(f"slot{seed}.npz"), "protocol_path": Path(f"slot{seed}.json"),
            "prediction_sha256": "c" * 64, "protocol_sha256": "d" * 64,
            "protocol": {"checkpoint_sha256": "e" * 64}} for seed in (0, 1, 2)}
        output = queue.HERE / "results_archive" / f"score_all_recovery_test_{os.getpid()}.json"
        expected_prediction_fp = queue.predictor.implementation_fingerprint()
        expected_evaluation_fp = queue.evaluate.implementation_fingerprint()
        events = []
        mask_ds = mock.MagicMock()
        mask_ds.shape = (100000, 128, 128, 1)
        mask_ds.__getitem__.side_effect = lambda _slice: events.append("gt_read") or target_mask
        h5 = mock.MagicMock()
        h5.__contains__.side_effect = lambda name: name == "mask"
        h5.__getitem__.return_value = mask_ds
        cm = mock.MagicMock()
        cm.__enter__.return_value = h5
        original_eval_path = queue.original_queue.EVALUATION_OUTPUT
        original_stat, original_is_file = Path.stat, Path.is_file

        def fake_stat(path, *args, **kwargs):
            return stat if str(path) == str(dataset) else original_stat(path, *args, **kwargs)

        def fake_is_file(path):
            return True if str(path) == str(dataset) else original_is_file(path)

        def fake_hash(path):
            name = Path(path).name
            if name.startswith("candidate"):
                return "a" * 64 if name.endswith(".npz") else "b" * 64
            if name.startswith("slot"):
                return "c" * 64 if name.endswith(".npz") else "d" * 64
            return "f" * 64

        def candidate(seed, **kwargs):
            events.append(f"candidate{seed}")
            if "expected_fingerprint" in kwargs:
                self.assertEqual(kwargs["expected_fingerprint"], expected_prediction_fp)
            return fake_rows[seed]

        def frozen_validator(seed, **kwargs):
            events.append(f"frozen{seed}")
            return fake_rows[seed]

        try:
            queue.original_queue.EVALUATION_OUTPUT = output
            with mock.patch.object(queue, "validate_prediction", side_effect=candidate), \
                 mock.patch.object(queue, "FROZEN_VALIDATE_PREDICTION", side_effect=frozen_validator), \
                 mock.patch.object(queue.evaluate.score_slot64, "prepare_all_slot_predictions",
                    return_value=slot_rows), \
                 mock.patch.object(queue.evaluate.h5py, "File", return_value=cm), \
                 mock.patch.object(queue.evaluate, "sha256_file", side_effect=fake_hash), \
                 mock.patch.object(queue.evaluate, "DATASET", dataset), \
                 mock.patch.object(queue.evaluate, "implementation_fingerprint", return_value=expected_evaluation_fp), \
                 mock.patch.object(queue.evaluate.predict_native64, "implementation_fingerprint",
                    return_value=expected_prediction_fp), \
                 mock.patch.object(queue, "valid_evaluation", return_value=True), \
                 mock.patch.object(Path, "stat", fake_stat), \
                 mock.patch.object(Path, "is_file", fake_is_file):
                report = queue.score_after_predictions(dataset=dataset)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["seed_scope"], [0, 1, 2])
            self.assertIn("gt_read", events)
            first_gt = events.index("gt_read")
            self.assertEqual(events.count("candidate0"), 2)
            self.assertEqual(events.count("candidate2"), 2)
            self.assertEqual(events.count("frozen1"), 1)
            self.assertTrue(all(events.index(f"candidate{seed}") < first_gt for seed in (0, 2)))
            self.assertLess(events.index("frozen1"), first_gt)
        finally:
            queue.original_queue.EVALUATION_OUTPUT = original_eval_path
            if output.exists():
                output.unlink()

    def test_existing_seed1_is_never_in_recovery_task_plan(self):
        rows = queue.task_plan()
        self.assertNotIn(1, [row["seed"] for row in rows])
        self.assertTrue(all("seed1" not in row["output"] for row in rows))


if __name__ == "__main__":
    unittest.main()
