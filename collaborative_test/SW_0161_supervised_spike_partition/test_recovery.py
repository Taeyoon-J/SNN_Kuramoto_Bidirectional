"""CPU regressions for the isolated SW161 row-root and score recovery."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0161_supervised_spike_partition import predict_evaluate as pe
from collaborative_test.SW_0161_supervised_spike_partition import recovery
from collaborative_test.SW_0161_supervised_spike_partition import train


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = recovery.HERE / "_r"
        self.root.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_actual_captured_failure_records_and_log_roles_are_required(self):
        records = json.loads(recovery.CAPTURED_FAILURE_RECORDS.read_text(encoding="utf-8-sig"))
        by_name = {row["source_name"]: row for row in records}
        runtime = self.root / "runtime"
        deploy_logs = self.root / "prediction_deployment_CPU_20261014"
        deploy_logs.mkdir()
        configurations = {}
        for arm in train.ARMS:
            expected = recovery.EXPECTED_FAILURE_RECORDS[arm]
            record = by_name[expected["source_name"]]["record"]
            state = runtime / arm / "state.json"
            child_log = runtime / arm / "attempt01.log"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(record), encoding="utf-8")
            child_log.write_text("prediction_complete count=320\\n", encoding="utf-8")
            (deploy_logs / f"{arm}_supervisor.log").write_text(
                "parent validation failed: FileNotFoundError(2, 'No such file or directory')\\n",
                encoding="utf-8")
            configurations[arm] = {"state": state, "log": child_log}
        score_record = by_name[recovery.EXPECTED_FAILURE_RECORDS["score"]["source_name"]]["record"]
        score_state = runtime / "score" / "state.json"
        score_state.parent.mkdir()
        score_state.write_text(json.dumps(score_record), encoding="utf-8")
        (deploy_logs / "score_supervisor.log").write_text(
            "score validation failed: FileNotFoundError(2, 'No such file or directory')\\n",
            encoding="utf-8")
        missing_score_queue_log = runtime / "score" / "score.log"
        with mock.patch.object(recovery.prediction_queue, "configure_arm",
                               side_effect=lambda arm: configurations[arm]), \
             mock.patch.object(recovery.score_queue, "STATE", score_state), \
             mock.patch.object(recovery.score_queue, "LOG", missing_score_queue_log):
            evidence = recovery._original_failure_evidence(deployment_logs=deploy_logs)
            self.assertEqual(evidence["pair_supervised_P"]["state"]["status"], "failed")
            self.assertIn("supervisor_log_sha256", evidence["pair_supervised_P"])
            self.assertEqual(evidence["score"]["state"]["error"],
                             "FileNotFoundError(2, 'No such file or directory')")
            bad = json.loads(configurations["pair_supervised_P"]["state"].read_text(encoding="utf-8"))
            bad["status"] = "failed_child_or_prediction_validation"
            configurations["pair_supervised_P"]["state"].write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                recovery._original_failure_evidence(deployment_logs=deploy_logs)

    def _make_two_row_prediction(self, arm):
        ids = np.asarray([1320, 1321], dtype=np.int64)
        labels = np.zeros((2, pe.GRID, pe.GRID), dtype=np.int64)
        labels[1, :, 16:] = 1
        rgb = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        artifact = {"manifest_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "manifest": {"implementation_fingerprint": {"train.py": "d" * 64},
                "cache_manifest_sha256": "e" * 64, "cache_bindings_sha256": "f" * 64}}
        bindings = {"experiment": train.EXPERIMENT, "arm": arm,
            "training_manifest_sha256": artifact["manifest_sha256"],
            "training_checkpoint_sha256": artifact["checkpoint_sha256"],
            "training_model_state_sha256": artifact["model_sha256"],
            "source_model_state_sha256": "1" * 64,
            "dataset_path": str((self.root / "dataset.h5").resolve()),
            "dataset_size_bytes": 12, "dataset_mtime_ns": 34}
        bind_sha = pe._sha_json(bindings)
        work = self.root / "work"
        arm_root = work / f"{arm}_seed1"
        (arm_root / "rows").mkdir(parents=True)
        (arm_root / "bindings.json").write_text(json.dumps({"bindings": bindings,
            "bindings_sha256": bind_sha}), encoding="utf-8")
        rgb_sha = __import__("hashlib").sha256(rgb[0:1].tobytes()).hexdigest()
        for i, image_id in enumerate(ids.tolist()):
            pe._write_row(arm_root, i, image_id, rgb_sha, bind_sha, labels[i])
        with mock.patch.object(pe, "COUNT", 2):
            rows_sha = pe._rows_digest(arm_root)
        protocol = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm,
            "seed": 1, "image_ids": [1320, 1321], "count": 2,
            "grid_size": [pe.GRID, pe.GRID], "prediction_shape": [2, pe.GRID, pe.GRID],
            "steps": 1024, "settle": 512, "live_tail_steps": 64, "optimizer_updates": 0,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": False,
            "prediction_bindings": bindings, "prediction_bindings_sha256": bind_sha,
            "prediction_row_commits_sha256": rows_sha,
            "implementation_fingerprint": pe.implementation_fingerprint(),
            "training_manifest_sha256": artifact["manifest_sha256"],
            "training_checkpoint_sha256": artifact["checkpoint_sha256"],
            "training_model_state_sha256": artifact["model_sha256"],
            "source_model_state_sha256_before": bindings["source_model_state_sha256"],
            "source_model_state_sha256_after": bindings["source_model_state_sha256"],
            "model_state_sha256_before": artifact["model_sha256"],
            "model_state_sha256_after": artifact["model_sha256"]}
        output = self.root / "predictions" / f"{arm}_seed1"
        with mock.patch.object(pe, "IMAGE_IDS", ids):
            pe._publish_prediction(output, labels, protocol)
        prepared = {"artifact": artifact, "rgb": rgb, "bindings": bindings,
            "bindings_sha256": bind_sha}
        return ids, labels, output.parent, work, prepared

    def test_real_public_row_producer_original_failure_and_arm_scoped_validator(self):
        for arm in train.ARMS:
            ids, expected_labels, prediction_root, work_root, prepared = self._make_two_row_prediction(arm)
            with mock.patch.object(pe, "IMAGE_IDS", ids), mock.patch.object(pe, "COUNT", 2), \
                 mock.patch.object(pe.contract, "require_fixed_image_ids", side_effect=lambda value, **kwargs: np.asarray(value)), \
                 mock.patch.object(pe, "prepare_prediction_cpu", return_value=prepared):
                # The original validator reproduces the known root-path error without any adapter.
                with self.assertRaises(FileNotFoundError):
                    recovery.FROZEN_VALIDATE(arm, prediction_root=prediction_root, work_root=work_root)
                validated = recovery.validate_prediction(arm, prediction_root=prediction_root, work_root=work_root)
                np.testing.assert_array_equal(validated["labels"], expected_labels)
                self.assertEqual(validated["ids"].tolist(), ids.tolist())

                row_json = work_root / f"{arm}_seed1" / "rows" / "0001" / "row.json"
                row_record = json.loads(row_json.read_text(encoding="utf-8"))
                row_record["image_id"] = 9999
                row_json.write_text(json.dumps(row_record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    recovery.validate_prediction(arm, prediction_root=prediction_root, work_root=work_root)
        with self.assertRaises(ValueError):
            recovery.validate_prediction("unregistered")

    def test_score_entry_validates_both_arms_before_scorer_can_read_ground_truth(self):
        validated = set()
        score_output = self.root / "score.json"
        receipt_path = self.root / "recovery_state.json"
        archive_path = self.root / "archive"
        fake_predictions = {}
        for arm in train.ARMS:
            folder = self.root / arm
            folder.mkdir()
            npz, protocol = folder / "predictions.npz", folder / "protocol.json"
            npz.write_bytes(arm.encode()); protocol.write_bytes((arm + " protocol").encode())
            fake_predictions[arm] = {"prediction_sha256": recovery.sha256_file(npz),
                "protocol_sha256": recovery.sha256_file(protocol), "npz_path": npz,
                "protocol_path": protocol}

        def validate(arm, **kwargs):
            validated.add(arm)
            return fake_predictions[arm]

        report = {"status": "complete", "registered_seed1_gate": {"registered_seed1_gate_pass": False},
            "ground_truth_opened_after_all_predictions_validated": True,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True}

        def score_all(*, output, **kwargs):
            self.assertEqual(validated, set(train.ARMS), "GT scoring began before both predictions were validated")
            Path(output).write_text("frozen score report", encoding="utf-8")
            return report

        with mock.patch.object(recovery, "_original_failure_evidence", return_value={}), \
             mock.patch.object(recovery, "fcntl", object()), \
             mock.patch.object(recovery, "_exclusive_old_queue_locks"), \
             mock.patch.object(recovery, "PINNED_PREDICTIONS", {
                 arm: {"prediction_sha256": fake_predictions[arm]["prediction_sha256"],
                       "protocol_sha256": fake_predictions[arm]["protocol_sha256"]}
                 for arm in train.ARMS}), \
             mock.patch.object(recovery, "validate_prediction", side_effect=validate), \
             mock.patch.object(recovery, "_archive_failure", return_value={"path": str(archive_path), "manifest_sha256": "a" * 64}), \
             mock.patch.object(recovery, "_fingerprint", return_value={"recovery.py": "b" * 64}), \
             mock.patch.object(recovery.evaluation, "implementation_fingerprint", return_value={"eval.py": "c" * 64}), \
             mock.patch.object(recovery.evaluation, "_rows_digest", return_value="d" * 64), \
             mock.patch.object(recovery.evaluation, "score_all", side_effect=score_all), \
             mock.patch.object(recovery.evaluation, "validate_score", return_value=report):
            receipt = recovery.recover_and_score(prediction_root=self.root, work_root=self.root,
                dataset=self.root / "unused.h5", output=score_output, archive=archive_path,
                recovery_state=receipt_path)
        self.assertEqual(receipt["status"], "complete_recovered")
        self.assertTrue(receipt["original_score_queue_state_preserved"])
        self.assertTrue(receipt["ground_truth_opened_after_all_predictions_validated"])
        self.assertEqual(json.loads(receipt_path.read_text(encoding="utf-8"))["score_sha256"],
            recovery.sha256_file(score_output))


if __name__ == "__main__":
    unittest.main()
