"""CPU contracts for SW176 scorer v2; prediction producer remains frozen v1."""
import json
import sys
import uuid
import unittest
from pathlib import Path
from unittest import mock

import numpy as np  # NumPy/OpenMP before legacy Torch imports.

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import h5py
from collaborative_test.SW_0176_local_phase_credit import (
    predict_evaluate as frozen, prediction_queue, score_queue_v2, score_v2,
)


class FakeOwner:
    def __init__(self, record=None):
        self.record = record

    def _proc_record(self, pid):
        return self.record


class ScoreV2Tests(unittest.TestCase):
    def _temporary(self):
        return (REPO / "collaborative_testing" / "collaborative_private_runtime_20261008"
                / "SW0176_evaluation_20261016" / "score_v2_test_tmp")

    def setUp(self):
        self.root = self._temporary()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tag = uuid.uuid4().hex
        self.created = []

    def path(self, name):
        path = self.root / f"{self.tag}-{name}"
        self.created.append(path)
        return path

    def tearDown(self):
        for path in reversed(self.created):
            if path.is_file():
                path.unlink()

    @staticmethod
    def _labels(count=320):
        grid = np.zeros((count, 32, 32), dtype=np.int32)
        for i in range(count):
            start = i % 5
            grid[i, 3:28, start + 8:start + 24] = 1
        return grid

    def test_core_accepts_native_standalone_partition_and_preserves_comparators_gt_last(self):
        root = self.root
        old_ids = frozen.IMAGE_IDS
        frozen.IMAGE_IDS = np.arange(320, dtype=np.int64)
        try:
            source = self._labels()
            standalone = np.flip(source, axis=2).copy()
            sw164 = self._labels()
            sw164[:, :, 0:4] = 2
            labels = {
                "SW0164": sw164,
                "SW0175_conditional": source.copy(),
                "SW0175_standalone": standalone.copy(),
                "candidate_conditional": source.copy(),
                "candidate_standalone": standalone.copy(),
            }
            gt32 = source.copy()
            gt128 = np.repeat(np.repeat(gt32, 4, axis=1), 4, axis=2).astype(np.int32)
            dataset = self.path("fixture.h5")
            with h5py.File(dataset, "w") as hdf5:
                hdf5.create_dataset("mask", data=gt128)
            stat = dataset.stat()

            refs = {}
            for name, arr in (("source_T", source), ("SW0164", sw164)):
                pred = self.path(f"{name}.npz")
                proto = self.path(f"{name}.json")
                pred.write_bytes((name + " prediction fixture").encode())
                proto.write_text(json.dumps({"fixture": name}), encoding="utf-8")
                refs[name] = {"labels": arr, "prediction_path": pred,
                              "protocol_path": proto,
                              "prediction_sha256": score_v2._sha_file(pred),
                              "protocol_sha256": score_v2._sha_file(proto)}
            pred = self.path("SW0175.npz")
            proto = self.path("SW0175.json")
            pred.write_bytes(b"SW0175 prediction fixture")
            proto.write_text('{"fixture":"SW0175"}', encoding="utf-8")
            refs["SW0175_control"] = {
                "labels": {"conditional": labels["SW0175_conditional"],
                           "standalone": labels["SW0175_standalone"]},
                "prediction_path": pred, "protocol_path": proto,
                "prediction_sha256": score_v2._sha_file(pred),
                "protocol_sha256": score_v2._sha_file(proto),
            }
            candidate = {
                "labels": {"conditional": labels["candidate_conditional"],
                           "standalone": labels["candidate_standalone"]},
                "prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64,
            }
            score_inputs = {
                "SW0164": labels["SW0164"], "Source-T": source,
                "SW0175_critical_component_conditional": labels["SW0175_conditional"],
                "SW0175_critical_component_standalone": labels["SW0175_standalone"],
                "local_phase_conditional": labels["candidate_conditional"],
                "local_phase_standalone": labels["candidate_standalone"],
            }
            baseline_scores = frozen._score_metrics(score_inputs, gt32)
            sw164_score = self.path("score164.json")
            sw175_score = self.path("score175.json")
            sw164_score.write_text(json.dumps({"scores": {
                "foreground_induced": baseline_scores["SW0164"]}}), encoding="utf-8")
            sw175_score.write_text(json.dumps({"scores": {
                "critical_component_conditional": baseline_scores[
                    "SW0175_critical_component_conditional"],
                "critical_component_standalone": baseline_scores[
                    "SW0175_critical_component_standalone"]}}), encoding="utf-8")
            prepared = {"references": refs, "dataset": dataset,
                        "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns}
            events = []
            real_file = h5py.File

            def track_file(*args, **kwargs):
                events.append("gt_open")
                return real_file(*args, **kwargs)

            def validate(*_args, **_kwargs):
                events.append("validate_predictions")
                return candidate

            with mock.patch.object(frozen, "validate_prediction", side_effect=validate), \
                 mock.patch.object(frozen.h5py, "File", side_effect=track_file), \
                 mock.patch.object(frozen.sw164_eval, "SCORE_PATH", sw164_score), \
                 mock.patch.object(frozen.eval175, "SCORE_PATH", sw175_score), \
                 mock.patch.object(frozen, "SW0164_SCORE_SHA256", score_v2._sha_file(sw164_score)), \
                 mock.patch.object(frozen, "SW0175_SCORE_SHA256", score_v2._sha_file(sw175_score)), \
                 mock.patch.object(frozen, "BOOTSTRAP_REPLICATES", 20):
                got_candidate, got_scores, paired, gates = score_v2._score_fixed320_core(
                    prediction_root=root, work_root=root, prepared=prepared)
            self.assertEqual(got_candidate["prediction_sha256"], "a" * 64)
            self.assertFalse(np.array_equal(
                got_scores["local_phase_standalone"]["foreground_iou"]["per_image"],
                got_scores["Source-T"]["foreground_iou"]["per_image"]))
            self.assertEqual(len(paired), 4)
            self.assertIn("standalone_fg_ari_mean_above_SW0164", gates)
            self.assertEqual(events[0], "validate_predictions")
            self.assertEqual(events[1], "gt_open")
            self.assertEqual(events[-1], "validate_predictions")
            self.assertEqual(got_scores["SW0164"], baseline_scores["SW0164"])
            self.assertEqual(got_scores["SW0175_critical_component_standalone"],
                             baseline_scores["SW0175_critical_component_standalone"])
        finally:
            frozen.IMAGE_IDS = old_ids

    def test_bad_conditional_membership_rejects_before_ground_truth_open(self):
        labels = self._labels()
        bad = labels.copy()
        bad[:, :, 0] = 1
        pred = {"labels": {"conditional": bad, "standalone": labels},
                "prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64}
        refs = {"source_T": {"labels": labels}}
        events = []
        with mock.patch.object(frozen, "validate_prediction", return_value=pred), \
             mock.patch.object(frozen, "_reference_hashes", return_value={}), \
             mock.patch.object(score_v2, "_sha_file", return_value="c" * 64), \
             mock.patch.object(frozen.h5py, "File") as gt_open:
            with self.assertRaisesRegex(ValueError, "conditional readout"):
                score_v2._score_fixed320_core(
                    prediction_root="unused", work_root="unused",
                    prepared={"references": refs})
        gt_open.assert_not_called()
        self.assertEqual(events, [])  # Explicitly before HDF5/GT access.

    def test_v2_queue_consumes_actual_v1_prediction_state_and_hashes(self):
        root = self.root
        out = root
        prediction = root / "predictions.npz"
        protocol = root / "protocol.json"
        self.created.extend((prediction, protocol))
        prediction.write_bytes(b"frozen predictions")
        protocol.write_text('{"ground_truth_used_for_prediction":false}', encoding="utf-8")
        state = {
            "experiment": "SW0176_prediction_queue_v1", "status": "complete",
            "output": str(out), "prediction_sha256": score_queue_v2._sha(prediction),
            "protocol_sha256": score_queue_v2._sha(protocol),
            "ground_truth_used_for_prediction": False,
        }
        state_path = self.path("state.json")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with mock.patch.object(score_queue_v2, "PREDICTION_STATE", state_path):
            consumed = score_queue_v2._prediction_dependency(FakeOwner())
        self.assertEqual(consumed, state)
        state["prediction_sha256"] = "0" * 64
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with mock.patch.object(score_queue_v2, "PREDICTION_STATE", state_path):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                score_queue_v2._prediction_dependency(FakeOwner())

        state.update({"status": "running_prediction", "supervisor_pid": 24680})
        state_path.write_text(json.dumps(state), encoding="utf-8")
        owner = FakeOwner({"state": "S", "argv": [
            sys.executable, "-m", score_queue_v2.PREDICTION_MODULE,
        ]})
        with mock.patch.object(score_queue_v2, "PREDICTION_STATE", state_path):
            self.assertIs(score_queue_v2._prediction_dependency(owner), False)


if __name__ == "__main__":
    unittest.main()
