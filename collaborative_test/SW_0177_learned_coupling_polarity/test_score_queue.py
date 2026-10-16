"""CPU tests for SW0177 GT-last scoring and prediction-state consumption."""
import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np  # NumPy/OpenMP before h5py/Torch imports.

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from collaborative_test.SW_0177_learned_coupling_polarity import (
    predict_evaluate as evaluate, score_queue, train,
)


class FakeOwner:
    def __init__(self, record=None): self.record = record
    def _proc_record(self, pid): return self.record


class FakeH5:
    def __init__(self, raw): self.dataset = FakeDataset(raw)
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def __contains__(self, key): return key == "mask"
    def __getitem__(self, key):
        if key != "mask": raise KeyError(key)
        return self.dataset


class FakeDataset:
    def __init__(self, raw): self.raw = raw; self.shape = (1640, 128, 128)
    def __getitem__(self, key): return self.raw


class ScoreQueueTests(unittest.TestCase):
    def setUp(self):
        self.central = REPO / "collaborative_testing" / "temporary"
        self.root = self.central.resolve()
        self.assertTrue(self.root.is_dir())
        self.created = []

    def file(self, label, data=b""):
        path = self.root / f"sw177_{uuid.uuid4().hex}_{label}"
        path.write_bytes(data)
        self.created.append(path)
        return path

    def tearDown(self):
        for path in self.created:
            if path.resolve().parent == self.root:
                path.unlink(missing_ok=True)

    def test_prediction_queue_dependency_requires_both_real_complete_receipts(self):
        outputs = {}
        for arm in train.ARMS:
            pred = self.file(f"{arm}_pred", (arm + " predictions").encode())
            protocol = self.file(f"{arm}_protocol", json.dumps({"arm": arm,
                "status": "complete", "prediction_sha256": score_queue._sha(pred),
                "ground_truth_used_for_prediction": False}).encode())
            state = {"experiment": "SW0177_prediction_queue_v1", "arm": arm,
                "status": "complete", "output": str(self.root),
                "prediction_sha256": score_queue._sha(pred),
                "protocol_sha256": score_queue._sha(protocol),
                "ground_truth_used_for_prediction": False}
            outputs[arm] = state
            got = score_queue._validate_prediction_receipt(arm, state,
                prediction_path=pred, protocol_path=protocol,
                prediction_sha256=score_queue._sha(pred),
                protocol_sha256=score_queue._sha(protocol),
                protocol_data=json.loads(protocol.read_text(encoding="utf-8")))
            self.assertEqual(got["state"], state)
            bad = dict(state, prediction_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "receipt/hash mismatch"):
                score_queue._validate_prediction_receipt(arm, bad,
                    prediction_path=pred, protocol_path=protocol,
                    prediction_sha256=score_queue._sha(pred),
                    protocol_sha256=score_queue._sha(protocol),
                    protocol_data=json.loads(protocol.read_text(encoding="utf-8")))

    def test_active_dependency_requires_exact_prediction_queue_module(self):
        runtime = {}
        for arm in train.ARMS:
            runtime[arm] = self.file(f"{arm}_state", json.dumps({
                "experiment": "SW0177_prediction_queue_v1", "arm": arm,
                "status": "running_prediction", "supervisor_pid": 811}).encode())
        owner = FakeOwner({"state": "S", "argv": [sys.executable, "-u", "-m",
            score_queue.PREDICTION_MODULE, "--arm", "signed"]})
        for status in sorted(score_queue.ACTIVE_PREDICTION):
            self.assertTrue(score_queue._prediction_is_pending(owner, {
                "status": status, "supervisor_pid": 811}, "signed"), status)
        self.assertFalse(score_queue._prediction_is_pending(owner, {
            "status": "running_prediction", "supervisor_pid": 811}, "nonnegative"))
        owner.record["argv"] += ["--worker", "--physical-gpu", "2"]
        self.assertFalse(score_queue._prediction_is_pending(owner, {
            "status": "running_prediction", "supervisor_pid": 811}, "signed"))

    @staticmethod
    def _labels():
        source = np.zeros((evaluate.COUNT, 32, 32), dtype=np.int32)
        native = np.zeros_like(source)
        sw164 = np.zeros_like(source)
        for i in range(evaluate.COUNT):
            source[i, 3:27, 5:20] = 1
            native[i, 2:30, (i % 4):(i % 4 + 12)] = 4
            sw164[i, 4:28, 6:21] = 2
        return source, native, sw164

    def test_score_gt_is_opened_only_after_both_frozen_predictions_and_rechecks(self):
        source, standalone, sw164 = self._labels()
        gt32 = source.copy()
        gt128 = np.repeat(np.repeat(gt32, 4, axis=1), 4, axis=2)
        raw = np.zeros((evaluate.COUNT, 128, 128), dtype=np.int32)
        raw[:] = gt128
        dataset = self.file("canonical_h5_identity", b"test dataset identity")
        stat = dataset.stat()
        refs = {}
        for name, values in (("source_T", source), ("SW0164", sw164)):
            pred = self.file(f"{name}_ref_npz", (name + " predictions").encode())
            protocol = self.file(f"{name}_ref_json", json.dumps({"reference": name}).encode())
            refs[name] = {"labels": values,
                "prediction_path": pred, "protocol_path": protocol,
                "prediction_sha256": evaluate.sha256_file(pred),
                "protocol_sha256": evaluate.sha256_file(protocol)}
        score_file = self.file("sw164_score", b"")
        expected_baseline = evaluate._score_metrics({"foreground_induced": sw164}, gt32)
        score_file.write_text(json.dumps({"scores": {
            "foreground_induced": expected_baseline["foreground_induced"]}}), encoding="utf-8")
        refs["SW0164_score"] = expected_baseline
        refs["SW0164_score_path"] = score_file
        refs["SW0164_score_sha256"] = evaluate.sha256_file(score_file)

        labels = {
            ("signed", "conditional"): source.copy(),
            ("nonnegative", "conditional"): source.copy(),
            ("signed", "standalone"): standalone.copy(),
            ("nonnegative", "standalone"): standalone.copy(),
        }
        supports = {"signed": standalone.copy(), "nonnegative": standalone.copy()}
        prediction_results = {}
        for arm in train.ARMS:
            prediction_results[arm] = {"result": {
                "labels": {ep: labels[(arm, ep)] for ep in evaluate.ENDPOINTS},
                "standalone_support": supports[arm],
                "prediction_sha256": (arm[0] * 64),
                "protocol_sha256": (arm[1] * 64)}, "state": {},
                "state_sha256": ("a" if arm == "signed" else "b") * 64}

        prepared = {"references": refs, "dataset": dataset,
            "dataset_path": str(dataset), "dataset_size_bytes": stat.st_size,
            "dataset_mtime_ns": stat.st_mtime_ns,
            "artifacts": {arm: self.root / f"fixture_{arm}" for arm in train.ARMS}}
        events = []
        class TrackFile:
            def __enter__(self): events.append("gt_open"); return FakeH5(raw)
            def __exit__(self, *_): return False

        def receipt(*_args, **_kwargs):
            events.append("freeze_predictions")
            return prediction_results

        def validate(arm, **_kwargs):
            events.append("post_gt_validate")
            return {"prediction_sha256": prediction_results[arm]["result"]["prediction_sha256"],
                    "protocol_sha256": prediction_results[arm]["result"]["protocol_sha256"]}

        real_sha = evaluate.sha256_file
        def score_sha(path):
            path = Path(path)
            if path.name == "manifest.json" and path.parent.name.startswith("fixture_"):
                return "f" * 64
            return real_sha(path)

        out = self.root / f"sw177_{uuid.uuid4().hex}_score.json"
        self.created.append(out)
        with mock.patch.object(evaluate, "_prediction_receipts", side_effect=receipt), \
             mock.patch.object(evaluate.h5py, "File", side_effect=lambda *_a, **_k: TrackFile()), \
             mock.patch.object(evaluate, "validate_prediction", side_effect=validate), \
             mock.patch.object(evaluate, "sha256_file", side_effect=score_sha), \
             mock.patch.object(evaluate, "SW0164_SCORE_SHA256", refs["SW0164_score_sha256"]), \
             mock.patch.object(evaluate, "BOOTSTRAP_REPLICATES", 20), \
             mock.patch.object(evaluate, "implementation_fingerprint", return_value={"fp": "1" * 64}):
            report = evaluate.score_all(output=out, prepared=prepared)
        self.assertLess(events.index("freeze_predictions"), events.index("gt_open"))
        self.assertGreater(events.index("post_gt_validate"), events.index("gt_open"))
        self.assertFalse(np.array_equal(standalone > 0, source > 0))
        self.assertTrue(report["gate"]["signed_standalone_foreground_exact_native_QCC"])
        self.assertTrue(report["gate"]["signed_conditional_foreground_exact_SourceT"])
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
