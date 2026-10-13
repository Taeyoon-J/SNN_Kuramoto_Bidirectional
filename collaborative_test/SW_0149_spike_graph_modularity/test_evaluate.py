from __future__ import annotations

import json
import contextlib
import io
import unittest
from pathlib import Path
from tempfile import gettempdir
from unittest import mock
from uuid import uuid4

import h5py
import numpy as np

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0149_spike_graph_modularity import evaluate


class ScoreFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(gettempdir()) / f"sw149_score_fixture_{uuid4().hex}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.dataset = cls.root / "masks.h5"
        masks = np.zeros((1640, 128, 128, 1), dtype=np.uint8)
        patch_target = np.zeros((32, 32), dtype=np.int64)
        patch_target[7:23, 4:12] = 1
        patch_target[7:23, 20:28] = 2
        for row in range(32):
            for col in range(32):
                masks[1320:1640, row * 4:(row + 1) * 4,
                      col * 4:(col + 1) * 4, 0] = patch_target[row, col]
        with h5py.File(cls.dataset, "w") as target:
            target.create_dataset("mask", data=masks, chunks=(1, 128, 128, 1), compression="gzip")
        cls.labels = np.broadcast_to(patch_target, (contract.COUNT, 32, 32)).copy()

    @classmethod
    def tearDownClass(cls):
        for child in cls.root.iterdir():
            child.unlink()
        cls.root.rmdir()

    def _frozen_row(self, name, labels, ids):
        archive = self.root / f"{name}.npz"
        protocol = self.root / f"{name}.json"
        archive.write_bytes(f"npz:{name}".encode())
        protocol.write_bytes(f"protocol:{name}".encode())
        return {"ids": np.asarray(ids, dtype=np.int64), "labels": labels,
                "archive_path": archive, "protocol_path": protocol,
                "prediction_sha256": evaluate.sha(archive),
                "protocol_sha256": evaluate.sha(protocol),
                "protocol": {"training_manifest_sha256": "a" * 64}}

    def test_score_uses_production_metrics_and_opens_gt_after_all_reference_freezes(self):
        ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
        own = {arm: self._frozen_row(arm, self.labels, ids) for arm in evaluate.ARMS}
        source = self._frozen_row("source", self.labels, ids)
        slot = self._frozen_row("slot", self.labels, ids)
        baseline_path = self.root / "slot_baseline.json"
        baseline = {"scores": {"slot32": {
            metric: {"three_seed_mean": 0.0} for metric in contract.METRICS}}}
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        baseline_sha = evaluate.sha(baseline_path)
        output = self.root / "score.json"
        calls = []
        real_h5_file = h5py.File

        def opened(path, *args, **kwargs):
            if Path(path) == self.dataset:
                calls.append("ground_truth")
            return real_h5_file(path, *args, **kwargs)

        def validate_arm(arm):
            calls.append(f"arm:{arm}")
            return own[arm]

        def validate_source(*args, **kwargs):
            calls.append("source")
            return source

        def validate_slot():
            calls.append("slot")
            return slot

        with (mock.patch.object(evaluate, "validate_prediction", side_effect=validate_arm),
              mock.patch.object(evaluate.sw137, "validate_prediction", side_effect=validate_source),
              mock.patch.object(evaluate, "_slot_reference", side_effect=validate_slot),
              mock.patch.object(evaluate, "SLOT_BASELINE", baseline_path),
              mock.patch.object(evaluate, "SLOT_BASELINE_SHA256", baseline_sha),
              mock.patch.object(evaluate.h5py, "File", side_effect=opened)):
            result = evaluate.score_all(output=output, dataset=self.dataset)
        self.assertEqual(calls, ["arm:encoder_graph", "arm:joint", "source", "slot", "ground_truth"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["bootstrap"]["seed"], 139)
        self.assertEqual(set(result["scores"]), {"encoder_graph", "joint", "source97_seed1", "Slot32_seed1"})
        self.assertTrue(all(row["valid_count"] == 320
                            for arm in result["scores"].values()
                            for row in arm.values()))
        self.assertFalse(result["may_promote_seed_expansion"])

    def test_score_rejects_reference_id_mismatch_before_ground_truth_open(self):
        ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
        own = {arm: self._frozen_row(arm, self.labels, ids) for arm in evaluate.ARMS}
        source = self._frozen_row("source_bad", self.labels, ids + 1)
        slot = self._frozen_row("slot_bad", self.labels, ids)
        baseline_path = self.root / "slot_baseline_bad.json"
        baseline_path.write_text(json.dumps({"scores": {"slot32": {
            metric: {"three_seed_mean": 0.0} for metric in contract.METRICS}}}), encoding="utf-8")
        calls = []
        with (mock.patch.object(evaluate, "validate_prediction", side_effect=lambda arm: own[arm]),
              mock.patch.object(evaluate.sw137, "validate_prediction", return_value=source),
              mock.patch.object(evaluate, "_slot_reference", return_value=slot),
              mock.patch.object(evaluate, "SLOT_BASELINE", baseline_path),
              mock.patch.object(evaluate, "SLOT_BASELINE_SHA256", evaluate.sha(baseline_path)),
              mock.patch.object(evaluate.h5py, "File", side_effect=lambda *a, **k: calls.append("GT"))):
            with self.assertRaisesRegex(ValueError, "image IDs differ"):
                evaluate.score_all(output=self.root / "bad_score.json", dataset=self.dataset)
        self.assertEqual(calls, [])

    def test_cli_prediction_and_scoring_ground_truth_flags_are_distinct(self):
        for stage, arm, expected_scoring in (
            ("predict", "encoder_graph", False),
            ("score", None, True),
        ):
            output = io.StringIO()
            result = {"status": "complete"}
            with contextlib.redirect_stdout(output):
                if stage == "predict":
                    with mock.patch.object(evaluate, "predict_arm", return_value=result):
                        self.assertEqual(evaluate.main(["--stage", stage, "--arm", arm]), 0)
                else:
                    with mock.patch.object(evaluate, "score_all", return_value=result):
                        self.assertEqual(evaluate.main(["--stage", stage]), 0)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["ground_truth_used_for_prediction"])
            self.assertEqual(payload["ground_truth_used_for_scoring"], expected_scoring)


if __name__ == "__main__":
    unittest.main()
