from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest import mock
import uuid

import h5py
import numpy
import numpy as np

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0160_end_to_end_source_encoder import predict_evaluate as evaluation, train


class _Scratch:
    def __init__(self, base):
        self.parent = Path(base) / "results_archive" / "cpu_test_scratch"
        self.path = self.parent / f"eval-{uuid.uuid4().hex}"

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, *_):
        resolved_parent = self.parent.resolve()
        resolved_path = self.path.resolve()
        if not resolved_path.is_relative_to(resolved_parent):
            raise RuntimeError("refusing to remove a test path outside its owned scratch root")
        for child in sorted(self.path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink()
        self.path.rmdir()


class SW0160PredictionTests(unittest.TestCase):
    def test_committed_row_roundtrip_and_tamper_rejection(self):
        labels = np.zeros((32, 32), dtype=np.int64)
        labels[:, 16:] = 2
        with _Scratch(evaluation.ROOT) as temp:
            record = evaluation._write_row(temp, 0, 1320, "a" * 64, "b" * 64, labels)
            loaded = evaluation._load_row(temp, 0, 1320, "a" * 64, "b" * 64)
            self.assertTrue(np.array_equal(loaded, labels))
            row = evaluation._row_dir(temp, 0) / "row.npz"
            row.write_bytes(row.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "corrupt or misbound"):
                evaluation._load_row(temp, 0, 1320, "a" * 64, "b" * 64)

    def test_gt_last_scoring_uses_fixed320_real_metrics_and_validates_all_predictions_first(self):
        image_ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
        labels = np.zeros((contract.COUNT, 32, 32), dtype=np.int64)
        labels[:, :, 16:] = 1
        source_labels = labels.copy()
        records = {}
        with _Scratch(evaluation.ROOT) as temp:
            temp = Path(temp)
            for name in (*train.model.ARMS, "source97", "SW0143_T"):
                npz_path = temp / f"{name}.npz"
                protocol_path = temp / f"{name}.json"
                with npz_path.open("wb") as stream:
                    np.savez_compressed(stream, image_ids=image_ids, qcc_labels=source_labels)
                protocol_path.write_text("{}", encoding="utf-8")
                records[name] = {"ids": image_ids, "labels": source_labels,
                    "npz_path": npz_path, "protocol_path": protocol_path,
                    "prediction_sha256": evaluation.sha256_file(npz_path),
                    "protocol_sha256": evaluation.sha256_file(protocol_path),
                    "protocol": {"prediction_bindings": {
                        "dataset_path": str((temp / "gt.h5").resolve()),
                        "dataset_size_bytes": None, "dataset_mtime_ns": None}}}
            dataset = temp / "gt.h5"
            with h5py.File(dataset, "w") as h5:
                masks = h5.create_dataset("mask", shape=(int(image_ids[-1]) + 1, 128, 128), dtype="u1")
                target = np.zeros((contract.COUNT, 128, 128), dtype=np.uint8)
                target[:, :, 64:] = 1
                masks[int(image_ids[0]):int(image_ids[-1]) + 1] = target
            stat = dataset.stat()
            for record in records.values():
                record["protocol"]["prediction_bindings"].update(
                    dataset_size_bytes=stat.st_size, dataset_mtime_ns=stat.st_mtime_ns)
            validation_order = []
            def validate(arm, **kwargs):
                validation_order.append(arm)
                return records[arm]
            source, reference = records["source97"], records["SW0143_T"]
            with (mock.patch.object(evaluation, "validate_prediction", side_effect=validate),
                  mock.patch.object(evaluation, "_source_and_t", return_value=(source, reference))):
                report = evaluation.score_all(output=temp / "score.json", prediction_root=temp,
                                               dataset=dataset)
            self.assertEqual(set(validation_order), set(train.model.ARMS))
            self.assertEqual(len(validation_order), 2)
            self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
            self.assertFalse(report["ground_truth_used_for_prediction"])
            self.assertTrue(report["ground_truth_used_for_scoring"])
            self.assertEqual(set(report["scores"]), {*train.model.ARMS, "source97", "SW0143_T"})
            self.assertAlmostEqual(report["means"]["full_encoder_joint"]["fg_ari"], 1.0)
            self.assertEqual(len(report["paired_bootstrap"]["full_encoder_joint_minus_SW0143_T_fg_ari"]["ci95"]), 2)

    def test_pilot_gate_requires_candidate_to_beat_both_registered_references(self):
        means = {name: {metric: 0.5 for metric in contract.METRICS}
                 for name in (*train.model.ARMS, "SW0143_T")}
        means["full_encoder_joint"]["fg_ari"] = 0.7
        means["frozen_base_joint"]["fg_ari"] = 0.6
        means["SW0143_T"]["fg_ari"] = 0.55
        paired = {"full_encoder_joint_minus_SW0143_T_fg_ari": {"ci95": [0.1, 0.2]},
                  "full_encoder_joint_minus_frozen_base_joint_fg_ari": {"ci95": [0.05, 0.15]}}
        gate, comparisons = evaluation._registered_seed1_gate(means, paired)
        self.assertTrue(gate["registered_seed1_gate_pass"])
        self.assertEqual(len(comparisons), 6)
        paired["full_encoder_joint_minus_frozen_base_joint_fg_ari"]["ci95"][0] = -0.01
        self.assertFalse(evaluation._registered_seed1_gate(means, paired)[0]["registered_seed1_gate_pass"])


if __name__ == "__main__":
    unittest.main()
