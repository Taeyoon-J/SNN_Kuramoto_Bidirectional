"""CPU contract tests for the public SW0162 prediction and GT-last score path."""
from __future__ import annotations

import hashlib
import json
import shutil
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0162_connected_signed_partition import predict_evaluate as pe
from collaborative_test.SW_0162_connected_signed_partition import preflight
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class PredictionScoreContracts(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[3]
        self.root = self.workspace / ("t162" + uuid.uuid4().hex[:8])
        self.root.mkdir()

    def tearDown(self):
        target = self.root.resolve()
        allowed = self.workspace
        if not target.is_relative_to(allowed):
            raise RuntimeError("refusing to remove test data outside this module")
        shutil.rmtree(target)

    def _fixture(self, root):
        ids = pe.IMAGE_IDS.copy()
        source_t_labels = np.zeros((pe.COUNT, 32, 32), dtype=np.int64)
        source_t_labels[:, :, :16] = 1
        qcc_labels = source_t_labels.copy()
        qcc_labels[:, :, 0] = 2  # Archived source QCC is allowed to differ from T.
        slot_labels = source_t_labels.copy()
        refs_dir = root / "refs"
        refs_dir.mkdir()
        refs = {}
        for name, labels in (("source_T", source_t_labels), ("source_qcc", qcc_labels), ("slot", slot_labels)):
            npz = refs_dir / f"{name}.npz"
            with npz.open("xb") as f:
                np.savez_compressed(f, image_ids=ids, labels=labels)
            protocol = refs_dir / f"{name}.json"
            protocol.write_text(json.dumps({"status": "complete", "name": name}), encoding="utf-8")
            record = {"ids": ids, "labels": labels, "prediction_sha256": _digest(npz),
                "protocol_sha256": _digest(protocol), "archive_path": npz, "protocol_path": protocol}
            if name == "source_T":
                record["npz_path"] = npz
            if name == "slot":
                record.update({"prediction_path": npz, "baseline_sha256": pe.SLOT_BASELINE_SHA256})
            refs[name] = record
        source_hash = "c" * 64
        prepared = {"preflight": {"source": {"source_state_sha256_before": source_hash}},
            "preflight_sha256": "p" * 64, "references": refs,
            "rgb_hashes": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(pe.COUNT)],
            "dataset": root / "train.h5", "dataset_path": str((root / "train.h5").resolve()),
            "dataset_size_bytes": 0, "dataset_mtime_ns": 0}
        return prepared, source_t_labels

    def _write_rows_and_predictions(self, root, prepared, source_t):
        work_root = root / "work"
        work = pe._work_dir(work_root)
        binding = {"fixture": "producer-shaped", "ids_sha": hashlib.sha256(pe.IMAGE_IDS.astype("<i8").tobytes()).hexdigest()}
        binding_sha = pe._sha_json(binding)
        labels = np.zeros((pe.GRID, pe.GRID), dtype=np.int32)
        labels[:, :16] = 1
        qcc = np.zeros((pe.GRID, pe.GRID), dtype=np.int32)
        qcc[:, :16] = 1
        diag = {"rollout_seconds": 1.25, "solver_seconds": .25, "peak_rss_bytes": 123456,
            "peak_cuda_allocated_bytes": 456, "peak_cuda_reserved_bytes": 789,
            "raw_group_sizes": [512, 512], "retained_group_count": 2,
            "discarded_mass": 0, "candidate_foreground_fraction": .5}
        trace_hashes = {key: "a" * 64 for key in preflight.TRACE_FIELDS}
        for i, image_id in enumerate(pe.IMAGE_IDS.tolist()):
            pe._write_row(work, i, image_id, prepared["rgb_hashes"][i], binding_sha,
                labels, qcc, "b" * 64, "d" * 64, 1.0, 3, trace_hashes, diag)
        with mock.patch.object(pe, "implementation_fingerprint", return_value={"test": "f" * 64}):
            pe._write_prediction_directory(pe._output_dir("connected_signed", root / "preds"),
                "connected_signed", np.stack([labels] * pe.COUNT), binding, binding_sha, work=work,
                source_state_before="c" * 64, source_state_after="c" * 64, source_t=prepared["references"]["source_T"])
            pe._write_prediction_directory(pe._output_dir("source_T_control", root / "preds"),
                "source_T_control", source_t.astype(np.int32), binding, binding_sha, work=work,
                source_t=prepared["references"]["source_T"])
        return work_root

    def test_public_artifact_writer_validator_and_tamper_guard(self):
            root = self.root
            prepared, source_t = self._fixture(root)
            work_root = self._write_rows_and_predictions(root, prepared, source_t)
            binding = {"fixture": "producer-shaped", "ids_sha": hashlib.sha256(pe.IMAGE_IDS.astype("<i8").tobytes()).hexdigest()}
            with (mock.patch.object(pe, "implementation_fingerprint", return_value={"test": "f" * 64}),
                  mock.patch.object(pe, "_prediction_bindings", return_value=binding)):
                candidate = pe.validate_prediction("connected_signed", prediction_root=root / "preds",
                    work_root=work_root, prepared=prepared)
                control = pe.validate_prediction("source_T_control", prediction_root=root / "preds",
                    work_root=work_root, prepared=prepared)
                np.testing.assert_array_equal(control["labels"], source_t)
                np.testing.assert_array_equal(candidate["ids"], pe.IMAGE_IDS)
                self.assertEqual(candidate["protocol"]["diagnostics"]["per_image"][0]["rollout_seconds"], 1.25)
                path = candidate["directory"] / "protocol.json"
                proto = json.loads(path.read_text(encoding="utf-8"))
                proto["bindings_sha256"] = "0" * 64
                path.write_text(json.dumps(proto), encoding="utf-8")
                with self.assertRaises(ValueError):
                    pe.validate_prediction("connected_signed", prediction_root=root / "preds",
                        work_root=work_root, prepared=prepared)

    def test_actual_public_predictor_captures_kuramoto_and_emits_fresh_qcc_control(self):
        with mock.patch.object(pe, "COUNT", 1), mock.patch.object(pe, "IMAGE_IDS", np.asarray([1320])):
            root = self.root / "predict"
            root.mkdir()
            qcc = (np.arange(1024, dtype=np.int64).reshape(1, 32, 32) % 3)
            prepared = {"preflight": {"source": {"source_state_sha256_before": "c" * 64}},
                "preflight_sha256": "p" * 64, "references": {"source_T": {
                    "labels": qcc, "prediction_sha256": "t" * 64, "protocol_sha256": "u" * 64}},
                "rgb": np.zeros((1, 128, 128, 3), dtype=np.uint8),
                "rgb_hashes": ["r" * 64]}
            coupling = SimpleNamespace()
            graph = SimpleNamespace(log_coupling_gain=pe.torch.tensor(0.0))
            wrapped = SimpleNamespace(core=SimpleNamespace(graph_generator=graph, kuramoto=coupling,
                kuramoto_backend="factorized"), eval=lambda: None)
            found = SimpleNamespace(encoder=SimpleNamespace(eval=lambda: None), wrapped=wrapped,
                encode=lambda image: pe.torch.zeros((1, 8, 1024), dtype=pe.torch.float32))
            trace = {key: pe.torch.ones(1, dtype=pe.torch.float32) for key in preflight.TRACE_FIELDS}
            actual_a = pe.torch.ones((1, 1024, 1024), dtype=pe.torch.float32)
            actual_q = pe.torch.full((1, 1024, 1024), .5, dtype=pe.torch.float32)
            captured_calls, written = [], {}

            def capture(graph_arg, invoke):
                captured_calls.append(graph_arg)
                return invoke(), actual_a

            def write_output(output, arm, labels, *args, **kwargs):
                written[arm] = np.asarray(labels).copy()

            image_stub = SimpleNamespace(to=lambda device: "cpu-image")

            with ExitStack() as stack:
                patches = [
                    mock.patch.object(pe, "prepare_prediction_cpu", return_value=prepared),
                    mock.patch.object(pe, "_prediction_bindings", return_value={"bound": True}),
                    mock.patch.object(pe, "implementation_fingerprint", return_value={"fp": "f" * 64}),
                    mock.patch.object(pe, "load_native32_foundation", return_value=found),
                    mock.patch.object(pe.torch, "from_numpy", return_value=image_stub),
                    mock.patch.object(pe.geodesic_operator, "install_operator"),
                    mock.patch.object(pe.torch.cuda, "is_available", return_value=True),
                    mock.patch.object(pe.torch.cuda, "set_device"),
                    mock.patch.object(pe.torch.cuda, "reset_peak_memory_stats"),
                    mock.patch.object(pe.torch.cuda, "max_memory_allocated", return_value=10),
                    mock.patch.object(pe.torch.cuda, "max_memory_reserved", return_value=20),
                    mock.patch.object(pe.preflight, "_source_state_digest", return_value="c" * 64),
                    mock.patch.object(pe.preflight, "capture_source_coupling", side_effect=capture),
                    mock.patch.object(pe, "late_rollout", return_value=trace),
                    mock.patch.object(pe, "_validate_trace"),
                    mock.patch.object(pe.preflight, "_actual_q", return_value=actual_q),
                    mock.patch.object(pe.preflight, "_validate_actual_A"),
                    mock.patch.object(pe.preflight, "_tensor_sha", return_value="a" * 64),
                    mock.patch.object(pe.sw136, "_qcc_labels",
                        return_value=prepared["references"]["source_T"]["labels"]),
                    mock.patch.object(pe.connected_signed, "connected_signed_merge",
                        return_value={"labels": np.zeros((32, 32), dtype=np.int32),
                            "clusters": [{"size": 1024}], "merge_count": 0}),
                    mock.patch.object(pe, "_write_prediction_directory", side_effect=write_output),
                    mock.patch.object(pe, "_finalize_rows_without_rollout", return_value={"status": "complete"}),
                ]
                active = [stack.enter_context(patch) for patch in patches]
                rollout = active[12]
                result = pe.predict_candidate(device="cuda:0", prediction_root=root / "preds",
                    work_root=root / "work", dataset=root / "data.h5")
            self.assertEqual(result["status"], "prediction_complete")
            self.assertEqual(captured_calls, [coupling])
            rollout.assert_called_once()
            self.assertEqual(set(written), {"connected_signed", "source_T_control"})
            np.testing.assert_array_equal(written["source_T_control"][0], qcc[0])

    def test_finalize_resumes_valid_pair_publish_without_rollout(self):
        prepared, source_t = self._fixture(self.root)
        work_root = self._write_rows_and_predictions(self.root, prepared, source_t)
        prediction_root = self.root / "preds"
        shutil.rmtree(pe._output_dir("source_T_control", prediction_root))
        binding = {"fixture": "producer-shaped", "ids_sha": hashlib.sha256(
            pe.IMAGE_IDS.astype("<i8").tobytes()).hexdigest()}
        with (mock.patch.object(pe, "implementation_fingerprint", return_value={"test": "f" * 64}),
              mock.patch.object(pe, "_prediction_bindings", return_value=binding)):
            result = pe._finalize_rows_without_rollout(prepared=prepared, bindings=binding,
                bind_sha=pe._sha_json(binding), work=pe._work_dir(work_root),
                prediction_root=prediction_root, work_root=work_root)
            control = pe.validate_prediction("source_T_control", prediction_root=prediction_root,
                work_root=work_root, prepared=prepared)
        self.assertTrue(result["resumed_from_committed_rows"])
        np.testing.assert_array_equal(control["labels"], source_t)

    def test_source_reference_resolver_uses_actual_heterogeneous_path_keys(self):
        ids = np.asarray([1320], dtype=np.int64)
        labels = np.zeros((1, 32, 32), dtype=np.int64)
        labels[:, :, :16] = 1
        qcc = labels.copy(); qcc[:, :, 0] = 2
        source_qcc_file = self.root / "source_qcc.npz"
        source_qcc_proto = self.root / "source_qcc.json"
        source_t_file = self.root / "source_t.npz"
        source_t_proto = self.root / "source_t.json"
        slot_root = self.root / "slot"
        slot_dir = slot_root / "seed1_epoch10"
        slot_dir.mkdir(parents=True)
        slot_file, slot_proto = slot_dir / "predictions.npz", slot_dir / "protocol.json"
        for path in (source_qcc_file, source_qcc_proto, source_t_file, source_t_proto, slot_file, slot_proto):
            path.write_bytes(b"fixture")
        np.savez(slot_file, image_ids=ids, labels=np.zeros((1, 128, 128), dtype=np.int64))
        slot_proto.write_text("{}", encoding="utf-8")
        qcc_record = {"ids": ids, "labels": qcc[0], "archive_path": source_qcc_file,
            "protocol_path": source_qcc_proto, "prediction_sha256": "q" * 64,
            "protocol_sha256": "r" * 64}
        t_record = {"ids": ids, "labels": labels[0], "npz_path": source_t_file,
            "protocol_path": source_t_proto, "prediction_sha256": "t" * 64,
            "protocol_sha256": "u" * 64, "arm": "T"}
        with ExitStack() as stack:
            for patcher in (
                mock.patch.object(pe, "IMAGE_IDS", ids), mock.patch.object(pe, "SLOT_ROOT", slot_root),
                mock.patch.object(pe, "SLOT_BASELINE", self.root / "baseline.json"),
                mock.patch.object(pe, "SLOT_BASELINE_SHA256", "b" * 64),
                mock.patch.object(pe, "sha256_file", side_effect=lambda p: "b" * 64 if Path(p).name == "baseline.json" else "s" * 64),
                mock.patch.object(pe.sw137, "validate_prediction", return_value=qcc_record),
                mock.patch.object(pe.sw143, "validate_arm_prediction", return_value=t_record),
                mock.patch.object(pe.sw143, "_arm_dir", return_value=self.root),
                mock.patch.object(contract, "validate_slot_protocol", return_value="v" * 64),
                mock.patch.object(contract, "require_fixed_image_ids", return_value=ids),
                mock.patch.object(contract, "slot_native_pixels_to32", return_value=np.zeros((1, 32, 32), dtype=np.int64)),
            ):
                stack.enter_context(patcher)
            refs = pe._source_references()
        self.assertEqual(refs["source_T"]["archive_path"], source_t_file)
        self.assertEqual(refs["source_qcc"]["archive_path"], source_qcc_file)
        self.assertNotEqual(refs["source_T"]["labels"].tolist(), refs["source_qcc"]["labels"].tolist())

    def test_public_cpu_inputs_probe_does_not_require_preflight_or_read_masks(self):
        rgb_path = self.root / "val_rgb.npy"
        rgb_path.write_bytes(b"registered cache placeholder")
        rgb_meta = {"status": "complete", "cache_shape": [320, 128, 128, 3],
            "ids_mapping_sha256": hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()}
        rgb_path.with_suffix(rgb_path.suffix + ".complete.json").write_text(json.dumps(rgb_meta), encoding="utf-8")
        dataset = self.root / "canonical.h5"
        dataset.write_bytes(b"hdf identity only")

        class RGBStub:
            shape = (320, 128, 128, 3)
            dtype = np.dtype(np.uint8)

            def __getitem__(self, key):
                count = len(range(*key.indices(320))) if isinstance(key, slice) else 1
                return np.zeros((count, 128, 128, 3), dtype=np.uint8)

        refs = {"source_T": {"labels": None}, "source_qcc": {"labels": None}, "slot": {"labels": None}}
        with (mock.patch.object(pe, "DATASET", dataset), mock.patch.object(pe.sw130, "VAL_RGB", rgb_path),
              mock.patch.object(pe, "_source_references", return_value=refs),
              mock.patch.object(pe, "sha256_file", return_value="m" * 64),
              mock.patch.object(pe.np, "load", return_value=RGBStub()),
              mock.patch.object(pe, "_preflight_record", side_effect=AssertionError("preflight must not be read")),
              mock.patch.object(pe.h5py, "File", side_effect=AssertionError("ground-truth masks must not be opened"))):
            prepared = pe.prepare_prediction_inputs_cpu(dataset=dataset)
        self.assertEqual(prepared["dataset_path"], str(dataset.resolve()))
        self.assertEqual(len(prepared["rgb_hashes"]), 320)
        self.assertNotIn("preflight", prepared)

    def test_score_path_validates_predictions_before_opening_masks(self):
            root = self.root
            prepared, source_t = self._fixture(root)
            # HDF5 mask payload is created only for this CPU contract test.
            import h5py
            with h5py.File(prepared["dataset"], "w") as hdf:
                masks = hdf.create_dataset("mask", shape=(1640, 128, 128, 1), dtype=np.uint8,
                    chunks=(1, 128, 128, 1), compression="lzf")
                gt = np.zeros((pe.COUNT, 128, 128, 1), dtype=np.uint8)
                gt[:, :, :64, 0] = 1
                masks[1320:1640] = gt
            stat = prepared["dataset"].stat()
            prepared["dataset_size_bytes"] = stat.st_size
            prepared["dataset_mtime_ns"] = stat.st_mtime_ns
            prepared["dataset_path"] = str(prepared["dataset"].resolve())
            work_root = self._write_rows_and_predictions(root, prepared, source_t)
            binding = {"fixture": "producer-shaped", "ids_sha": hashlib.sha256(pe.IMAGE_IDS.astype("<i8").tobytes()).hexdigest()}
            seen = []
            real_file = h5py.File

            def guarded_h5(*args, **kwargs):
                seen.append("gt-open")
                self.assertEqual(seen[:2], ["all-predictions-validated", "gt-open"])
                return real_file(*args, **kwargs)

            real_validator = pe.validate_prediction

            def validator(*args, **kwargs):
                result = real_validator(*args, **kwargs)
                if len(seen) == 0 and args[0] == "source_T_control":
                    seen.append("all-predictions-validated")
                return result

            with (mock.patch.object(pe, "prepare_prediction_cpu", return_value=prepared),
                  mock.patch.object(pe, "implementation_fingerprint", return_value={"test": "f" * 64}),
                  mock.patch.object(pe, "_prediction_bindings", return_value=binding),
                  mock.patch.object(pe, "validate_prediction", side_effect=validator),
                  mock.patch.object(pe.h5py, "File", side_effect=guarded_h5)):
                report = pe.score_all(output=root / "score.json", prediction_root=root / "preds",
                    work_root=work_root, dataset=prepared["dataset"])
                checked = pe.validate_score(root / "score.json", prediction_root=root / "preds",
                    work_root=work_root, dataset=prepared["dataset"], prepared=prepared)
            self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
            self.assertIs(report["ground_truth_used_for_prediction"], False)
            self.assertIs(report["ground_truth_used_for_scoring"], True)
            self.assertEqual(report["bootstrap"]["seed"], 162)
            self.assertIn("connected_signed_minus_source_T_fg_ari", report["paired_bootstrap"])
            self.assertEqual(checked["status"], "valid")

            # Keep all report-internal means, paired draws, and gate coherent while
            # tampering one per-image metric; the GT-backed recomputation must catch it.
            forged = json.loads((root / "score.json").read_text(encoding="utf-8"))
            forged["scores"]["connected_signed"]["fg_ari"]["per_image"][0] += .25
            forged_fg = float(np.mean(forged["scores"]["connected_signed"]["fg_ari"]["per_image"]))
            forged["scores"]["connected_signed"]["fg_ari"]["mean"] = forged_fg
            forged["means"]["connected_signed"]["fg_ari"] = forged_fg
            for model in pe.ARMS:
                for baseline in ("source_T", "source_qcc", "Slot70k"):
                    key = f"{model}_minus_{baseline}_fg_ari"
                    forged["paired_bootstrap"][key] = pe._bootstrap_delta(
                        forged["scores"][model]["fg_ari"]["per_image"],
                        forged["scores"][baseline]["fg_ari"]["per_image"])
            forged["seed1_expansion_gate"] = pe._registered_gate(forged["means"], forged["paired_bootstrap"])
            (root / "score.json").write_text(json.dumps(forged), encoding="utf-8")
            with (mock.patch.object(pe, "implementation_fingerprint", return_value={"test": "f" * 64}),
                  mock.patch.object(pe, "_prediction_bindings", return_value=binding),
                  self.assertRaisesRegex(ValueError, "independently recomputed")):
                pe.validate_score(root / "score.json", prediction_root=root / "preds",
                    work_root=work_root, dataset=prepared["dataset"], prepared=prepared)


if __name__ == "__main__":
    unittest.main()
