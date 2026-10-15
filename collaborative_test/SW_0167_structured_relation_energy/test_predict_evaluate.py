"""CPU contracts for SW0167 variable-K inference and frozen scoring."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0167_structured_relation_energy import predict_evaluate as pe
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


class PredictionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = (Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
                    / ("pe167" + uuid.uuid4().hex[:8]))
        self.tmp.mkdir()

    def tearDown(self):
        root = (Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary").resolve()
        target = self.tmp.resolve()
        if not target.is_relative_to(root):
            raise RuntimeError("unsafe fixture cleanup path")
        shutil.rmtree(target)

    def _write_row(self, candidate, control):
        work = self.tmp / "work"
        folder = pe._rowdir(work, 0)
        folder.mkdir(parents=True)
        arrays = {"structured_hinge": candidate, "local_margin": candidate,
                  "source_T_control": control}
        npz = folder / "row.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        labels_sha = {name: hashlib.sha256(value.astype("<i4", copy=False).tobytes()).hexdigest()
                      for name, value in arrays.items()}
        record = {"index": 0, "image_id": 1320, "rgb_sha256": "a" * 64,
                  "bindings_sha256": "b" * 64, "npz_sha256": _sha(npz),
                  "labels_sha256": labels_sha, "q_sha256": "c" * 64,
                  "actual_a_sha256": "d" * 64,
                  "trace_sha256": {k: format(i + 1, "064x")
                                   for i, k in enumerate(pe.source_capture.TRACE_FIELDS)},
                  "gain": 1.0, "rollout_seconds": 1.0,
                  "cuda_peak_allocated_bytes": 0, "foreground_support_exact": True}
        (folder / "row.json").write_text(json.dumps(record), encoding="utf-8")
        return work

    def test_row_reader_accepts_native_variable_k_through_1024(self):
        candidate = (np.arange(1024, dtype=np.int32) + 1).reshape(32, 32)
        control = np.full((32, 32), 127, dtype=np.int32)
        work = self._write_row(candidate, control)
        got, _ = pe._read_row(work, 0, 1320, "a" * 64, "b" * 64)
        self.assertEqual(int(got["structured_hinge"].max()), 1024)
        self.assertEqual(int(got["source_T_control"].max()), 127)

    def test_row_reader_rejects_candidate_overflow_and_control_overflow(self):
        candidate = np.ones((32, 32), dtype=np.int32)
        candidate[-1, -1] = 1025
        control = np.ones((32, 32), dtype=np.int32)
        work = self._write_row(candidate, control)
        with self.assertRaises(ValueError):
            pe._read_row(work, 0, 1320, "a" * 64, "b" * 64)
        candidate[:] = 1
        control[-1, -1] = 129
        shutil.rmtree(work)
        work = self._write_row(candidate, control)
        with self.assertRaises(ValueError):
            pe._read_row(work, 0, 1320, "a" * 64, "b" * 64)

    def test_prediction_import_preserves_assigned_gpu_without_cuda_init(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        code = ("import numpy; import torch, os; from collaborative_test.SW_0167_structured_relation_energy "
                "import predict_evaluate; assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'; "
                "assert not torch.cuda.is_initialized()")
        proc = subprocess.run([sys.executable, "-c", code], cwd=pe.ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_predictor_uses_numpy_qcc_and_one_shared_actual_rollout(self):
        qcc = np.ones((1, 32, 32), dtype=np.int64)
        q = torch.full((1, 1024, 1024), 0.25)
        a = torch.full((1, 1024, 1024), 0.01)
        trace = {name: torch.zeros(1) for name in pe.source_capture.TRACE_FIELDS}
        trace["component_spikes"] = torch.zeros((1, 4, 1024, 1024))
        graph = object()
        found = SimpleNamespace(
            wrapped=SimpleNamespace(core=SimpleNamespace(kuramoto=graph)),
            encode=lambda image: torch.zeros((1, 1)))

        class Head:
            def __call__(self, features, e0, wa):
                return e0, wa

        candidate = (np.arange(1024, dtype=np.int32) + 1).reshape(32, 32)
        registered_zero_head = np.ones((1, 32, 32), dtype=np.int32)
        registered_zero_head[:, :, 16:] = 2
        calls = []

        def solver(energy, support, *, grid_shape):
            calls.append(1)
            labels = candidate if len(calls) < 3 else registered_zero_head[0]
            return {"labels": labels, "merge_count": 0}

        def capture(actual_graph, thunk):
            self.assertIs(actual_graph, graph)
            return thunk(), a

        with (mock.patch.object(pe.source_capture, "capture_source_coupling", side_effect=capture),
              mock.patch.object(pe, "late_rollout", return_value=trace),
              mock.patch.object(pe.source_capture, "_trace_check"),
              mock.patch.object(pe.source_capture, "_validate_actual_A"),
              mock.patch.object(pe.source_capture, "_actual_q", return_value=q),
              mock.patch.object(pe.sw136, "_qcc_labels", return_value=qcc),
              mock.patch.object(pe.source_capture, "_tensor_sha", return_value="e" * 64),
              mock.patch.object(pe.feature_modes, "component_rho", return_value=torch.zeros((4, 1024, 1024))),
              mock.patch.object(pe.feature_modes, "features_from_rho_and_context_q",
                                return_value=torch.zeros((1024, 1024, 10))),
              mock.patch.object(pe.structured_energy, "energy_components",
                                return_value=(torch.zeros((1024, 1024), dtype=torch.float64),
                                              torch.ones((1024, 1024), dtype=torch.float64))),
              mock.patch.object(pe.structured_energy, "base_energy",
                                return_value=np.zeros((1024, 1024), dtype=np.float64)),
              mock.patch.object(pe.structured_energy, "solve_energy", side_effect=solver)):
            result = pe._predict_one(found, {"structured_hinge": Head(), "local_margin": Head()},
                                     torch.zeros((1, 128, 128, 3)),
                                     {"source_T": {"labels": qcc},
                                      "SW0164": {"labels": registered_zero_head}}, 0, 1.0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(int(result["labels"]["structured_hinge"].max()), 1024)
        np.testing.assert_array_equal(result["control"], qcc[0])
        self.assertFalse(np.array_equal(registered_zero_head[0], result["control"]))

    def test_comparison_builder_uses_source_control_local_arm_and_seed164(self):
        count = 8

        def metric(values):
            v = np.asarray(values, dtype=np.float64)
            return {"mean": float(v.mean()), "per_image": v}

        def score(fg, obj, iou):
            return {"fg_ari": metric(fg), "matched_object_iou": metric(obj),
                    "foreground_iou": metric(iou)}

        baseline = score(np.full(count, .5), np.full(count, .4), np.full(count, .3))
        scores = {
            "structured_hinge": score(np.full(count, .8), np.full(count, .5), np.full(count, .3)),
            "local_margin": score(np.full(count, .6), np.full(count, .4), np.full(count, .3)),
            "source_T_control": score(np.full(count, .55), np.full(count, .4), np.full(count, .3)),
        }
        with (mock.patch.object(pe, "COUNT", count),
              mock.patch.object(pe, "BOOTSTRAP_REPLICATES", 128)):
            means, paired, gate, exact = pe._score_comparisons(scores, baseline)
        self.assertIn("SW0164_seed1", means)
        self.assertIn("structured_hinge_minus_source_T_fg_ari", paired)
        self.assertTrue(exact)
        self.assertTrue(gate["seed_expansion_gate_passed"])

    def test_public_gt_last_score_and_independent_validator(self):
        ids = np.asarray([1320, 1321], dtype=np.int64)
        gt32 = np.zeros((2, 32, 32), dtype=np.int32)
        gt32[:, :, :16] = 1
        gt32[:, :, 16:] = 2
        pixel_gt = np.repeat(np.repeat(gt32, 4, axis=1), 4, axis=2)[..., None].astype(np.uint8)
        dataset = self.tmp / "masks.h5"
        with __import__("h5py").File(dataset, "w") as hdf5:
            mask = hdf5.create_dataset("mask", shape=(1322, 128, 128, 1), dtype=np.uint8,
                                       chunks=(1, 128, 128, 1), compression="gzip")
            mask[1320:1322] = pixel_gt
        st = dataset.stat()

        source = np.ones((2, 32, 32), dtype=np.int32)
        source[:, :, 16:] = 2
        slot = np.ones((2, 32, 32), dtype=np.int32)
        slot[:, :, 16:] = 2
        candidate = source.copy()
        candidate[:, 8:24, 8:24] = 3
        local = source.copy()
        local[:, 4:28, 4:28] = 3
        ref_rows = {}
        for name, labels in (("source_T", source), ("SW0164", source), ("slot", slot)):
            pred = self.tmp / f"{name}.npz"
            proto = self.tmp / f"{name}.json"
            np.savez_compressed(pred, image_ids=ids, labels=labels)
            proto.write_text(json.dumps({"status": "complete", "seed": 1}), encoding="utf-8")
            ref_rows[name] = {"ids": ids, "labels": labels,
                "prediction_path": pred, "protocol_path": proto,
                "prediction_sha256": _sha(pred), "protocol_sha256": _sha(proto)}
        prepared = {"references": ref_rows, "dataset": dataset,
            "dataset_path": str(dataset.resolve()), "dataset_size_bytes": st.st_size,
            "dataset_mtime_ns": st.st_mtime_ns, "rgb_hashes": ["1"*64, "2"*64]}
        pfile = self.tmp / "sw164-score.json"
        with mock.patch.object(contract, "COUNT", 2), mock.patch.object(contract, "IMAGE_IDS", ids):
            sw164metrics = contract.production_metrics32(source, gt32)
        pfile.write_text(json.dumps({"scores": {"foreground_induced": sw164metrics}}), encoding="utf-8")
        outputs = {
            "structured_hinge": candidate,
            "local_margin": local,
            "source_T_control": source,
        }
        events = []

        def validate_prediction(arm, **kwargs):
            events.append("prediction:" + arm)
            labels = outputs[arm]
            return {"arm": arm, "ids": ids, "labels": labels,
                "prediction_sha256": hashlib.sha256(arm.encode()).hexdigest(),
                "protocol_sha256": hashlib.sha256((arm+"p").encode()).hexdigest()}

        real_modal = contract.modal_native32

        def modal(arr):
            events.append("ground_truth")
            return real_modal(arr)

        output = self.tmp / "score.json"
        with (mock.patch.object(pe, "COUNT", 2),
              mock.patch.object(pe, "IMAGE_IDS", ids),
              mock.patch.object(pe, "ARMS", ("structured_hinge", "local_margin", "source_T_control")),
              mock.patch.object(contract, "COUNT", 2),
              mock.patch.object(contract, "IMAGE_IDS", ids),
              mock.patch.object(pe, "BOOTSTRAP_REPLICATES", 32),
              mock.patch.object(pe, "validate_prediction", side_effect=validate_prediction),
              mock.patch.object(pe, "implementation_fingerprint", return_value={"fixture": "a"*64}),
              mock.patch.object(pe.sw164_eval, "SCORE_PATH", pfile),
              mock.patch.object(pe, "SW0164_SEED1_SCORE_SHA256", _sha(pfile)),
              mock.patch.object(contract, "modal_native32", side_effect=modal)):
            report = pe.score_all(output=output, prepared=prepared)
            first_gt = events.index("ground_truth")
            self.assertTrue(all(any(e == "prediction:" + arm for e in events[:first_gt])
                                for arm in pe.ARMS))
            self.assertEqual(report["scores"]["SW0164_seed1"]["fg_ari"]["mean"],
                             sw164metrics["fg_ari"]["mean"])
            valid = pe.validate_score(output, prepared=prepared)
            self.assertEqual(valid["status"], "valid")

            altered = json.loads(output.read_text(encoding="utf-8"))
            altered["means"]["structured_hinge"]["fg_ari"] += .1
            output.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                pe.validate_score(output, prepared=prepared)


if __name__ == "__main__":
    unittest.main()
