"""Versioned SW176 GT-last scorer: Source-T parity is conditional-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from collaborative_test.SW_0176_local_phase_credit import predict_evaluate as frozen

HERE = Path(__file__).resolve().parent
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_v2_20261016.json"
EXPERIMENT = "SW0176_local_phase_fixed320_score_v2"
QUEUE_EXPERIMENT = "SW0176_score_queue_v2"


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    queue_path = HERE / "score_queue_v2.py"
    if not queue_path.is_file():
        raise FileNotFoundError(queue_path)
    return {
        "frozen_prediction_implementation": frozen.implementation_fingerprint(),
        "score_v2_sha256": _sha_file(Path(__file__)),
        "score_queue_v2_sha256": _sha_file(queue_path),
    }


def _score_fixed320_core(*, prediction_root, work_root, prepared):
    # This public frozen validator checks both endpoint schemas and the
    # conditional Source-T foreground contract, before any GT access.
    candidate = frozen.validate_prediction(frozen.train.ARM,
        prediction_root=prediction_root, work_root=work_root, prepared=prepared)
    refs = prepared["references"]
    frozen._reference_hashes(refs)
    if not np.array_equal(candidate["labels"]["conditional"] > 0,
                          refs["source_T"]["labels"] > 0):
        raise ValueError("SW0176 conditional readout changed fixed Source-T foreground membership")
    sw164_path = Path(frozen.sw164_eval.SCORE_PATH)
    sw175_path = Path(frozen.eval175.SCORE_PATH)
    if (_sha_file(sw164_path) != frozen.SW0164_SCORE_SHA256
            or _sha_file(sw175_path) != frozen.SW0175_SCORE_SHA256):
        raise ValueError("fixed320 immutable SW0164/SW0175 score receipt changed before GT")
    sw164_report = json.loads(sw164_path.read_text(encoding="utf-8-sig"))
    sw175_report = json.loads(sw175_path.read_text(encoding="utf-8-sig"))

    before = prepared["dataset"].stat()
    if (before.st_size != prepared["dataset_size_bytes"]
            or before.st_mtime_ns != prepared["dataset_mtime_ns"]):
        raise ValueError("canonical HDF5 identity changed before GT scoring")
    with frozen.h5py.File(prepared["dataset"], "r") as hdf5:
        if "mask" not in hdf5 or hdf5["mask"].shape[0] <= int(frozen.IMAGE_IDS[-1]):
            raise ValueError("canonical HDF5 validation mask range is missing")
        raw = np.asarray(hdf5["mask"][int(frozen.IMAGE_IDS[0]):int(frozen.IMAGE_IDS[-1]) + 1])
    gt = frozen.contract.modal_native32(raw)

    label_sets = {
        "SW0164": refs["SW0164"]["labels"],
        "Source-T": refs["source_T"]["labels"],
        "SW0175_critical_component_conditional": refs["SW0175_control"]["labels"]["conditional"],
        "SW0175_critical_component_standalone": refs["SW0175_control"]["labels"]["standalone"],
        "local_phase_conditional": candidate["labels"]["conditional"],
        "local_phase_standalone": candidate["labels"]["standalone"],
    }
    scores = frozen._score_metrics(label_sets, gt)
    for metric in frozen.contract.METRICS:
        if frozen._json_sha(scores["SW0164"][metric]) != frozen._json_sha(
                sw164_report["scores"]["foreground_induced"][metric]):
            raise ValueError(f"SW0164 reference metrics fail production recomputation: {metric}")
        for endpoint in frozen.ENDPOINTS:
            key = f"SW0175_critical_component_{endpoint}"
            expected = sw175_report["scores"][f"critical_component_{endpoint}"][metric]
            if frozen._json_sha(scores[key][metric]) != frozen._json_sha(expected):
                raise ValueError(f"SW0175 matched control metrics fail production recomputation: {endpoint}/{metric}")

    # The conditional endpoint is explicitly constrained to Source-T. The
    # standalone endpoint is a native-QCC partition and is scored independently.
    if not np.array_equal(scores["local_phase_conditional"]["foreground_iou"]["per_image"],
                          scores["Source-T"]["foreground_iou"]["per_image"]):
        raise ValueError("SW0176 conditional readout changed fixed Source-T foreground IoU")

    paired, gates = {}, {}
    for endpoint in frozen.ENDPOINTS:
        cand = scores[f"local_phase_{endpoint}"]
        comparisons = {
            "SW0164": scores["SW0164"],
            "SW0175_critical_component": scores[f"SW0175_critical_component_{endpoint}"],
        }
        for name, baseline in comparisons.items():
            key = f"{endpoint}_local_phase_minus_{name}_fg_ari"
            paired[key] = frozen._bootstrap(cand["fg_ari"]["per_image"],
                                            baseline["fg_ari"]["per_image"])
            gates[f"{endpoint}_fg_ari_mean_above_{name}"] = cand["fg_ari"]["mean"] > baseline["fg_ari"]["mean"]
            gates[f"{endpoint}_fg_ari_ci95_lower_above_zero_{name}"] = paired[key]["ci95"][0] > 0
            gates[f"{endpoint}_foreground_iou_not_below_{name}"] = (
                cand["foreground_iou"]["mean"] >= baseline["foreground_iou"]["mean"])
            gates[f"{endpoint}_matched_object_iou_not_below_{name}"] = (
                cand["matched_object_iou"]["mean"] >= baseline["matched_object_iou"]["mean"])
    gates["conditional_foreground_membership_exact_Source_T"] = True
    gates["co_primary_gate_passed"] = all(gates.values())

    frozen._reference_hashes(refs)
    if (_sha_file(sw164_path) != frozen.SW0164_SCORE_SHA256
            or _sha_file(sw175_path) != frozen.SW0175_SCORE_SHA256):
        raise ValueError("immutable baseline score changed during SW0176 scoring")
    after = prepared["dataset"].stat()
    if after.st_size != prepared["dataset_size_bytes"] or after.st_mtime_ns != prepared["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 changed during SW0176 scoring")
    fresh = frozen.validate_prediction(frozen.train.ARM,
        prediction_root=prediction_root, work_root=work_root, prepared=prepared)
    if fresh["prediction_sha256"] != candidate["prediction_sha256"]:
        raise ValueError("SW0176 fixed320 prediction changed during GT scoring")
    return candidate, scores, paired, gates


def score_all(*, output=SCORE_PATH, prediction_root=frozen.PREDICTION_ROOT,
              work_root=frozen.WORK_ROOT, prepared=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    p = prepared or frozen.prepare_prediction_inputs_cpu()
    candidate, scores, paired, gates = _score_fixed320_core(
        prediction_root=prediction_root, work_root=work_root, prepared=p)
    report = {
        "experiment": EXPERIMENT, "status": "complete", "seed_scope": [frozen.SEED],
        "endpoint_scope": list(frozen.ENDPOINTS),
        "implementation_fingerprint": implementation_fingerprint(),
        "prediction_sha256": candidate["prediction_sha256"],
        "prediction_protocol_sha256": candidate["protocol_sha256"],
        "reference_prediction_sha256": {k: v["prediction_sha256"] for k, v in p["references"].items()},
        "reference_protocol_sha256": {k: v["protocol_sha256"] for k, v in p["references"].items()},
        "SW0164_score_sha256": frozen.SW0164_SCORE_SHA256,
        "SW0175_score_sha256": frozen.SW0175_SCORE_SHA256,
        "matched_control_training_manifest_sha256": (
            p["manifests"][frozen.train.ARM]["bindings"]["control_receipt"]["manifest_sha256"]),
        "scores": scores, "paired_bootstrap": paired, "gate": gates,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "ground_truth_scope": "fixed320 canonical validation modal-native32 masks",
        "dataset_path": p["dataset_path"], "dataset_size_bytes": p["dataset_size_bytes"],
        "dataset_mtime_ns": p["dataset_mtime_ns"],
        "bootstrap_seed": frozen.BOOTSTRAP_SEED,
        "bootstrap_replicates": frozen.BOOTSTRAP_REPLICATES,
    }
    frozen._atomic_json(output, report)
    validate_score(output, prediction_root=prediction_root, work_root=work_root, prepared=p)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=frozen.PREDICTION_ROOT,
                   work_root=frozen.WORK_ROOT, prepared=None):
    p = prepared or frozen.prepare_prediction_inputs_cpu()
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("experiment") != EXPERIMENT or report.get("status") != "complete"
            or report.get("seed_scope") != [frozen.SEED]
            or report.get("implementation_fingerprint") != implementation_fingerprint()
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("SW0176 v2 score provenance or GT scope is invalid")
    candidate, scores, paired, gates = _score_fixed320_core(
        prediction_root=prediction_root, work_root=work_root, prepared=p)
    expected = {
        "prediction_sha256": candidate["prediction_sha256"],
        "prediction_protocol_sha256": candidate["protocol_sha256"],
        "reference_prediction_sha256": {k: v["prediction_sha256"] for k, v in p["references"].items()},
        "reference_protocol_sha256": {k: v["protocol_sha256"] for k, v in p["references"].items()},
        "SW0164_score_sha256": frozen.SW0164_SCORE_SHA256,
        "SW0175_score_sha256": frozen.SW0175_SCORE_SHA256,
        "matched_control_training_manifest_sha256": (
            p["manifests"][frozen.train.ARM]["bindings"]["control_receipt"]["manifest_sha256"]),
        "scores": scores, "paired_bootstrap": paired, "gate": gates,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("SW0176 v2 independent score recomputation differs from report")
    return report
