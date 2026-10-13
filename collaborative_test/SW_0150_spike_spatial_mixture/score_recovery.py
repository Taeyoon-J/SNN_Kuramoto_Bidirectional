"""CPU-only recovery scorer for the frozen SW0150 prediction artifacts.

This preserves the original score calculation and prediction validators. It
only accepts the exact registered SW0135 baseline schema omitted by the
original scorer, and writes to a separate create-once recovery output.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0150_spike_spatial_mixture import evaluate as sealed


EXPECTED_EVALUATE_SHA256 = "2941bdb9aa0c14d207317c1234347fc44d730fea253de61b7aafda61b331bfaa"
RECOVERY_OUTPUT = sealed.HERE / "results_archive" / "evaluation_seed1_recovery_20261013.json"
BASELINE_CONTRACT = {
    "image_ids": [1320, 1639, 320],
    "native_image_size": [128, 128],
    "patch_grid": [32, 32],
    "patch_pixels": 4,
    "metrics": list(contract.METRICS),
    "native_prediction": "direct32 labels; no conversion from 16-grid outputs",
    "slot_prediction": "saved native128 labels modal-pooled by production4x4 converter; existing perimeter-background remap preserved",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_json_schema(actual, expected) -> bool:
    """Compare exact JSON structure and scalar types (bool must not equal int)."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_same_json_schema(actual[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_same_json_schema(left, right)
                        for left, right in zip(actual, expected)))
    return actual == expected


def _validate_baseline_contract(baseline: dict) -> dict[str, float]:
    if (baseline.get("status") != "complete"
            or baseline.get("comparison") != "slot32 baseline only"
            or not _same_json_schema(baseline.get("evaluation_contract"), BASELINE_CONTRACT)):
        raise ValueError("registered Slot32 baseline contract/schema changed")
    try:
        means = {metric: float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
                 for metric in contract.METRICS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("registered Slot32 baseline metrics are incomplete") from exc
    if any(not np.isfinite(value) for value in means.values()):
        raise ValueError("registered Slot32 comparison means are nonfinite")
    return means


def _registered_baseline() -> tuple[dict, dict[str, float]]:
    if sha256_file(sealed.SLOT_BASELINE) != sealed.SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 three-seed baseline changed")
    baseline = json.loads(sealed.SLOT_BASELINE.read_text(encoding="utf-8"))
    return baseline, _validate_baseline_contract(baseline)


def score_all(*, output=RECOVERY_OUTPUT, dataset=sealed.DATASET,
              prediction_root=sealed.PREDICTION_ROOT, training_root=sealed.train.TRAIN):
    """Score frozen predictions with the original GT-last contract."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0150 recovery score: {output}")
    if sha256_file(sealed.HERE / "evaluate.py") != EXPECTED_EVALUATE_SHA256:
        raise ValueError("sealed SW0150 scorer bytes differ from the reviewed original")

    # Validate all candidate and reference predictions before opening any masks.
    own = {name: sealed.validate_prediction(name, prediction_root=prediction_root,
                                            training_root=training_root)
           for name in (*sealed.ARMS, "source_p")}
    source_qcc = sealed.sw137.validate_prediction(
        sealed.sw137.OUTPUT_ROOT / "seed1", seed=sealed.SEED)
    slot = sealed._slot_reference()
    if not all(np.array_equal(own[name]["ids"], sealed.IMAGE_IDS) for name in own):
        raise ValueError("SW0150 prediction IDs do not exactly match the fixed320 range")
    if (not np.array_equal(source_qcc["ids"], sealed.IMAGE_IDS)
            or not np.array_equal(slot["ids"], sealed.IMAGE_IDS)
            or not np.array_equal(own["source_p"]["qcc_labels"], source_qcc["labels"])):
        raise ValueError("SW0150/source97/Slot fixed320 prediction arrays are not aligned")

    baseline, slot_means = _registered_baseline()
    paths = [row[key] for row in own.values()
             for key in ("archive_path", "protocol_path")]
    paths.extend((source_qcc["archive_path"], source_qcc["protocol_path"],
                  slot["npz"], slot["protocol_path"], sealed.SLOT_BASELINE))
    frozen_hashes = {str(path): sha256_file(path) for path in paths}
    if (any(frozen_hashes[str(row["archive_path"])] != row["prediction_sha256"]
            or frozen_hashes[str(row["protocol_path"])] != row["protocol_sha256"]
            for row in own.values())
            or frozen_hashes[str(source_qcc["archive_path"])] != source_qcc["prediction_sha256"]
            or frozen_hashes[str(source_qcc["protocol_path"])] != source_qcc["protocol_sha256"]
            or frozen_hashes[str(slot["npz"])] != slot["prediction_sha256"]
            or frozen_hashes[str(slot["protocol_path"])] != slot["protocol_sha256"]
            or frozen_hashes[str(sealed.SLOT_BASELINE)] != sealed.SLOT_BASELINE_SHA256):
        raise ValueError("a frozen SW0150/source/Slot prediction changed before target access")

    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= int(sealed.IMAGE_IDS[-1]):
            raise ValueError("registered HDF5 native128 target masks lack fixed320 range")
        native_masks = np.asarray(
            h5["mask"][int(sealed.IMAGE_IDS[0]):int(sealed.IMAGE_IDS[-1]) + 1])
    target = contract.modal_native32(native_masks)
    scores = {
        "encoder_graph": {
            "primary": contract.production_metrics32(own["encoder_graph"]["primary_labels"], target),
            "qcc": contract.production_metrics32(own["encoder_graph"]["qcc_labels"], target)},
        "joint": {
            "primary": contract.production_metrics32(own["joint"]["primary_labels"], target),
            "qcc": contract.production_metrics32(own["joint"]["qcc_labels"], target)},
        "source_p": {"primary": contract.production_metrics32(
            own["source_p"]["primary_labels"], target)},
        "source97_qcc": {"qcc": contract.production_metrics32(source_qcc["labels"], target)},
        "slot32_seed1": {"slot32": contract.production_metrics32(slot["labels"], target)},
    }
    metric_means = lambda result: {metric: float(result[metric]["mean"])
                                   for metric in contract.METRICS}
    means = {name: {readout: metric_means(values)
                    for readout, values in readouts.items()}
             for name, readouts in scores.items()}
    draws = np.random.RandomState(139).randint(
        0, sealed.COUNT, size=(10000, sealed.COUNT))
    paired = {}
    for reference, score_key, readout in (("source_qcc", "source97_qcc", "primary"),
                                          ("control_primary", "encoder_graph", "primary")):
        reference_readout = "qcc" if reference == "source_qcc" else readout
        paired[f"joint_primary_minus_{reference}"] = {
            metric: sealed._bootstrap_delta(
                scores["joint"]["primary"][metric]["per_image"],
                scores[score_key][reference_readout][metric]["per_image"], draws)
            for metric in contract.METRICS}
    for arm in sealed.ARMS:
        paired[f"{arm}_primary_minus_source_primary"] = {
            metric: sealed._bootstrap_delta(
                scores[arm]["primary"][metric]["per_image"],
                scores["source_p"]["primary"][metric]["per_image"], draws)
            for metric in contract.METRICS}
    paired["source_primary_minus_source_qcc"] = {
        metric: sealed._bootstrap_delta(
            scores["source_p"]["primary"][metric]["per_image"],
            scores["source97_qcc"]["qcc"][metric]["per_image"], draws)
        for metric in contract.METRICS}
    gate = {
        "joint_primary_fg_at_least_source_qcc_plus_0_01":
            means["joint"]["primary"]["fg_ari"] >= means["source97_qcc"]["qcc"]["fg_ari"] + 0.01,
        "joint_primary_fg_at_least_encoder_graph_primary_plus_0_01":
            means["joint"]["primary"]["fg_ari"] >= means["encoder_graph"]["primary"]["fg_ari"] + 0.01,
        "joint_source_qcc_fg_ci_lower_positive":
            paired["joint_primary_minus_source_qcc"]["fg_ari"]["ci95"][0] > 0,
        "joint_control_fg_ci_lower_positive":
            paired["joint_primary_minus_control_primary"]["fg_ari"]["ci95"][0] > 0,
        "joint_primary_foreground_iou_above_slot_three_seed_plus_0_05":
            means["joint"]["primary"]["foreground_iou"] > slot_means["foreground_iou"] + 0.05,
        "joint_primary_matched_object_iou_above_slot_three_seed_plus_0_05":
            means["joint"]["primary"]["matched_object_iou"] > slot_means["matched_object_iou"] + 0.05,
    }
    report = {
        "experiment": sealed.NAME, "status": "complete", "seed_scope": [sealed.SEED],
        "scores": scores, "means": means,
        "registered_slot32_three_seed_means": slot_means,
        "paired_bootstrap": paired,
        "bootstrap": {"samples": 10000, "rng": "numpy RandomState", "seed": 139,
                      "common_image_indices": True},
        "promotion_gate": gate, "may_promote_seed_expansion": all(gate.values()),
        "prediction_provenance": {
            name: {"prediction_sha256": row["prediction_sha256"],
                  "protocol_sha256": row["protocol_sha256"],
                  "training_manifest_sha256": row["protocol"].get("training_manifest_sha256"),
                  "checkpoint_sha256": row["protocol"].get("checkpoint_sha256")}
            for name, row in own.items()},
        "reference_provenance": {
            "source97_qcc_prediction_sha256": source_qcc["prediction_sha256"],
            "source97_qcc_protocol_sha256": source_qcc["protocol_sha256"],
            "slot32_seed1_prediction_sha256": slot["prediction_sha256"],
            "slot32_seed1_protocol_sha256": slot["protocol_sha256"],
            "slot32_three_seed_baseline_sha256": sealed.SLOT_BASELINE_SHA256},
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "limits": "Fixed seed1 native32 pilot only; no fresh 70k native32 training, scaling, or universal superiority claim.",
        "score_recovery_provenance": {
            "reason": "Original score stopped before HDF5 access because its exact baseline-contract schema omitted two registered descriptive fields.",
            "baseline_contract_sha256": sha256_file(sealed.SLOT_BASELINE),
            "sealed_original_evaluate_sha256": EXPECTED_EVALUATE_SHA256,
            "recovery_module_sha256": sha256_file(Path(__file__)),
            "score_algorithm": "copied unchanged from frozen SW0150 evaluate.score_all",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECOVERY_OUTPUT)
    args = parser.parse_args(argv)
    report = score_all(output=args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "may_promote_seed_expansion": report["may_promote_seed_expansion"]},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
