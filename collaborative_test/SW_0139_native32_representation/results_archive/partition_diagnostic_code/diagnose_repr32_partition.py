"""Read-only partition-error diagnostic for the completed SW0139 seed1 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# This diagnostic only reads saved predictions and targets. Prevent accidental
# CUDA context creation before importing the experiment modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
def resolve_repo_root(env=None, fallback=None):
    env = os.environ if env is None else env
    candidate = Path(env["PATCH_V2_REPO"]).expanduser() if env.get("PATCH_V2_REPO") else fallback
    if candidate is None:
        candidate = HERE.parents[1] / "patch_v2_sw"
    candidate = Path(candidate).resolve()
    if not (candidate / "collaborative_test/SW_0139_native32_representation/evaluate.py").is_file():
        raise FileNotFoundError(f"PATCH_V2_REPO is not the expected patch_v2_sw tree: {candidate}")
    return candidate


REPO = resolve_repo_root()
sys.path[:0] = [str(REPO), str(REPO / "collaborative_test"),
                str(REPO / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, modal_native32, production_metrics32,
    require_fixed_image_ids,
)
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0139_native32_representation import evaluate as sw139

SEED = 1
EVAL_DIR = (REPO / "collaborative_test/SW_0139_native32_representation"
            / "results_archive/evaluation_seed1")
SUMMARY = EVAL_DIR / "evaluation_control_context.json"
TRAINING_ROOT = sw139.run.OUT
DATASET = Path(sw139.DATASET)
OUTPUT = (REPO / "collaborative_test/SW_0139_native32_representation/results_archive"
          / "partition_diagnostic_seed1.json" if Path(__file__).name == "<stdin>"
          else HERE / "repr32_partition_diagnostic_seed1.json")
ARMS = ("control", "context_residual")


def _pairs(n):
    n = np.asarray(n, dtype=np.int64)
    return n * (n - 1) // 2


def partition_error(target, prediction):
    """Count pairwise merge/split errors over true-FG patches, including pred0."""
    gt = np.asarray(target)
    pred = np.asarray(prediction)
    if (gt.ndim != 2 or pred.shape != gt.shape or gt.size != GRID * GRID
            or not np.issubdtype(gt.dtype, np.integer)
            or not np.issubdtype(pred.dtype, np.integer)
            or np.issubdtype(gt.dtype, np.bool_) or np.issubdtype(pred.dtype, np.bool_)
            or not np.isfinite(gt).all() or not np.isfinite(pred).all()
            or np.any(gt < 0) or np.any(pred < 0)):
        raise ValueError("expected nonnegative integer native32 target and prediction labels")

    foreground = gt > 0
    true_labels, true_inverse = np.unique(gt[foreground], return_inverse=True)
    pred_labels, pred_inverse = np.unique(pred[foreground], return_inverse=True)
    table = np.bincount(true_inverse * len(pred_labels) + pred_inverse,
                        minlength=len(true_labels) * len(pred_labels)).reshape(
                            len(true_labels), len(pred_labels)).astype(np.int64)
    same_both = int(_pairs(table).sum())
    same_true = int(_pairs(table.sum(axis=1)).sum())
    same_pred = int(_pairs(table.sum(axis=0)).sum())
    all_pairs = int(_pairs(np.asarray([int(foreground.sum())])).item())
    merges = same_pred - same_both
    splits = same_true - same_both
    bg_index = np.flatnonzero(pred_labels == 0)
    bg_merge = 0
    bg_pairs = 0
    if bg_index.size:
        bg_column = table[:, int(bg_index[0])]
        bg_pairs = int(_pairs(np.asarray([int(bg_column.sum())])).item())
        bg_merge = bg_pairs - int(_pairs(bg_column).sum())

    pred_fg_labels = np.unique(pred[foreground & (pred > 0)])
    all_pred_fg_labels = np.unique(pred[pred > 0])
    true_objects = int(true_labels.size)
    missed = int(np.count_nonzero(foreground & (pred == 0)))
    n_foreground = int(foreground.sum())
    return {
        "true_foreground_patches": n_foreground,
        "true_object_count": true_objects,
        "predicted_foreground_object_count_on_true_fg_patches": int(pred_fg_labels.size),
        "all_predicted_nonzero_object_labels": int(all_pred_fg_labels.size),
        "predicted_object_count_including_background_label": int(pred_labels.size),
        "foreground_missed_patches": missed,
        "foreground_missed_fraction": (float(missed / n_foreground) if n_foreground else None),
        "true_fg_pair_count": all_pairs,
        "same_true_object_pairs_T": same_true,
        "same_predicted_label_pairs_P": same_pred,
        "same_true_and_predicted_pairs_S": same_both,
        "false_merge_pairs_P_minus_S": merges,
        "false_merge_fraction_of_all_true_fg_pairs": (float(merges / all_pairs) if all_pairs else None),
        "false_merge_fraction_of_P": (float(merges / same_pred) if same_pred else None),
        "false_split_pairs_T_minus_S": splits,
        "false_split_fraction_of_all_true_fg_pairs": (float(splits / all_pairs) if all_pairs else None),
        "false_split_fraction_of_T": (float(splits / same_true) if same_true else None),
        "predicted_background_column_pair_count": bg_pairs,
        "predicted_background_column_merge_pairs": bg_merge,
        "predicted_background_column_merge_fraction_of_all_true_fg_pairs": (
            float(bg_merge / all_pairs) if all_pairs else None),
        "contingency_true_object_rows_predicted_label_columns": table.tolist(),
        "contingency_true_object_labels": [int(v) for v in true_labels],
        "contingency_predicted_labels_including_zero": [int(v) for v in pred_labels],
    }


def _read_prediction_inputs(summary_path=SUMMARY, dataset=DATASET):
    """Validate all archived prediction/report bytes before opening target masks."""
    summary_path = Path(summary_path)
    summary_bytes = summary_path.read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    summary = json.loads(summary_bytes.decode("utf-8-sig"))
    if (summary.get("experiment") != "SW0139_native32_representation"
            or summary.get("status") not in {"partial_complete", "complete"}
            or summary.get("seed") != SEED
            or summary.get("count") != COUNT
            or summary.get("image_ids") != [IMAGE_IDS[0], IMAGE_IDS[-1]]
            or summary.get("ground_truth_used_for_prediction") is not False
            or summary.get("ground_truth_used_for_scoring") is not True
            or not set(ARMS).issubset(summary.get("valid_arms", []))):
        raise ValueError("SW0139 completed control/context summary contract is invalid")

    rows = {}
    for arm in ARMS:
        row = sw139.validate_arm_prediction(arm, seed=SEED, root=EVAL_DIR,
                                            training_root=TRAINING_ROOT)
        expected = summary.get("prediction_provenance", {}).get(arm)
        if row["protocol"].get("prediction_sha256") != expected:
            raise ValueError(f"SW0139 {arm} prediction hash differs from completed summary")
        if not np.array_equal(row["ids"], np.asarray(IMAGE_IDS, dtype=np.int64)):
            raise ValueError(f"SW0139 {arm} prediction IDs do not match fixed320 slice")
        rows[arm] = row

    source = sw137.validate_prediction(sw137._prediction_dir(SEED), seed=SEED)
    if (not np.array_equal(source["ids"], np.asarray(IMAGE_IDS, dtype=np.int64))
            or summary.get("source_prediction_sha256") != source["prediction_sha256"]):
        raise ValueError("SW0139 source97 prediction does not match saved summary provenance")

    # Check every saved production metric array and its mean before GT is used.
    for name, row in [("control", rows["control"]),
                      ("context_residual", rows["context_residual"]),
                      ("source97", source)]:
        saved_scores = summary.get("scores", {}).get(name)
        if not isinstance(saved_scores, dict):
            raise ValueError(f"completed summary lacks {name} metrics")
        for metric in METRICS:
            saved = saved_scores.get(metric)
            if (not isinstance(saved, dict) or saved.get("valid_count") != COUNT
                    or len(saved.get("per_image", [])) != COUNT
                    or not np.isfinite(np.asarray(saved.get("per_image"), dtype=np.float64)).all()):
                raise ValueError(f"completed summary has invalid {name}/{metric} array")

    # Hashes and report bytes are checked a second time at the GT boundary.
    frozen = {"summary_sha256": summary_sha,
              "source_prediction_sha256": source["prediction_sha256"],
              "source_protocol_sha256": source["protocol_sha256"],
              "arm_prediction_sha256": {arm: rows[arm]["prediction_sha256"] for arm in ARMS},
              "arm_protocol_sha256": {arm: rows[arm]["protocol_sha256"]
                                       for arm in ARMS}}
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != summary_sha:
        raise ValueError("SW0139 summary changed during validation")
    for arm, row in rows.items():
        if (sw139.run.sha256_file(row["prediction_path"]) != row["prediction_sha256"]
                or sw139.run.sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError(f"SW0139 {arm} predictions changed before target access")
    if (sw139.run.sha256_file(source["archive_path"]) != source["prediction_sha256"]
            or sw139.run.sha256_file(source["protocol_path"]) != source["protocol_sha256"]):
        raise ValueError("SW0139 source prediction changed before target access")

    with sw139.h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= IMAGE_IDS[-1]:
            raise ValueError("registered target masks do not contain the fixed320 IDs")
        target_pixels = np.asarray(h5["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    target = modal_native32(target_pixels)
    return summary, rows, source, target, frozen


def analyze(*, summary_path=SUMMARY, dataset=DATASET, output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing partition diagnostic: {output}")
    summary, rows, source, target, frozen = _read_prediction_inputs(summary_path, dataset)
    prediction_rows = {"source97": source["labels"],
                       "control": rows["control"]["labels"],
                       "context_residual": rows["context_residual"]["labels"]}
    per_image = {name: [partition_error(target[i], labels[i]) for i in range(COUNT)]
                 for name, labels in prediction_rows.items()}
    fields = (
        "true_object_count", "predicted_foreground_object_count_on_true_fg_patches",
        "all_predicted_nonzero_object_labels",
        "predicted_object_count_including_background_label", "foreground_missed_fraction",
        "false_merge_pairs_P_minus_S", "false_merge_fraction_of_all_true_fg_pairs",
        "false_merge_fraction_of_P", "false_split_pairs_T_minus_S",
        "false_split_fraction_of_all_true_fg_pairs", "false_split_fraction_of_T",
        "predicted_background_column_pair_count", "predicted_background_column_merge_pairs",
        "predicted_background_column_merge_fraction_of_all_true_fg_pairs",
    )
    means = {}
    for name, records in per_image.items():
        means[name] = {}
        for field in fields:
            values = [row[field] for row in records if row[field] is not None]
            means[name][field] = {"mean": float(np.mean(values)) if values else None,
                                  "valid_images": len(values)}
        means[name]["summary_metric_mean_recheck"] = {}
        scores = summary["scores"][name]
        for metric in METRICS:
            actual = production_metrics32(prediction_rows[name], target)[metric]
            reported = scores[metric]
            if (not np.allclose(actual["per_image"], reported["per_image"], rtol=0, atol=1e-12)
                    or abs(actual["mean"] - float(reported["mean"])) > 1e-12):
                raise ValueError(f"saved SW0139 {name}/{metric} scores fail production-metric replay")
            means[name]["summary_metric_mean_recheck"][metric] = actual["mean"]

    paired = {}
    for name in ("control", "context_residual"):
        paired[name] = {}
        for field in fields:
            left = np.asarray([r[field] if r[field] is not None else np.nan
                               for r in per_image[name]], dtype=np.float64)
            right = np.asarray([r[field] if r[field] is not None else np.nan
                                for r in per_image["source97"]], dtype=np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            delta = left[valid] - right[valid]
            paired[name][field] = {"mean_delta_vs_source97": float(delta.mean()) if delta.size else None,
                                   "valid_paired_images": int(delta.size)}
    report = {
        "status": "complete", "diagnostic": "SW0139 seed1 true-FG partition error decomposition",
        "seed": SEED, "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [GRID, GRID], "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True, "training_or_inference_updates": 0,
        "input_summary_sha256": frozen["summary_sha256"], "frozen_prediction_hashes": frozen,
        "source97_means": means["source97"], "control_means": means["control"],
        "context_residual_means": means["context_residual"],
        "paired_deltas_vs_source97": paired,
        "per_image": per_image,
        "definitions": {
            "P": "all true-FG patch pairs with the same predicted label, including predicted label 0",
            "T": "all true-FG patch pairs with the same true object label",
            "S": "true-FG patch pairs sharing both their true object and predicted label",
            "false_merge": "P-S; predicted-background-column merge pairs are also broken out",
            "false_split": "T-S; pairwise partition counts are label-permutation invariant",
            "scope": "ancillary error decomposition, not a new evaluation contract or causal-layer proof",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = analyze(summary_path=args.summary, dataset=args.dataset, output=args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "seed": SEED, "count": COUNT}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
