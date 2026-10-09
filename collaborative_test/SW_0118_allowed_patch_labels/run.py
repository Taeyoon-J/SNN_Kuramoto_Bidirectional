"""CPU-only scorer for frozen SW0097 and SW0092 validation predictions.

All prediction and provenance checks run before this program opens validation
ground-truth masks. This is a GT-conditioned diagnostic, not model evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np  # Import NumPy before torch in the Windows environment.
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0118_allowed_patch_labels.scorer import score_allowed_patch_labels
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks

DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
IDS = tuple(range(1320, 1640))
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SOURCE_SHA = {
    0: "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf",
    1: "76f379d5a7e4cd9d12fdf0b701f3dd9dbbf28b5eea270a9cee2d30d87b4dad98",
    2: "798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661",
}
OURS_PREDICTION_SHA = {
    0: None,  # Bound through the immutable SW0114 report's frozen_predictions_sha256.
    1: "2eb0407714549a7d09d6bb55925095a8385537da8de9a3352be55316ce409225",
    2: "da82e728097eef6266bada9d37dd4aa4454479c6d658f35a703d3ad4bfebcb20",
}
VAL_GAMMA_SHA = "f20b019bd46adb378b2bfcecbb252d960da2592ef651dcd419753034c8b705f0"
SLOT_PREDICTION_SHA = {
    0: "ef1dce0e16bfe0f9340f9c157b0a1400320c7dc2479e4ef68b20a65437f0108b",
    1: "f5c80fceb2506d5356c3aa23ac39556448ebcb4f9454eeebabdc2ef148b1fcbd",
    2: "c23fe2f5b97fe0f2db74c1d39365619bb934d2e7765e73c672a31475acda5ee3",
}
SLOT_CHECKPOINT_SHA = {
    0: "0301a456a586da5e1132c703d095f4736c4351cb86388eabe2aff7f1c96ae091",
    1: "3afadf5958d494241763dcc7ef0600bbeb78611d8489cb4126fce462bf0992b9",
    2: "c665b2b52cb59ee5044b91cebff039a4b5199fa3edc2b5fa769094fcadadb200",
}
RELEASED_SLOT_PREDICTION_SHA = "81c71e590090609a9c339ec362ee32dd0dafc72d36c104c30da429b27ece19fd"
RELEASED_SLOT_MEAN = {
    "fg_ari": 0.8946414574875314,
    "foreground_iou": 0.22351117892035982,
    "matched_object_iou": 0.24596825542415796,
}
SW114_SEED0_REPORT_SHA = "f7798077a44267d11813e7d93de3efc64e9bd82b1428628a607f0dfe8e661687"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256_literal(value: str, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Registered {name} must be a 64-character lowercase SHA256")
    return value


def validate_registered_hashes() -> None:
    for seed, value in SOURCE_SHA.items():
        validate_sha256_literal(value, f"SW0097 seed{seed} core SHA")
    for seed, value in OURS_PREDICTION_SHA.items():
        if value is not None:
            validate_sha256_literal(value, f"ours seed{seed} prediction SHA")
    for seed, value in SLOT_PREDICTION_SHA.items():
        validate_sha256_literal(value, f"matched Slot seed{seed} prediction SHA")
    for seed, value in SLOT_CHECKPOINT_SHA.items():
        validate_sha256_literal(value, f"matched Slot seed{seed} checkpoint SHA")
    validate_sha256_literal(VAL_GAMMA_SHA, "validation gamma SHA")
    validate_sha256_literal(RELEASED_SLOT_PREDICTION_SHA, "released Slot prediction SHA")
    validate_sha256_literal(SW114_SEED0_REPORT_SHA, "SW114 seed0 report SHA")


def _tensor_labels(value, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must contain integer labels")
        result = value.to(torch.int64).clone()
    else:
        array = np.asarray(value)
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer labels")
        result = torch.as_tensor(array, dtype=torch.int64).clone()
    if result.shape == (320, 1, 16, 16):
        result = result[:, 0]
    if tuple(result.shape) != (320, 16, 16):
        raise ValueError(f"{name} must have shape [320,16,16], got {tuple(result.shape)}")
    if bool((result < 0).any()):
        raise ValueError(f"{name} has negative labels")
    return result


def _ids(value, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    result = [int(x) for x in value]
    if result != list(IDS):
        raise ValueError(f"{name} must equal the ordered fixed IDs 1320..1639")
    return result


def _load_pt(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def _reference_per_image(path: Path, seed: int) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ids") != [1320, 1639] or payload.get("images") != 320:
        raise ValueError(f"SW0097 seed{seed} source report does not cover the fixed 320 IDs")
    scored = payload["sweep"][0]["scored_targets"]["our_hdf5"]
    for metric in METRICS:
        values = np.asarray(scored["per_image"][metric], dtype=np.float64)
        if values.shape != (320,) or not np.isfinite(values).all():
            raise ValueError(f"SW0097 seed{seed} {metric} must have 320 finite values")
        if int(scored["valid_count"][metric]) != 320:
            raise ValueError(f"SW0097 seed{seed} {metric} valid_count is not 320")
        mean = float(scored["metrics"][metric])
        if not math.isfinite(mean) or abs(float(values.mean()) - mean) > 1e-10:
            raise ValueError(f"SW0097 seed{seed} {metric} mean does not match per-image values")
    return scored, sha256_file(path)


def _load_ours(seed: int) -> tuple[torch.Tensor, dict]:
    if seed == 0:
        folder = HERE.parent / "SW_0114_resolution32_pilot" / "results_archive" / "evaluation_seed0"
        prediction_path = folder / "frozen_predictions.pt"
        report_path = folder / "evaluation.json"
        if sha256_file(report_path) != SW114_SEED0_REPORT_SHA:
            raise ValueError("SW0114 seed0 evaluation report SHA mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete" or report.get("seed") != 0
                or report.get("ids") != [1320, 1639] or report.get("count") != 320
                or report.get("ground_truth_used_during_prediction") is not False
                or report.get("source_checkpoint_sha256") != SOURCE_SHA[0]
                or report.get("source_baseline_equivalence", {}).get("pass") is not True):
            raise ValueError("SW0114 seed0 report does not prove registered fixed-source baseline")
        if sha256_file(prediction_path) != report.get("frozen_predictions_sha256"):
            raise ValueError("SW0114 seed0 frozen prediction SHA mismatch")
        blob = _load_pt(prediction_path)
        if "source" not in blob or not isinstance(blob["source"], dict):
            raise ValueError("SW0114 seed0 predictions lack source labels")
        labels = _tensor_labels(blob["source"].get("labels"), "SW0114 seed0 source labels")
        meta = {"provider": "SW0114 frozen source prediction", "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path), "report_path": str(report_path),
                "report_sha256": SW114_SEED0_REPORT_SHA,
                "checkpoint_sha256": report["source_checkpoint_sha256"],
                "gamma_sha256": report.get("registered_native16_gamma_sha256"),
                "ground_truth_used_for_prediction": False}
        return labels, meta

    path = HERE / "results_archive" / f"frozen_our_seed{seed}.pt"
    if sha256_file(path) != OURS_PREDICTION_SHA[seed]:
        raise ValueError(f"SW0118 seed{seed} frozen prediction SHA mismatch")
    blob = _load_pt(path)
    if blob.get("seed") != seed or blob.get("source") != "SW0097_positive_frozen":
        raise ValueError(f"SW0118 seed{seed} frozen prediction provenance mismatch")
    if blob.get("checkpoint_sha256") != SOURCE_SHA[seed] or blob.get("gamma_sha256") != VAL_GAMMA_SHA:
        raise ValueError(f"SW0118 seed{seed} source or gamma SHA mismatch")
    if blob.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"SW0118 seed{seed} prediction used ground truth")
    image_ids = _ids(blob.get("image_ids", []), f"SW0118 seed{seed} image_ids")
    labels = _tensor_labels(blob.get("prediction"), f"SW0118 seed{seed} prediction")
    return labels, {"provider": "frozen SW0097 source prediction", "prediction_path": str(path),
                    "prediction_sha256": sha256_file(path), "checkpoint_sha256": blob["checkpoint_sha256"],
                    "gamma_sha256": blob["gamma_sha256"], "image_ids": [image_ids[0], image_ids[-1]],
                    "ground_truth_used_for_prediction": False}


def _load_slot(seed: int) -> tuple[torch.Tensor, dict, dict]:
    folder = ROOT / "trained_models" / "SW0092_slot_our70000_eval" / f"seed{seed}_epoch10"
    path = folder / "patch_masks.pt"
    if sha256_file(path) != SLOT_PREDICTION_SHA[seed]:
        raise ValueError(f"SW0092 seed{seed} frozen prediction SHA mismatch")
    blob = _load_pt(path)
    if blob.get("patch_size") != 8:
        raise ValueError(f"SW0092 seed{seed} patch size must be 8")
    ids = _ids(blob.get("image_ids", []), f"SW0092 seed{seed} image_ids")
    labels = _tensor_labels(blob.get("prediction"), f"SW0092 seed{seed} prediction")

    slot_root = ROOT / "trained_models" / f"SW0092_slot_our70000_s{seed}{'_final' if seed == 0 else ''}"
    training_path = slot_root / "training_protocol.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (training.get("seed") != seed or training.get("unique_training_images") != 70000
            or training.get("epochs") != 10
            or training.get("training_id_segments_inclusive") != [[0, 999], [1640, 70639]]
            or training.get("ground_truth_used_for_training") is not False):
        raise ValueError(f"SW0092 seed{seed} training protocol mismatch")
    summary_path = folder / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})
    if (protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != 320
            or protocol.get("seed") != seed or protocol.get("checkpoint_sha256") != SLOT_CHECKPOINT_SHA[seed]
            or protocol.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"SW0092 seed{seed} evaluation protocol mismatch")
    if summary.get("patch_size") != 8:
        raise ValueError(f"SW0092 seed{seed} evaluation patch size mismatch")
    score = summary.get("scores", {})
    means = score.get("mean", {})
    valid = score.get("valid_count", {})
    for metric in METRICS:
        mean = float(means.get(metric, float("nan")))
        if not math.isfinite(mean) or int(valid.get(metric, -1)) != 320:
            raise ValueError(f"SW0092 seed{seed} {metric} summary mean/count invalid")
    metadata = {"provider": "SW0092 frozen Slot prediction", "prediction_path": str(path),
                "prediction_sha256": sha256_file(path), "checkpoint_sha256": SLOT_CHECKPOINT_SHA[seed],
                "training_protocol_sha256": sha256_file(training_path),
                "evaluation_summary_sha256": sha256_file(summary_path),
                "image_ids": [ids[0], ids[-1]], "ground_truth_used_for_prediction": False}
    return labels, metadata, {"mean": means, "valid_count": valid}


def _validate_released_slot_protocol(protocol: dict) -> None:
    if (protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != 320
            or protocol.get("seed") != 0 or protocol.get("batch_size") != 1
            or protocol.get("num_slots") != 11 or protocol.get("iterations") != 3
            or protocol.get("checkpoint_source") != "gs://gresearch/slot-attention/object-discovery/ckpt-500"
            or protocol.get("checkpoint_prefix") != "/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/checkpoint/ckpt-500"
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("model_sha256") != "96c2b12d8b28c22fd2605eccf9027bda304f38bb9332f8c1620898f471c67ec4"):
        raise ValueError("Released Slot inference protocol mismatch")


def _load_released_slot() -> tuple[torch.Tensor, dict]:
    folder = ROOT / "collaborative_test" / "SW_0046_aligned_slot_audit" / "results" / "seed0_validation1320_1639"
    path = folder / "patch_masks.pt"
    protocol_path = folder / "protocol.json"
    summary_path = folder / "evaluation_summary.json"
    if sha256_file(path) != RELEASED_SLOT_PREDICTION_SHA:
        raise ValueError("Released Slot frozen prediction SHA mismatch")
    blob = _load_pt(path)
    if blob.get("patch_size") != 8:
        raise ValueError("Released Slot patch size must be 8")
    ids = _ids(blob.get("image_ids", []), "Released Slot image_ids")
    labels = _tensor_labels(blob.get("prediction"), "Released Slot prediction")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_released_slot_protocol(protocol)
    score = summary.get("scores", {})
    for metric in METRICS:
        mean = float(score.get("mean", {}).get(metric, float("nan")))
        if (int(score.get("valid_count", {}).get(metric, -1)) != 320
                or not math.isfinite(mean) or abs(mean - RELEASED_SLOT_MEAN[metric]) > 1e-10):
            raise ValueError(f"Released Slot {metric} summary does not match frozen reference")
    metadata = {"provider": "released Slot Attention checkpoint (separate training data)",
                "prediction_path": str(path), "prediction_sha256": RELEASED_SLOT_PREDICTION_SHA,
                "protocol_sha256": sha256_file(protocol_path), "summary_sha256": sha256_file(summary_path),
                "image_ids": [ids[0], ids[-1]], "checkpoint_source": protocol.get("checkpoint_source"),
                "checkpoint_prefix": protocol.get("checkpoint_prefix"),
                "model_sha256": protocol.get("model_sha256"),
                "ground_truth_used_for_prediction": False,
                "training_comparability": "single released checkpoint; different training data; not a matched 70k mean"}
    return labels, metadata


def _score_standard(pred: torch.Tensor, target: torch.Tensor) -> dict:
    scores = evaluate_patch_masks(pred, target)
    result = {}
    for metric in METRICS:
        values = scores["per_image"][metric].detach().cpu().numpy().astype(np.float64)
        finite = np.isfinite(values)
        result[metric] = {"per_image": [float(v) if math.isfinite(float(v)) else None for v in values],
                          "mean": float(values[finite].mean()) if finite.any() else None,
                          "valid_count": int(finite.sum())}
    return result


def _assert_matches_reference(actual: dict, expected: dict, label: str) -> None:
    for metric in METRICS:
        row = actual[metric]
        ref_values = np.asarray(expected["per_image"][metric], dtype=np.float64)
        values = np.asarray(row["per_image"], dtype=np.float64)
        if (values.shape != (320,) or not np.isfinite(values).all()
                or not np.array_equal(values, ref_values)):
            delta = float(np.max(np.abs(values - ref_values))) if values.shape == (320,) else float("inf")
            if not math.isfinite(delta) or delta > 1e-10:
                raise ValueError(f"{label} does not reproduce registered {metric} per-image values")
        expected_mean = float(expected["metrics"][metric])
        if row["valid_count"] != 320 or row["mean"] is None or abs(row["mean"] - expected_mean) > 1e-10:
            raise ValueError(f"{label} does not reproduce registered {metric} mean/count")


def _assert_matches_slot_means(actual: dict, expected: dict, label: str) -> None:
    for metric in METRICS:
        row = actual[metric]
        expected_mean = float(expected["mean"][metric])
        if (row["valid_count"] != 320 or row["mean"] is None
                or not math.isfinite(expected_mean) or abs(row["mean"] - expected_mean) > 1e-10
                or int(expected["valid_count"][metric]) != 320):
            raise ValueError(f"{label} does not reproduce registered {metric} mean/count")


def run(output: Path, dataset: Path = DATASET) -> dict:
    validate_registered_hashes()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite SW0118 output: {output}")
    if dataset.resolve() != DATASET.resolve():
        raise ValueError(f"Only the registered canonical CLEVR HDF5 is permitted: {DATASET}")

    # Load and validate all six prediction blobs and all immutable references
    # before opening the HDF5 mask dataset.
    frozen = {}
    source_refs = {}
    slot_scores = {}
    for seed in range(3):
        source_ref_path = (ROOT / "collaborative_test" / "SW_0097_graph_adaptation" / "results"
                           / f"seed{seed}_positive_frozen" / "evaluation.json")
        source_refs[seed] = _reference_per_image(source_ref_path, seed)
        our_labels, our_meta = _load_ours(seed)
        slot_labels, slot_meta, slot_score = _load_slot(seed)
        frozen[seed] = {"ours": (our_labels, our_meta), "slot": (slot_labels, slot_meta)}
        slot_scores[seed] = slot_score
    released_slot = _load_released_slot()

    # Ground truth is first opened here, only after every prediction has passed
    # its shape, identity, checkpoint, report and hash checks.
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    import h5py
    with h5py.File(dataset, "r") as h5:
        masks = np.asarray(h5["mask"][1320:1640])
    if masks.shape not in ((320, 128, 128), (320, 128, 128, 1)):
        raise ValueError(f"Validation mask slice has unexpected shape {masks.shape}")
    masks_tensor = torch.as_tensor(masks, dtype=torch.int64, device="cpu")

    records = {}
    per_model_means = {name: {metric: [] for metric in METRICS} for name in ("ours", "slot")}
    oracle_seed_means = {name: [] for name in ("ours", "slot")}
    for seed in range(3):
        predictions = {}
        for model_name in ("ours", "slot"):
            labels, metadata = frozen[seed][model_name]
            predictions[model_name] = labels
        # Reproduce source results exactly before using GT for the oracle.
        target = clevr_mask_patch(masks_tensor, 8)["patch_labels"]
        source_metric = _score_standard(predictions["ours"], target)
        expected_source = source_refs[seed][0]
        _assert_matches_reference(source_metric, expected_source, f"SW0097 source seed{seed}")
        for model_name in ("ours", "slot"):
            labels, metadata = frozen[seed][model_name]
            original = _score_standard(labels, target)
            if model_name == "ours":
                _assert_matches_reference(original, expected_source, f"SW0097 source seed{seed}")
            else:
                _assert_matches_slot_means(original, slot_scores[seed], f"SW0092 Slot seed{seed}")
            oracle = [score_allowed_patch_labels(labels[i], masks_tensor[i]) for i in range(320)]
            for metric in METRICS:
                values = original[metric]
                per_model_means[model_name][metric].append(values["mean"])
            records[f"seed{seed}_{model_name}"] = {
                "prediction_provenance": metadata,
                "original_metrics": original,
                "allowed_label_oracle": {
                    "metric_name": oracle[0]["metric_name"], "is_standard_metric": False,
                    "per_image": [{k: v for k, v in row.items() if k != "resolved_target_labels"}
                                  for row in oracle],
                    "mean": (float(np.mean([x["allowed_label_oracle_fg_ari"] for x in oracle
                                           if x["score_valid"]]))
                             if any(x["score_valid"] for x in oracle) else None),
                    "valid_count": sum(int(x["score_valid"]) for x in oracle),
                },
            }
            oracle_seed_means[model_name].append(
                records[f"seed{seed}_{model_name}"]["allowed_label_oracle"]["mean"])
        if seed == 0:
            released_labels, released_meta = released_slot
            released_original = _score_standard(released_labels, target)
            _assert_matches_slot_means(released_original,
                                       {"mean": RELEASED_SLOT_MEAN,
                                        "valid_count": {name: 320 for name in METRICS}},
                                       "released Slot checkpoint")
            released_oracle = [score_allowed_patch_labels(released_labels[i], masks_tensor[i])
                               for i in range(320)]
            records["released_slot_seed0"] = {
                "prediction_provenance": released_meta,
                "original_metrics": released_original,
                "allowed_label_oracle": {
                    "metric_name": released_oracle[0]["metric_name"], "is_standard_metric": False,
                    "per_image": [{k: v for k, v in row.items() if k != "resolved_target_labels"}
                                  for row in released_oracle],
                    "mean": (float(np.mean([x["allowed_label_oracle_fg_ari"] for x in released_oracle
                                           if x["score_valid"]]))
                             if any(x["score_valid"] for x in released_oracle) else None),
                    "valid_count": sum(int(x["score_valid"]) for x in released_oracle),
                },
            }
    summary = {
        "experiment": "SW0118",
        "status": "complete_diagnostic_only",
        "dataset": str(dataset), "image_ids": [1320, 1639], "count": 320,
        "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_allowed_label_oracle": True,
        "selection_or_promotion_use": False,
        "reference_provenance": {f"source_seed{seed}": source_refs[seed][1] for seed in range(3)},
        "results": records,
        "released_slot_comparison": {
            "description": "One released checkpoint trained on different data; shown separately and not treated as a matched three-seed 70k baseline.",
            "original_metrics": RELEASED_SLOT_MEAN,
            "allowed_label_oracle_fg_ari": records["released_slot_seed0"]["allowed_label_oracle"]["mean"],
        },
        "modelwise_mean_of_seed_means": {
            model: {metric: float(np.mean(values)) for metric, values in metrics.items() if values}
            for model, metrics in per_model_means.items()
        },
        "oracle_fg_ari_mean_of_seed_means": {
            model: (float(np.mean([value for value in values if value is not None]))
                    if any(value is not None for value in values) else None)
            for model, values in oracle_seed_means.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    temp = output / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    summary = run(args.output, args.dataset)
    compact = {
        "experiment": summary["experiment"],
        "status": summary["status"],
        "output": str((args.output.resolve() / "summary.json")),
        "original_modelwise_mean_of_seed_means": summary["modelwise_mean_of_seed_means"],
        "oracle_fg_ari_mean_of_seed_means": summary["oracle_fg_ari_mean_of_seed_means"],
        "released_slot_comparison": summary["released_slot_comparison"],
    }
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
