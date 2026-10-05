"""Strict artifact validators used before SW0056 phase recovery/completion."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TRAIN_SEGMENTS = [[0, 999], [1640, 3139]]
VAL_IDS = [1320, 1639]


def validate_loss_rows(rows, updates, smoke=False):
    if len(rows) != updates:
        raise ValueError(f"expected {updates} loss rows, found {len(rows)}")
    for step, row in enumerate(rows, 1):
        expected_batch = (16 if step == 1 else 8) if smoke else (8 if step == updates else 16)
        if int(row["step"]) != step or int(row["batch_size"]) != expected_batch:
            raise ValueError(f"loss row {step} has wrong step/batch size")
        for key in ("learning_rate", "reconstruction_mse", "elapsed_seconds"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key} in loss row {step}")
        if float(row["reconstruction_mse"]) < 0:
            raise ValueError(f"negative reconstruction loss in row {step}")


def validate_prediction_arrays(image_ids, labels, background_slots, reconstruction_mse):
    if not np.array_equal(image_ids, np.arange(1320, 1640, dtype=np.int64)):
        raise ValueError("prediction IDs do not exactly match validation IDs")
    if labels.shape != (320, 128, 128) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"invalid prediction label shape/dtype: {labels.shape}, {labels.dtype}")
    if np.any(labels < 0) or np.any(labels > 11):
        raise ValueError("prediction labels outside 0..11")
    if background_slots.shape != (320,):
        raise ValueError("background slot provenance is missing or malformed")
    if reconstruction_mse.shape != (320,) or not np.isfinite(reconstruction_mse).all():
        raise ValueError("reconstruction diagnostics are missing/non-finite")


def validate_training(out, seed, smoke=False):
    out = Path(out)
    marker = out / "TRAINING_COMPLETED"
    protocol_path = out / "training_protocol.json"
    loss_path = out / "training_loss.csv"
    checkpoint_dir = out / "checkpoint"
    updates = 2 if smoke else 1563
    checkpoint_prefix = f"ckpt-{updates}"
    if not marker.is_file() or not protocol_path.is_file() or not loss_path.is_file():
        raise ValueError("training completion marker, protocol, or loss CSV missing")
    marker_text = marker.read_text(encoding="utf-8")
    if f"updates={updates}" not in marker_text or f"{checkpoint_prefix}" not in marker_text:
        raise ValueError("training completion marker has wrong update/checkpoint")
    if not (checkpoint_dir / f"{checkpoint_prefix}.index").is_file():
        raise ValueError(f"TensorFlow checkpoint index missing: {checkpoint_prefix}")
    if not list(checkpoint_dir.glob(f"{checkpoint_prefix}.data-*")):
        raise ValueError(f"TensorFlow checkpoint data shard missing: {checkpoint_prefix}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("seed") != seed or protocol.get("validation_ids_inclusive") != VAL_IDS:
        raise ValueError("training protocol seed/validation range mismatch")
    if smoke:
        expected = {"protocol_type": "smoke_test_only", "unique_training_images": 24,
                    "effective_image_exposures": 24, "optimizer_updates": 2}
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise ValueError(f"smoke protocol {key} mismatch")
    else:
        expected = {"training_id_segments_inclusive": TRAIN_SEGMENTS,
                    "unique_training_images": 2500, "exposures_per_image": 10,
                    "effective_image_exposures": 25000, "optimizer_updates": 1563,
                    "full_batch_updates": 1562, "final_batch_size": 8,
                    "resolution": [128, 128], "num_slots": 11, "iterations": 3}
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise ValueError(f"training protocol {key} mismatch")
    with loss_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate_loss_rows(rows, updates, smoke)
    if smoke:
        check = protocol.get("smoke_verification", {})
        if (check.get("batch_sizes") != [16, 8] or check.get("finite_losses") is not True or
                check.get("auxiliary_final_batch_gradient_applied_to_primary") is not True or
                check.get("final_partial_update_changed_primary_weights") is not True or
                check.get("primary_weights_changed") is not True):
            raise ValueError("smoke verification did not certify the partial-batch path")
    return protocol


def validate_inference(val_out, seed, training_protocol_path):
    val_out = Path(val_out)
    marker = val_out / "INFERENCE_COMPLETED"
    pred_path = val_out / "predictions.npz"
    protocol_path = val_out / "protocol.json"
    if not marker.is_file() or not pred_path.is_file() or not protocol_path.is_file():
        raise ValueError("inference marker, predictions, or protocol missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("seed") != seed or protocol.get("image_ids") != VAL_IDS or protocol.get("count") != 320:
        raise ValueError("inference seed/split/count mismatch")
    if protocol.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("inference protocol does not certify GT-free prediction")
    if protocol.get("training_protocol_path") != str(Path(training_protocol_path).resolve()):
        raise ValueError("inference does not reference the expected training protocol")
    if protocol.get("training_protocol", {}).get("seed") != seed:
        raise ValueError("embedded training protocol seed mismatch")
    checkpoint_sum = protocol.get("checkpoint_sha256", "")
    if len(checkpoint_sum) != 64:
        raise ValueError("inference protocol lacks checkpoint SHA256")
    expected_checkpoint = (Path(training_protocol_path).resolve().parent / "checkpoint" / "ckpt-1563").resolve()
    if Path(protocol.get("checkpoint_prefix", "")).resolve() != expected_checkpoint:
        raise ValueError("inference checkpoint prefix is not the validated SW0056 checkpoint")
    if not Path(str(expected_checkpoint) + ".index").is_file():
        raise ValueError("validated training checkpoint disappeared before inference")
    with np.load(pred_path) as saved:
        image_ids = saved["image_ids"]
        labels = saved["labels"]
        background_slots = saved["background_slots"] if "background_slots" in saved else np.empty((0,))
        reconstruction_mse = saved["reconstruction_mse"] if "reconstruction_mse" in saved else np.empty((0,))
        validate_prediction_arrays(image_ids, labels, background_slots, reconstruction_mse)
    return protocol


def validate_scoring(val_out, seed):
    val_out = Path(val_out)
    for name in ("SCORING_COMPLETED", "evaluation_summary.json", "per_image.csv", "patch_masks.pt"):
        if not (val_out / name).is_file():
            raise ValueError(f"scoring artifact missing: {name}")
    summary = json.loads((val_out / "evaluation_summary.json").read_text(encoding="utf-8"))
    if summary.get("ground_truth_used_for_prediction") is not False or summary.get("protocol", {}).get("seed") != seed:
        raise ValueError("scoring summary seed/GT provenance mismatch")
    means = summary.get("scores", {}).get("mean", {})
    for name in ("fg_ari", "foreground_iou", "matched_object_iou"):
        if name not in means or means[name] is None or not math.isfinite(float(means[name])):
            raise ValueError(f"scoring metric missing/non-finite: {name}")
    with (val_out / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320 or [int(row["image_id"]) for row in rows] != list(range(1320, 1640)):
        raise ValueError("per-image scoring CSV does not cover IDs1320-1639 exactly")
    import torch
    masks = torch.load(val_out / "patch_masks.pt", map_location="cpu", weights_only=False)
    if masks.get("image_ids") != list(range(1320, 1640)) or masks.get("patch_size") != 8:
        raise ValueError("patch-mask scoring provenance mismatch")
    for key in ("prediction", "target"):
        if tuple(masks[key].shape) != (320, 16, 16):
            raise ValueError(f"bad {key} patch-mask shape")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("training", "inference", "scoring", "complete", "smoke"))
    parser.add_argument("path", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--training-protocol", type=Path)
    args = parser.parse_args()
    if args.phase in ("training", "smoke"):
        validate_training(args.path, args.seed, smoke=args.phase == "smoke")
    elif args.phase == "inference":
        if args.training_protocol is None:
            parser.error("inference validation requires --training-protocol")
        validate_inference(args.path, args.seed, args.training_protocol)
    else:
        if args.training_protocol is None:
            parser.error("full completion validation requires --training-protocol")
        validate_inference(args.path, args.seed, args.training_protocol)
        validate_scoring(args.path, args.seed)
        if not (args.path.parent / "TRAINING_COMPLETED").is_file():
            raise ValueError("training completion marker missing")
    print(f"SW0056 {args.phase} artifacts valid for seed{args.seed}")


if __name__ == "__main__":
    main()
