"""Frozen SW0139 pilot predictions and GT-last native32 QCC scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0139_native32_representation import run
from collaborative_test.SW_0139_native32_representation.model import ContextResidualEncoder

ARMS = run.ARMS
SEED = 1
EVALUATION_ROOT = HERE / "results_archive" / "evaluation_seed1"
DATASET = Path(sw130.source97.DATASET)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _write_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def implementation_fingerprint():
    paths = (HERE / "protocol.json", HERE / "model.py", HERE / "run.py", HERE / "evaluate.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
             ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
             ROOT / "collaborative_test/SW_0137_native32_source_qcc/run.py",
             ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0139 evaluation dependency missing: {missing[0]}")
    return {p.relative_to(ROOT).as_posix(): run.sha256_file(p) for p in paths}


def validate_training(seed, arm, folder, *, lambda_path=run.LAMBDA_PATH):
    if seed != SEED or arm not in ARMS:
        raise ValueError("SW0139 endpoint currently registers seed1 and the three declared arms")
    folder = Path(folder)
    manifest_path, marker_path = folder / "manifest.json", folder / "TRAINING_COMPLETED.json"
    checkpoint_path, optimizer_path, history_path = (
        folder / "checkpoint.pt", folder / "optimizer.pt", folder / "history.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    foundation = sw135.load_native32_foundation(seed, "cpu", verify_rgb_assets=True)
    ids = [int(x) for x in foundation.image_ids]
    expected_id_sha = run._logical_id_sha(ids)
    expected_lambda_sha = run.sha256_file(lambda_path) if arm == "cross_view" else None
    if (manifest.get("experiment") != "SW0139_native32_representation"
            or manifest.get("status") != "training_complete"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("updates") != run.UPDATES or manifest.get("passes") != 1
            or manifest.get("batch_size") != run.LOGICAL_BATCH
            or manifest.get("microbatch_size") != run.MICROBATCH
            or manifest.get("unique_training_images") != run.TRAIN_IMAGES
            or manifest.get("total_image_exposures") != run.TRAIN_IMAGES * (2 if arm == "cross_view" else 1)
            or manifest.get("views_per_image") != (2 if arm == "cross_view" else 1)
            or manifest.get("training_ids") != ids or manifest.get("training_ids_sha256") != expected_id_sha
            or manifest.get("source_core_sha256") != foundation.provenance["source_core_sha256"]
            or manifest.get("source_manifest_sha256") != foundation.provenance["source_manifest_sha256"]
            or manifest.get("source_training_ids_sha256") != foundation.provenance["source_training_ids_sha256"]
            or manifest.get("pool_indices_sha256") != run._logical_id_sha(foundation.pool_indices)
            or manifest.get("implementation_fingerprint") != run.implementation_fingerprint()
            or manifest.get("asset_hashes") != foundation.provenance["rgb_asset_validation"]
            or manifest.get("lambda_sha256") != expected_lambda_sha
            or run.sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256")
            or run.sha256_file(optimizer_path) != manifest.get("optimizer_sha256")
            or run.sha256_file(history_path) != manifest.get("history_sha256")
            or marker.get("status") != "training_complete"
            or marker.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
            or marker.get("manifest_sha256") != run.sha256_file(manifest_path)
            or marker.get("updates") != run.UPDATES
            or marker.get("ground_truth_used_for_training") is not False):
        raise ValueError(f"SW0139 {arm} training manifest/checkpoint is not the registered seed1 pilot")
    preflight_path = run.ARCHIVE / f"preflight_{arm}_seed{seed}.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (run.sha256_file(preflight_path) != manifest.get("preflight_sha256")
            or preflight.get("status") != "disposable_update_complete"
            or preflight.get("implementation_fingerprint") != run.implementation_fingerprint()
            or preflight.get("seed") != seed or preflight.get("arm") != arm):
        raise ValueError("SW0139 completion artifact no longer matches its preflight")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = history.get("records")
    if history.get("updates") != run.UPDATES or not isinstance(rows, list) or len(rows) != run.UPDATES:
        raise ValueError("SW0139 history must contain all256 updates")
    for update, row in enumerate(rows, 1):
        expected = ids[(update - 1) * run.LOGICAL_BATCH:update * run.LOGICAL_BATCH]
        if (row.get("update") != update or row.get("batch_start") != (update - 1) * run.LOGICAL_BATCH
                or row.get("training_ids") != expected
                or row.get("training_ids_sha256") != run._logical_id_sha(expected)
                or not np.isfinite(np.asarray(row.get("old_loss_micro_mean", []), dtype=np.float64)).all()
                or len(row.get("old_loss_micro_mean", [])) != run.MICROS
                or (arm == "cross_view" and (len(row.get("contrastive_loss_micro_mean", [])) != run.MICROS
                    or not np.isfinite(np.asarray(row["contrastive_loss_micro_mean"], dtype=np.float64)).all()))
                or not math.isfinite(float(row.get("gradient_norm_before_clip", float("nan"))))
                or float(row["gradient_norm_before_clip"]) <= 0):
            raise ValueError(f"SW0139 history update {update} violates order/loss/gradient contract")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (state.get("seed") != seed or state.get("arm") != arm
            or state.get("source_core_sha256") != manifest.get("source_core_sha256")
            or state.get("training_ids_sha256") != expected_id_sha
            or state.get("implementation_fingerprint") != run.implementation_fingerprint()
            or (arm == "cross_view" and float(state.get("lambda", float("nan"))) != float(manifest["lambda"]))):
        raise ValueError("SW0139 checkpoint payload identity differs from its manifest")
    for key in ("wrapped_state_dict", "encoder_state_dict"):
        values = state.get(key)
        if not isinstance(values, dict) or not values or any(
                torch.is_tensor(v) and not torch.isfinite(v).all() for v in values.values()):
            raise ValueError(f"SW0139 checkpoint has missing/nonfinite {key}")
    if (arm == "context_residual") != isinstance(state.get("context_state_dict"), dict):
        raise ValueError("SW0139 context state presence does not match selected arm")
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    steps = []
    for st in optimizer.get("state", {}).values():
        value = st.get("step")
        steps.append(int(value.item() if torch.is_tensor(value) else value))
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in st.values()):
            raise ValueError("SW0139 optimizer contains nonfinite moments")
    if not steps or set(steps) != {run.UPDATES}:
        raise ValueError("SW0139 Adam optimizer state must contain exactly256 updates")
    return {"manifest": manifest, "checkpoint": checkpoint_path,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "history": history,
            "preflight": preflight, "foundation": foundation}


def _arm_prediction_dir(arm, root=EVALUATION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0139 evaluation arm")
    return Path(root) / arm


@torch.no_grad()
def predict_arm(seed, arm, *, device="cuda:0", training_root=run.OUT,
                output_root=EVALUATION_ROOT):
    verified = validate_training(seed, arm, Path(training_root) / f"{arm}_seed{seed}")
    output = _arm_prediction_dir(arm, output_root)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0139 prediction: {output}")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    foundation = sw135.load_native32_foundation(seed, device, verify_rgb_assets=True)
    payload = torch.load(verified["checkpoint"], map_location=device, weights_only=True)
    foundation.wrapped.load_state_dict(payload["wrapped_state_dict"], strict=True)
    foundation.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    context = None
    if arm == "context_residual":
        context = ContextResidualEncoder(seed=139 + seed).to(device)
        context.load_state_dict(payload["context_state_dict"], strict=True)
        context.eval()
    foundation.wrapped.eval(); foundation.encoder.eval()
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    labels = np.empty((contract.COUNT, contract.GRID, contract.GRID), dtype=np.int64)
    ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
    for i in range(contract.COUNT):
        rgb = read_eval_rgb(cache, i, device)
        gamma = run._features_to_gamma(foundation, rgb, context)
        trace = foundation.rollout(gamma, total_steps=run.STEPS, live_tail_steps=0)
        labels[i:i + 1] = sw136_eval._qcc_labels(trace)
        if i % 40 == 39:
            print(json.dumps({"stage": "sw0139_predict", "arm": arm,
                              "seed": seed, "predicted": i + 1, "count": contract.COUNT}), flush=True)
    if labels.shape != (320, 32, 32) or not np.isfinite(labels).all():
        raise ValueError("SW0139 QCC predictor emitted invalid fixed320 labels")
    npz = output / "predictions.npz"
    _write_npz_once(npz, image_ids=ids, qcc_labels=labels)
    protocol = {"experiment": "SW0139_native32_representation", "status": "predictions_complete",
                "seed": seed, "arm": arm, "image_ids": [1320, 1639], "count": 320,
                "prediction_shape": [320, 32, 32], "steps": 1024, "settle": 512,
                "readout": {"type": "original_native_actual_spike_product_QCC", "threshold": .50,
                            "min_group_size": 8, "background": "largest connected component"},
                "checkpoint_sha256": verified["checkpoint_sha256"],
                "training_manifest_sha256": run.sha256_file(Path(training_root) / f"{arm}_seed{seed}" / "manifest.json"),
                "source_core_sha256": verified["manifest"]["source_core_sha256"],
                "prediction_sha256": run.sha256_file(npz),
                "implementation_fingerprint": implementation_fingerprint(),
                "ground_truth_used_for_prediction": False, "optimizer_updates": 0}
    _write_once(output / "protocol.json", protocol)
    return protocol


def read_eval_rgb(cache, row, device):
    raw = np.asarray(cache[row:row + 1]).copy()
    if raw.dtype != np.uint8 or raw.shape != (1, 128, 128, 3):
        raise ValueError("SW0139 validation cache must be uint8 native128 RGB")
    # Match the training/foundation path: exactly one /255 conversion, then NCHW.
    return torch.from_numpy(raw).to(device=device).permute(0, 3, 1, 2).float() / 255.0


def validate_arm_prediction(arm, *, seed=SEED, root=EVALUATION_ROOT, training_root=run.OUT):
    directory = _arm_prediction_dir(arm, root)
    npz, prot = directory / "predictions.npz", directory / "protocol.json"
    pbytes = prot.read_bytes(); psha = hashlib.sha256(pbytes).hexdigest()
    protocol = json.loads(pbytes.decode("utf-8"))
    trained = validate_training(seed, arm, Path(training_root) / f"{arm}_seed{seed}")
    if (protocol.get("status") != "predictions_complete"
            or protocol.get("experiment") != "SW0139_native32_representation"
            or protocol.get("seed") != seed or protocol.get("arm") != arm
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != 320
            or protocol.get("prediction_shape") != [320, 32, 32]
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("checkpoint_sha256") != trained["checkpoint_sha256"]
            or protocol.get("source_core_sha256") != trained["manifest"]["source_core_sha256"]
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != run.sha256_file(npz)):
        raise ValueError(f"invalid SW0139 frozen predictions for {arm}")
    with np.load(npz, allow_pickle=False) as saved:
        if not {"image_ids", "qcc_labels"}.issubset(saved.files):
            raise ValueError("SW0139 predictions archive is missing required arrays")
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if labels.shape != (320, 32, 32) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("SW0139 QCC labels violate fixed320 native32 schema")
    if run.sha256_file(npz) != protocol["prediction_sha256"] or run.sha256_file(prot) != psha:
        raise ValueError("SW0139 prediction changed during validation")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False), "protocol": protocol,
            "prediction_path": npz, "protocol_path": prot,
            "prediction_sha256": protocol["prediction_sha256"], "protocol_sha256": psha}


def score_all(*, prediction_root=EVALUATION_ROOT, training_root=run.OUT,
              source_root=sw137.OUTPUT_ROOT, dataset=DATASET,
              output=EVALUATION_ROOT / "evaluation.json", bootstrap_samples=10000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0139 evaluation summary: {output}")
    arm_rows, rejected_arms = {}, {}
    for arm in ARMS:
        prediction_dir = _arm_prediction_dir(arm, prediction_root)
        if not prediction_dir.exists():
            rejected_arms[arm] = "no frozen prediction artifact"
            continue
        try:
            arm_rows[arm] = validate_arm_prediction(arm, root=prediction_root,
                                                    training_root=training_root)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            rejected_arms[arm] = f"invalid frozen prediction: {type(exc).__name__}: {exc}"
    if not arm_rows:
        raise ValueError("SW0139 cannot score: no valid arm predictions are available")
    source = sw137.validate_prediction(sw137._prediction_dir(SEED, source_root), seed=SEED)
    if not np.array_equal(source["ids"], np.asarray(contract.IMAGE_IDS, dtype=np.int64)):
        raise ValueError("SW0139 source QCC reference does not match fixed320 IDs")
    # Reverify all frozen prediction bytes immediately before opening target masks.
    for row in arm_rows.values():
        if (run.sha256_file(row["prediction_path"]) != row["prediction_sha256"]
                or run.sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("SW0139 arm prediction changed before GT scoring")
    if (run.sha256_file(source["archive_path"]) != source["prediction_sha256"]
            or run.sha256_file(source["protocol_path"]) != source["protocol_sha256"]):
        raise ValueError("SW0139 source prediction changed before GT scoring")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= 1639:
            raise ValueError("SW0139 fixed320 HDF5 targets are unavailable")
        pixel_gt = np.asarray(h5["mask"][1320:1640])
    gt = contract.modal_native32(pixel_gt)
    scores = {arm: contract.production_metrics32(row["labels"], gt)
              for arm, row in arm_rows.items()}
    scores["source97"] = contract.production_metrics32(source["labels"], gt)
    rng = np.random.RandomState(139)
    samples = rng.randint(0, contract.COUNT, size=(bootstrap_samples, contract.COUNT))
    comparisons = {}
    for candidate in ARMS:
        if candidate not in scores:
            continue
        comparisons[candidate] = {}
        references = [ref for ref in ("source97", "control") if ref in scores and ref != candidate]
        for reference in references:
            comparisons[candidate][reference] = {}
            for metric in METRICS:
                left = np.asarray(scores[candidate][metric]["per_image"], dtype=np.float64)
                right = np.asarray(scores[reference][metric]["per_image"], dtype=np.float64)
                delta = left - right
                low, high = np.quantile(delta[samples].mean(axis=1), [.025, .975])
                comparisons[candidate][reference][metric] = {
                    "mean_difference": float(delta.mean()), "ci95": [float(low), float(high)]}
    gates = {}
    for candidate in ("context_residual", "cross_view"):
        if candidate not in scores:
            gates[candidate] = {"available": False, "may_justify_seeds0_2": False}
            continue
        refs_available = all(ref in scores and ref in comparisons.get(candidate, {})
                             for ref in ("source97", "control"))
        above = refs_available and all(scores[candidate]["fg_ari"]["mean"] > scores[ref]["fg_ari"]["mean"]
                                       for ref in ("source97", "control"))
        ci_positive = refs_available and all(comparisons[candidate][ref]["fg_ari"]["ci95"][0] > 0
                                             for ref in ("source97", "control"))
        gates[candidate] = {"mean_fg_ari_above_source_and_control": above,
                            "paired_fg_ari_lower_bounds_positive": ci_positive,
                            "available": True,
                            "may_justify_seeds0_2": bool(above and ci_positive)}
    all_arms_valid = set(arm_rows) == set(ARMS)
    report = {"experiment": "SW0139_native32_representation",
              "status": "complete" if all_arms_valid else "partial_complete",
              "seed": SEED, "count": contract.COUNT, "image_ids": [1320, 1639],
              "scores": scores, "paired_comparisons": comparisons,
              "candidate_expansion_gates": gates,
              "valid_arms": list(arm_rows), "rejected_arms": rejected_arms,
              "bootstrap": {"method": "paired_image", "samples": bootstrap_samples,
                            "random_state": 139, "common_indices_across_arms": True},
              "prediction_provenance": {arm: row["prediction_sha256"] for arm, row in arm_rows.items()},
              "source_prediction_sha256": source["prediction_sha256"],
              "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
              "interpretation": "Seed1 representation pilot only; no three-seed or data-scaling claim."}
    _write_once(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--seed", type=int, choices=(1,), default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-root", type=Path, default=run.OUT)
    parser.add_argument("--prediction-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--source-root", type=Path, default=sw137.OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            raise ValueError("prediction stage requires an arm")
        result = predict_arm(args.seed, args.arm, device=args.device,
                             training_root=args.training_root, output_root=args.prediction_root)
    else:
        result = score_all(prediction_root=args.prediction_root, training_root=args.training_root,
                           source_root=args.source_root, dataset=args.dataset,
                           output=args.output or args.prediction_root / "evaluation.json")
    print(json.dumps({"status": result.get("status"), "experiment": result.get("experiment")},
                     allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
