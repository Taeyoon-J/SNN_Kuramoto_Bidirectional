"""Frozen SW0134 endpoint evaluation; commits predictions before reading masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0134_native_spike_binding import run, train
from collaborative_test.SW_0134_native_spike_binding.binder import canonical_hard_labels
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout

ROOT = run.ROOT
HERE = run.HERE
IDS = tuple(range(1320, 1640))
EVAL_BATCH = 8
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
ARMS = ("actual_joint", "gate_joint", "actual_frozen")


def _source_reference(seed):
    path = (ROOT / "collaborative_test/SW_0097_graph_adaptation/results" /
            f"seed{seed}_positive_frozen/evaluation.json")
    if not path.is_file():
        raise FileNotFoundError(f"registered SW0097 source evaluation missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ids") != [IDS[0], IDS[-1]] or payload.get("images") != len(IDS):
        raise ValueError("SW0097 source evaluation does not match fixed320 IDs")
    try:
        score = payload["sweep"][0]["scored_targets"]["our_hdf5"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("SW0097 evaluation schema changed") from exc
    values = {}
    for metric in METRICS:
        row = score.get("per_image", {}).get(metric)
        if (not isinstance(row, list) or len(row) != len(IDS)
                or score.get("valid_count", {}).get(metric) != len(IDS)
                or not all(math.isfinite(float(v)) for v in row)):
            raise ValueError(f"SW0097 source reference {metric} is malformed")
        mean = float(score["metrics"][metric])
        if not math.isfinite(mean) or abs(float(np.mean(row)) - mean) > 1e-10:
            raise ValueError(f"SW0097 source {metric} mean does not match per-image scores")
        values[metric] = [float(v) for v in row]
    if payload.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("registered source predictions must not use GT")
    return path, values


def _load_training(seed, arm, device, output_root, assets=None):
    folder = Path(output_root) / f"{arm}_seed{seed}"
    manifest_path = folder / "manifest.json"
    marker_path = folder / "TRAINING_COMPLETED.json"
    checkpoint_path = folder / "checkpoint.pt"
    optimizer_path = folder / "optimizers.pt"
    history_path = folder / "history.json"
    for path in (manifest_path, marker_path, checkpoint_path, optimizer_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed SW0134 training artifact missing: {path}")
    manifest, state, checkpoint_path = train.validate_completed_training(
        seed, arm, folder, assets=assets)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, *_ = run.load_models(seed, device)
    wrapped.load_state_dict(state["wrapped_state_dict"], strict=True)
    encoder.load_state_dict(state["encoder_state_dict"], strict=True)
    binder.load_state_dict(state["binder_state_dict"], strict=True)
    decoder.load_state_dict(state["decoder_state_dict"], strict=True)
    wrapped.eval(); encoder.eval(); binder.eval(); decoder.eval()
    wrapped.requires_grad_(False); encoder.requires_grad_(False)
    binder.requires_grad_(False); decoder.requires_grad_(False)
    return (wrapped, encoder, patcher, mean, std, clip, binder, decoder,
            manifest, state, checkpoint_path)


@torch.no_grad()
def _predict_arm(seed, arm, device, rows, validation_cache, source_gamma_cache,
                 training_root, assets=None):
    if arm == "source97_native":
        wrapped, encoder, patcher, mean, std, clip, *_ = run.load_models(seed, device)
        wrapped.eval(); encoder.eval()
        binder = decoder = None
        manifest = {"source_core_sha256": run.source97.EXPECTED_SOURCE_SHAS[seed]}
        checkpoint = run.source_contract(seed)[0]
    else:
        (wrapped, encoder, patcher, mean, std, clip, binder, decoder,
         manifest, _state, checkpoint) = _load_training(seed, arm, device, training_root,
                                                        assets=assets)
    primary_rows, qcc_rows = [], []
    for start in range(0, len(rows), EVAL_BATCH):
        selected = rows[start:start + EVAL_BATCH]
        images = run.sw130.read_batch(validation_cache, selected, device)
        gamma = (source_gamma_cache[torch.as_tensor(selected, dtype=torch.long)].to(device)
                 if arm == "source97_native" else
                 run.sw130.encode(encoder, patcher, mean, std, clip, images))
        trace = late_rollout(wrapped, gamma, total_steps=1024, live_tail_steps=0)
        components = trace["component_spikes"]
        if tuple(components.shape) != (len(selected), 4, 256, 1024):
            raise AssertionError("SW0134 evaluation emitted an unexpected spike trace shape")
        qcc_labels, _hard, _groups = run.sw130.hard_labels(
            components.mean(dim=1), components, settle=512)
        qcc_rows.append(qcc_labels.detach().to(device="cpu", dtype=torch.int64))
        if arm == "source97_native":
            primary_rows.append(qcc_labels.detach().to(device="cpu", dtype=torch.int64))
        else:
            head_trace = components[..., -512:]
            if arm == "gate_joint":
                head_trace = trace["component_gates"][..., -512:]
            _p, _slots, _features = binder(head_trace)
            primary_rows.append(canonical_hard_labels(_p).to(dtype=torch.int64, device="cpu"))
    primary = torch.cat(primary_rows)
    qcc = torch.cat(qcc_rows)
    if tuple(primary.shape) != (len(IDS), 16, 16) or tuple(qcc.shape) != (len(IDS), 16, 16):
        raise AssertionError("SW0134 endpoint did not cover exactly 320 16x16 predictions")
    return {"primary": primary, "qcc": qcc}, {
        "arm": arm, "seed": seed, "checkpoint_sha256": run.sha(checkpoint),
        "source_core_sha256": manifest["source_core_sha256"],
        "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "ground_truth_used_for_prediction": False,
    }


def _commit_predictions(output, predictions, provenance, seed):
    prediction_path = output / "frozen_predictions.pt"
    manifest_path = output / "prediction_manifest.json"
    with prediction_path.open("xb") as stream:
        torch.save({"image_ids": list(IDS), "predictions": predictions,
                    "provenance": provenance}, stream)
        stream.flush()
    prediction_sha = run.sha(prediction_path)
    run.write_once(manifest_path, {
        "status": "predictions_complete", "experiment": "SW0134_native_spike_binding",
        "seed": seed, "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "arms": ["source97_native", *ARMS], "prediction_sha256": prediction_sha,
        "ground_truth_used_for_prediction": False,
        "contract": {"steps": 1024, "settle": 512, "batch_size": EVAL_BATCH,
                     "primary": "SW0134 slot argmax; QCC secondary"},
        "provenance": provenance})
    if run.sha(prediction_path) != prediction_sha:
        raise AssertionError("SW0134 prediction bundle changed after commit")
    stored = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (stored.get("image_ids") != list(IDS)
            or set(stored.get("predictions", {})) != {"source97_native", *ARMS}
            or any(tuple(value.shape) != (len(IDS), 16, 16)
                   for arm in stored["predictions"].values()
                   for value in arm.values())):
        raise ValueError("SW0134 frozen predictions failed complete-shape verification")
    return prediction_path, prediction_sha, run.sha(manifest_path)


def _load_targets(dataset):
    if Path(dataset).resolve() != DATASET.resolve():
        raise ValueError("SW0134 only scores against the registered canonical CLEVR dataset")
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    with h5py.File(dataset, "r") as handle:
        masks = np.asarray(handle["mask"][IDS[0]:IDS[-1] + 1])
    if masks.shape not in ((len(IDS), 128, 128), (len(IDS), 128, 128, 1)):
        raise ValueError(f"registered validation mask slice has unexpected shape {masks.shape}")
    return clevr_mask_patch(torch.as_tensor(masks, dtype=torch.int64), 8)["patch_labels"]


def _score(pred, target):
    from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
    score = evaluate_patch_masks(pred, target)
    output = {}
    for key, metric in (("patch_fg_ari", "fg_ari"),
                        ("patch_foreground_iou", "foreground_iou"),
                        ("patch_matched_object_iou", "matched_object_iou")):
        values = score["per_image"][metric].detach().cpu().numpy().astype(np.float64)
        finite = np.isfinite(values)
        output[key] = {"per_image": [float(x) if math.isfinite(float(x)) else None for x in values],
                       "mean": float(values[finite].mean()) if finite.any() else None,
                       "valid_count": int(finite.sum())}
    return output


def _source_reproduction(actual, expected):
    rows = {}
    passed = True
    names = {"patch_fg_ari": "fg_ari", "patch_foreground_iou": "foreground_iou",
             "patch_matched_object_iou": "matched_object_iou"}
    for metric, source_key in names.items():
        got = np.asarray(actual[metric]["per_image"], dtype=np.float64)
        want = np.asarray(expected[source_key], dtype=np.float64)
        diff = float(np.max(np.abs(got - want))) if got.shape == want.shape == (320,) else float("inf")
        ok = bool(np.isfinite(got).all() and np.isfinite(want).all() and diff <= 1e-10)
        rows[metric] = {"max_abs_diff": diff, "passed": ok}
        passed = passed and ok
    return {"passed": passed, "metrics": rows}


def evaluate(seed, device, output, *, dataset=DATASET,
             training_root=None, validation_cache_path=None):
    if seed not in run.SEEDS:
        raise ValueError("SW0134 evaluation seed must be registered")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0134 evaluation attempt: {output}")
    if Path(dataset).resolve() != DATASET.resolve():
        raise ValueError("SW0134 only permits the registered canonical validation dataset")
    output.mkdir(parents=True, exist_ok=False)
    source_path, source_values = _source_reference(seed)
    assets = run.sw130.validate_rgb_assets()
    source_gamma_cache, _gamma_manifest = run.source97.validate_gamma_cache(
        run.source97.GAMMA_VAL, run.source97.GAMMA_VAL_MANIFEST, validation=True)
    validation_path = Path(validation_cache_path or run.sw130.VAL_RGB)
    if run.sha(validation_path) != assets["validation_cache_sha256"]:
        raise ValueError("validation RGB cache differs from preflight-bound bytes")
    validation_cache = np.load(validation_path, mmap_mode="r")
    rows = np.arange(len(IDS), dtype=np.int64)
    predictions, provenance = {}, {}
    arms = ("source97_native", *ARMS)
    for arm in arms:
        predictions[arm], provenance[arm] = _predict_arm(
            seed, arm, torch.device(device), rows, validation_cache, source_gamma_cache,
            training_root or (ROOT / "trained_models/SW0134_native_spike_binding"),
            assets=assets)
    prediction_path, prediction_sha, prediction_manifest_sha = _commit_predictions(
        output, predictions, provenance, seed)

    # Targets are opened only after every arm's complete prediction bundle is
    # saved, hashed, and re-read against the fixed-ID/shape contract.
    target = _load_targets(dataset)
    result = {arm: {kind: _score(bundle[kind], target)
                    for kind in ("primary", "qcc")}
              for arm, bundle in predictions.items()}
    source_repro = _source_reproduction(result["source97_native"]["qcc"], source_values)
    report = {
        "status": "complete" if source_repro["passed"] else "failed_source97_reproduction",
        "experiment": "SW0134_native_spike_binding", "seed": seed,
        "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_training": False,
        "ground_truth_used_for_scoring": True,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": prediction_sha,
        "prediction_manifest_sha256": prediction_manifest_sha,
        "source97_evaluation_sha256": run.sha(source_path),
        "arm_provenance": provenance,
        "scores": result,
        "source97_qcc_per_image_reproduction": source_repro,
        "evaluation_contract": {"batch_size": EVAL_BATCH, "time_steps": 1024,
                                "settle_steps": 512, "native_qcc_secondary": True},
        "implementation_fingerprint": {
            **run.implementation_fingerprint(),
            (HERE / "train.py").relative_to(ROOT).as_posix(): run.sha(HERE / "train.py"),
            (HERE / "evaluate.py").relative_to(ROOT).as_posix(): run.sha(Path(__file__)),
        },
        "created_unix": time.time(),
        "interpretation": "Fixed-budget pilot endpoint; no promotion implied by this report alone.",
    }
    run.write_once(output / "evaluation.json", report)
    run.write_once(output / "evaluation_manifest.json", {
        "experiment": "SW0134_native_spike_binding", "seed": seed,
        "evaluation_sha256": run.sha(output / "evaluation.json"),
        "prediction_sha256": prediction_sha,
        "source97_evaluation_sha256": run.sha(source_path),
        "implementation_fingerprint": report["implementation_fingerprint"],
        "ground_truth_used_for_prediction": False})
    if not source_repro["passed"]:
        raise AssertionError("SW0134 source QCC does not reproduce; saved failure artifacts preserved")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=run.SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path,
                        default=ROOT / "trained_models/SW0134_native_spike_binding")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)
    report = evaluate(args.seed, args.device, args.output, dataset=args.dataset,
                      training_root=args.training_root)
    print(json.dumps({"status": report["status"], "experiment": "SW0134_native_spike_binding",
                      "seed": args.seed, "output": str(args.output),
                      "prediction_sha256": report["prediction_sha256"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
