"""GT-free first-64 TRAIN permutation diagnostic for SW0150 decoder alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0150_spike_spatial_mixture import evaluate, integration, train
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import gaussian_mixture_nll

NAME = train.NAME
DIAGNOSTIC = "first64_train_color_bin_assignment_permutation"
IMAGE_COUNT = 64
PERMUTATION_SEED = 150002
PROTOCOL_PATH = HERE / "DIAGNOSIS_PROTOCOL.json"
OUTPUT_DEFAULT = HERE / "results_archive" / "color_bin_assignment_diagnostic_seed1.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    # Inherit the complete registered training dependency graph (including the
    # native rollout implementation), then bind this diagnostic's own inputs.
    fingerprint = dict(train.implementation_fingerprint())
    paths = (HERE / "diagnosis.py", PROTOCOL_PATH, HERE / "evaluate.py")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0150 diagnostic dependency missing: {missing[0]}")
    fingerprint.update({path.relative_to(ROOT).as_posix(): sha256_file(path)
                        for path in paths})
    return fingerprint


def _patch_color_bins(rgb_uint8):
    """Return joint 8x8x8 RGB-bin IDs for 4x4 patch means."""
    if not isinstance(rgb_uint8, np.ndarray):
        rgb_uint8 = np.asarray(rgb_uint8)
    if (rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 4
            or tuple(rgb_uint8.shape[1:]) != (128, 128, 3)
            or rgb_uint8.shape[0] < 1):
        raise ValueError("diagnostic RGB must be NHWC uint8 [B,128,128,3]")
    patch_means = rgb_uint8.reshape(
        rgb_uint8.shape[0], 32, 4, 32, 4, 3).mean(axis=(2, 4), dtype=np.float64) / 255.0
    channel_bins = np.floor(patch_means * 8.0).astype(np.int64).clip(0, 7)
    return channel_bins[..., 0] * 64 + channel_bins[..., 1] * 8 + channel_bins[..., 2]


def permute_within_color_bins(assignment, rgb_uint8, image_ids, *, seed=PERMUTATION_SEED):
    """Shuffle P node rows within each image's joint RGB patch-mean bins."""
    if (not torch.is_tensor(assignment) or assignment.ndim != 3
            or assignment.shape[1] != 1024 or assignment.shape[0] < 1
            or not 1 <= assignment.shape[2] <= 11 or not assignment.is_floating_point()):
        raise ValueError("P must be floating point [B,1024,K], 1<=K<=11")
    if not torch.isfinite(assignment).all() or torch.any(assignment < 0):
        raise ValueError("P must be finite and nonnegative")
    if not torch.allclose(assignment.sum(dim=-1), torch.ones_like(assignment[..., 0]),
                          rtol=0, atol=1e-6):
        raise ValueError("P must be row-stochastic")
    rgb = np.asarray(rgb_uint8)
    bins = _patch_color_bins(rgb)
    ids = [int(value) for value in image_ids]
    if (len(ids) != assignment.shape[0] or len(set(ids)) != len(ids)
            or bins.shape[0] != assignment.shape[0]):
        raise ValueError("image IDs, P, and RGB batch must align with unique IDs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("permutation seed must be a nonnegative integer")

    rng = np.random.RandomState(seed)
    result = assignment.clone()
    rows = []
    for batch_index, image_id in enumerate(ids):
        flat_bins = bins[batch_index].reshape(-1)
        eligible_rows = 0
        occupied_multirow_bins = 0
        shuffled_positions = 0
        for color_bin in np.unique(flat_bins):
            indices = np.flatnonzero(flat_bins == color_bin)
            if indices.size < 2:
                continue
            occupied_multirow_bins += 1
            eligible_rows += int(indices.size)
            permutation = rng.permutation(indices.size)
            shuffled_positions += int(np.count_nonzero(permutation != np.arange(indices.size)))
            dst = torch.as_tensor(indices, dtype=torch.long, device=assignment.device)
            src = torch.as_tensor(indices[permutation], dtype=torch.long,
                                   device=assignment.device)
            result[batch_index, dst] = assignment[batch_index, src]
        changed = int(torch.any(result[batch_index] != assignment[batch_index], dim=-1)
                      .sum().detach().cpu().item())
        rows.append({
            "image_id": image_id,
            "occupied_multrow_bin_count": occupied_multirow_bins,
            "eligible_rows": eligible_rows,
            "permuted_row_positions": shuffled_positions,
            "changed_assignment_rows": changed,
            "changed_row_fraction": changed / 1024.0,
        })
    return result, rows


def _validate_pair_lineage(admission, training):
    if set(training) != set(train.ARMS):
        raise ValueError("both completed SW0150 training arms are required")
    shared = None
    for arm in train.ARMS:
        row = training[arm]
        manifest = row["manifest"]
        binding = manifest.get("bindings", {})
        current = {
            "preflight_sha256": manifest.get("preflight_sha256"),
            "warm_sha256": manifest.get("preflight_warm_sha256"),
            "source_core_sha256": manifest.get("source_core_sha256"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "source_training_ids_sha256": manifest.get("source_training_ids_sha256"),
            "rgb_asset_validation": manifest.get("rgb_asset_validation"),
            "initial_parameter_sha256": binding.get("initial_parameter_sha256"),
            "implementation_sha256": manifest.get("implementation_sha256"),
        }
        expected = {
            "preflight_sha256": admission["report_sha256"],
            "warm_sha256": admission["warm_sha256"],
            "source_core_sha256": admission["source"]["core_sha256"],
            "source_manifest_sha256": train.sha(admission["source"]["manifest_path"]),
            "source_training_ids_sha256": train.ids_sha(admission["source"]["ids"]),
            "rgb_asset_validation": admission["assets"],
            "initial_parameter_sha256": admission["report"]["arms"][arm][
                "initial_parameter_group_sha256"],
            "implementation_sha256": train.implementation_sha(),
        }
        if current != expected or binding.get("preflight_sha256") != admission["report_sha256"]:
            raise ValueError(f"{arm} completed training differs from validated source/warm lineage")
        if shared is None:
            shared = current
        elif any(shared[key] != current[key] for key in (
                "preflight_sha256", "warm_sha256", "source_core_sha256",
                "source_manifest_sha256", "source_training_ids_sha256",
                "rgb_asset_validation")):
            raise ValueError("the two candidate arms do not share exact source/preflight/warm lineage")
    return True


def _score_image(gamma, assignment, rgb, decoder, sigma2):
    return float(gaussian_mixture_nll(gamma, assignment, rgb, decoder,
                                      sigma2=sigma2).detach().cpu().item())


@torch.no_grad()
def _diagnose_arm(arm, *, foundation, source, admission, rgb_cache):
    decoder = foundation.wrapped.spatial_decoder
    before_sha = evaluate.state_dict_sha256((("wrapped", foundation.wrapped),
                                             ("encoder", foundation.encoder)))
    assignments, gamma_rows, rgb_rows, native_slots = [], [], [], []
    rollout_count = 0
    for start in range(0, IMAGE_COUNT, train.MICRO):
        stop = start + train.MICRO
        rgb_np = np.asarray(rgb_cache[source["rows"][start:stop]]).copy()
        if rgb_np.shape != (train.MICRO, 128, 128, 3) or rgb_np.dtype != np.uint8:
            raise ValueError("registered source TRAIN RGB cache returned an invalid diagnostic batch")
        rgb = torch.from_numpy(rgb_np).to(next(foundation.encoder.parameters()).device)
        ids = source["ids"][start:stop]
        result = integration.forward_assignment(foundation, rgb,
                                                live_tail_steps=train.TAIL)
        rollout_count += len(ids)
        gamma = result["gamma_content"]
        probability = result["p"]
        if (result.get("live_tail_steps") != train.TAIL
                or tuple(gamma.shape) != (train.MICRO, 1024, 8)
                or probability.ndim != 3
                or tuple(probability.shape[:2]) != (train.MICRO, 1024)
                or not 1 <= probability.shape[2] <= 11
                or not torch.isfinite(gamma).all()
                or not torch.isfinite(probability).all()
                or torch.any(probability < 0)
                or not torch.allclose(probability.sum(dim=-1),
                                      torch.ones_like(probability[..., 0]),
                                      rtol=0, atol=1e-6)):
            raise ValueError("SW0150 diagnostic rollout/assignment shape differs from training")
        assignments.append(probability.detach())
        native_slots.append(int(probability.shape[2]))
        gamma_rows.append(gamma.detach())
        rgb_rows.append(rgb_np)
        del result, rgb
    # Each independently assigned microbatch can have a different K. Pad only
    # the absent trailing slots with exact zero mass; slot zero remains BG.
    max_slots = max(native_slots)
    assignment = torch.cat([
        torch.nn.functional.pad(piece, (0, max_slots - piece.shape[2]), value=0.0)
        for piece in assignments
    ], dim=0)
    gamma_content = torch.cat(gamma_rows, dim=0)
    all_rgb = np.concatenate(rgb_rows, axis=0)
    shuffled, intervention_rows = permute_within_color_bins(
        assignment, all_rgb, source["ids"][:IMAGE_COUNT], seed=PERMUTATION_SEED)
    per_image = []
    for index, image_id in enumerate(source["ids"][:IMAGE_COUNT]):
        gamma = gamma_content[index:index + 1]
        original = assignment[index:index + 1]
        permuted = shuffled[index:index + 1]
        image = torch.from_numpy(all_rgb[index:index + 1].copy()).to(assignment.device)
        original_nll = _score_image(gamma, original, image, decoder, admission["sigma2"])
        repeat_delta = None
        if index == 0:
            repeat_nll = _score_image(gamma, original.clone(), image, decoder,
                                      admission["sigma2"])
            repeat_delta = repeat_nll - original_nll
            if repeat_delta != 0.0:
                raise AssertionError("original/original NLL sanity check was not exact")
        permuted_nll = _score_image(gamma, permuted, image, decoder, admission["sigma2"])
        per_image.append({
            **intervention_rows[index],
            "native_microbatch_slots": native_slots[index // train.MICRO],
            "padded_slots": max_slots,
            "nll_original": original_nll,
            "nll_original_repeat_delta": repeat_delta,
            "nll_color_bin_permuted": permuted_nll,
            "paired_delta_permuted_minus_original": permuted_nll - original_nll,
        })
    after_sha = evaluate.state_dict_sha256((("wrapped", foundation.wrapped),
                                            ("encoder", foundation.encoder)))
    if before_sha != after_sha:
        raise AssertionError("diagnostic inference mutated the completed model")
    if rollout_count != IMAGE_COUNT or len(per_image) != IMAGE_COUNT:
        raise AssertionError("diagnostic did not use exactly one trace for each of 64 TRAIN images")
    changed_rows = sum(row["changed_assignment_rows"] for row in per_image)
    mean_delta = float(np.mean(
        [row["paired_delta_permuted_minus_original"] for row in per_image]))
    if changed_rows == 0:
        interpretation = "inconclusive_no_assignment_rows_changed"
    elif mean_delta < 0:
        interpretation = "permuted_nll_better_descriptive_objective_alignment_failure"
    else:
        interpretation = "original_assignment_not_worse_descriptive_only"
    del decoder, assignment, gamma_content, shuffled, assignments, gamma_rows, rgb_rows, all_rgb
    return {
        "arm": arm,
        "rollouts": rollout_count,
        "live_tail_steps": train.TAIL,
        "model_state_sha256_before": before_sha,
        "model_state_sha256_after": after_sha,
        "interpretation": interpretation,
        "mean_changed_row_fraction": float(np.mean(
            [row["changed_row_fraction"] for row in per_image])),
        "mean_eligible_rows": float(np.mean([row["eligible_rows"] for row in per_image])),
        "mean_paired_delta_permuted_minus_original": mean_delta,
        "per_image": per_image,
    }


def diagnose(*, device="cuda:0"):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered SW0150 color-bin diagnostic requires CUDA")
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("SW0150 diagnostic protocol is missing")
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8-sig"))
    if (protocol.get("experiment") != NAME or protocol.get("diagnostic") != DIAGNOSTIC
            or protocol.get("data", {}).get("image_count") != IMAGE_COUNT
            or protocol.get("data", {}).get("split") != "registered TRAIN only"
            or protocol.get("data", {}).get("ground_truth_used") is not False
            or protocol.get("data", {}).get("masks_read") is not False
            or protocol.get("data", {}).get("optimizer_updates") != 0
            or protocol.get("models") != list(train.ARMS)
            or protocol.get("rollout") != {
                "steps": 1024, "settle_steps": 512, "live_tail_steps": 64,
                "one_actual_rollout_per_image_per_model": True,
                "shared_trace_for_original_and_intervened_scores": True}
            or protocol.get("intervention", {}).get("seed") != PERMUTATION_SEED
            or protocol.get("intervention", {}).get("bins_per_channel") != 8
            or protocol.get("intervention", {}).get("assignment") != "permute P rows only within each image"
            or protocol.get("intervention", {}).get("recompute_decoder_pooling_and_patch_centers") is not True
            or protocol.get("intervention", {}).get("gamma") != "held fixed and detached by the registered likelihood"
            or protocol.get("readout", {}).get("objective") != "unchanged registered SW0150 Gaussian mixture NLL"
            or protocol.get("readout", {}).get("background_only", "").startswith("K=1 is valid") is False
            or "zero-mass padded columns" not in protocol.get("readout", {}).get("padded_slots", "")):
        raise ValueError("SW0150 color-bin diagnostic protocol does not match implementation")

    admission = train.validate_training_inputs()
    source = admission["source"]
    training = {arm: train.validate_training(arm) for arm in train.ARMS}
    _validate_pair_lineage(admission, training)
    ids = source["ids"][:IMAGE_COUNT]
    rows = source["rows"][:IMAGE_COUNT]
    if len(ids) != IMAGE_COUNT or len(set(ids)) != IMAGE_COUNT or len(rows) != IMAGE_COUNT:
        raise ValueError("source ordered TRAIN prefix does not contain 64 unique images")

    cache = np.load(train.sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered SW0150 RGB cache schema changed")
    asset_record = admission["assets"]
    if train.sha(train.sw130.TRAIN_RGB) != asset_record.get("train_cache_sha256"):
        raise ValueError("registered source TRAIN RGB bytes changed after preflight validation")

    arms = {}
    for arm in train.ARMS:
        foundation, eval_source, assets, trained = evaluate._load_trained_foundation(arm, device)
        if (eval_source["ids"] != source["ids"]
                or eval_source["core_sha256"] != source["core_sha256"]
                or assets != admission["assets"]
                or trained["manifest_sha256"] != training[arm]["manifest_sha256"]):
            raise ValueError(f"{arm} fresh diagnostic model failed source/training binding")
        arms[arm] = _diagnose_arm(arm, foundation=foundation, source=source,
                                  admission=admission, rgb_cache=cache)
        del foundation

    fingerprint = implementation_fingerprint()
    report = {
        "status": "complete",
        "experiment": NAME,
        "diagnostic": DIAGNOSTIC,
        "ground_truth_used": False,
        "masks_read": False,
        "optimizer_updates": 0,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "implementation_fingerprint": fingerprint,
        "implementation_sha256": hashlib.sha256(json.dumps(
            fingerprint, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest(),
        "source": {
            "seed": train.SEED,
            "core_sha256": source["core_sha256"],
            "manifest_sha256": train.sha(source["manifest_path"]),
            "training_ids_sha256": train.ids_sha(ids),
            "rgb_asset_validation": admission["assets"],
            "preflight_sha256": admission["report_sha256"],
            "warm_sha256": admission["warm_sha256"],
            "first64_image_ids": ids,
            "first64_pool_rows": [int(value) for value in rows],
        },
        "training": {arm: {
            "manifest_sha256": training[arm]["manifest_sha256"],
            "checkpoint_sha256": training[arm]["checkpoint_sha256"],
            "optimizer_sha256": train.sha(training[arm]["optimizer"]),
            "history_sha256": train.sha(training[arm]["history"]),
            "implementation_sha256": training[arm]["manifest"]["implementation_sha256"],
        } for arm in train.ARMS},
        "intervention": {
            "seed": PERMUTATION_SEED,
            "bins_per_channel": 8,
            "patch_mean": "native 4x4 RGB patch mean / 255",
            "within_image_joint_bin_only": True,
            "recomputed_decoder_pooling_and_centers": True,
            "gamma_fixed_detached": True,
            "trace_per_image_per_arm": 1,
            "sigma2": admission["sigma2"],
        },
        "arms": arms,
        "interpretation_limit": (
            "Objective-alignment diagnostic only; a positive permuted-minus-original NLL is not a "
            "causal, segmentation, generalization, or promotion result."
        ),
    }
    del cache
    return report


def _write_create_once(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserve existing diagnostic result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"preserve existing diagnostic result: {args.output}")
    report = diagnose(device=args.device)
    _write_create_once(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "ground_truth_used": False, "optimizer_updates": 0}, allow_nan=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
