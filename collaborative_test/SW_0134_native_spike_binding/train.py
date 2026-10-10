"""SW0134 fixed-budget paired training stage; consumes immutable preflights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0134_native_spike_binding import run
from collaborative_test.SW_0134_native_spike_binding import preflight_queue

ROOT = run.ROOT
HERE = run.HERE
PILOT_SEED = 1
PASSES = 16
UPDATES = 4096


def _validate_preflight(seed, archive, assets):
    path = Path(archive) / f"preflight_seed{seed}.json"
    if not path.is_file():
        raise RuntimeError(f"required SW0134 seed{seed} preflight is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    source = run.source_contract(seed)
    ids = source[4]
    expected_id_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    seed0_report = seed0_sha = None
    if seed:
        seed0_path = Path(archive) / "preflight_seed0.json"
        if not seed0_path.is_file():
            raise RuntimeError("SW0134 seed1/2 require the passed seed0 calibration record")
        seed0_report = json.loads(seed0_path.read_text(encoding="utf-8"))
        seed0_sha = run.sha(seed0_path)
    pool_sha = hashlib.sha256(np.asarray(source[3], dtype="<i8").tobytes()).hexdigest()
    if not preflight_queue.validate_preflight_record(
            record, seed, fingerprint=run.implementation_fingerprint(),
            source_sha=source[5], source_manifest_sha=run.sha(source[1]),
            training_ids=ids, pool_sha=pool_sha, asset_hashes=assets,
            archive=archive, seed0_report=seed0_report, seed0_report_sha=seed0_sha):
        raise RuntimeError(f"SW0134 seed{seed} preflight evidence failed canonical validation")
    if (record.get("status") != "passed"
            or record.get("experiment") != "SW0134_native_spike_binding"
            or record.get("seed") != seed
            or record.get("implementation_fingerprint") != run.implementation_fingerprint()
            or record.get("source_core_sha256") != run.source97.EXPECTED_SOURCE_SHAS[seed]
            or record.get("source_manifest_sha256") != run.sha(source[1])
            or record.get("training_ids") != ids
            or record.get("training_ids_sha256") != expected_id_sha
            or record.get("pool_indices_sha256") != hashlib.sha256(
                np.asarray(source[3], dtype="<i8").tobytes()).hexdigest()
            or record.get("batch_size") != run.BATCH
            or record.get("train_time_steps") != run.TRAIN_STEPS
            or record.get("settle_steps") != run.TRAIN_SETTLE
            or record.get("live_tail_steps") != run.TAIL
            or record.get("asset_hashes") != assets
            or record.get("ground_truth_used") is not False):
        raise RuntimeError(f"SW0134 seed{seed} preflight does not bind current registered recipe")
    if record.get("lambda_source_seed") != 0:
        raise RuntimeError(f"SW0134 seed{seed} preflight does not bind seed0 lambda")
    value = float(record.get("lambda", float("nan")))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"SW0134 seed{seed} preflight lambda is invalid")
    for arm in ("actual_joint", "gate_joint"):
        info = record.get("warm_artifacts", {}).get(arm, {})
        warm_path = Path(info.get("path", ""))
        if not warm_path.is_file() or run.sha(warm_path) != info.get("sha256"):
            raise RuntimeError(f"SW0134 seed{seed} {arm} warm artifact missing or changed")
        warm = torch.load(warm_path, map_location="cpu", weights_only=True)
        if (warm.get("seed") != seed or warm.get("arm") != arm
                or warm.get("updates") != run.WARMUP_UPDATES
                or warm.get("asset_hashes") != assets
                or warm.get("implementation_fingerprint") != run.implementation_fingerprint()
                or warm.get("source_core_sha256") != run.source97.EXPECTED_SOURCE_SHAS[seed]
                or warm.get("training_ids") != [int(x) for x in ids[:512]]):
            raise RuntimeError(f"SW0134 seed{seed} {arm} warm artifact contract mismatch")
        steps = []
        for state in warm["optimizer_state_dict"].get("state", {}).values():
            step = state.get("step")
            steps.append(int(step.item() if torch.is_tensor(step) else step))
        expected_params = sum(1 for p in run.NativeSpikeSlotBinder().parameters()) + sum(
            1 for p in run.RelativeSlotRGBDecoder().parameters())
        if len(steps) != expected_params or set(steps) != {run.WARMUP_UPDATES}:
            raise RuntimeError(f"SW0134 seed{seed} {arm} Adam moments are not exactly step 32")
        if not math.isfinite(float(info.get("loss_first_last", [float("nan")])[0])):
            raise RuntimeError(f"SW0134 seed{seed} {arm} warm loss is not finite")
    return record, value, run.sha(path)


def _head_parameters(binder, decoder):
    return list(binder.parameters()) + list(decoder.parameters())


def _restore_warm(preflight, arm, device, binder, decoder, optimizer):
    arm_key = "actual_joint" if arm == "actual_frozen" else arm
    warm_path = Path(preflight["warm_artifacts"][arm_key]["path"])
    warm = run._read_warm(warm_path, device, binder, decoder,
                           seed=PILOT_SEED, arm=arm_key,
                           ids=run.source_contract(PILOT_SEED)[4],
                           assets=preflight["asset_hashes"])
    optimizer.load_state_dict(warm["optimizer_state_dict"])
    return warm, run.sha(warm_path)


def _optimizer_steps(optimizer_state):
    if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("state"), dict):
        return None
    steps = []
    for state in optimizer_state["state"].values():
        if "step" not in state:
            return None
        value = state["step"]
        steps.append(int(value.item() if torch.is_tensor(value) else value))
        if any(torch.is_tensor(t) and not bool(torch.isfinite(t).all())
               for t in state.values()):
            return None
    return steps


def validate_completed_training(seed, arm, folder, *, assets=None, archive=None):
    """Reconstruct the full registered 16-pass history and optimizer steps."""
    folder = Path(folder)
    manifest_path, marker_path = folder / "manifest.json", folder / "TRAINING_COMPLETED.json"
    checkpoint_path, optimizer_path, history_path = (
        folder / "checkpoint.pt", folder / "optimizers.pt", folder / "history.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    source = run.source_contract(seed)
    assets = run.sw130.validate_rgb_assets() if assets is None else assets
    archive = Path(archive or run.ARCHIVE)
    preflight_path = archive / f"preflight_seed{seed}.json"
    ids, pool = source[4], np.asarray(source[3], dtype=np.int64)
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if (manifest.get("status") != "training_complete"
            or manifest.get("experiment") != "SW0134_native_spike_binding"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("updates") != UPDATES or manifest.get("passes") != PASSES
            or manifest.get("batch_size") != run.BATCH
            or manifest.get("unique_training_images") != len(ids)
            or manifest.get("total_image_exposures") != len(ids) * PASSES
            or manifest.get("training_ids") != ids
            or manifest.get("training_ids_sha256") != ids_sha
            or manifest.get("source_core_sha256") != run.source97.EXPECTED_SOURCE_SHAS[seed]
            or manifest.get("source_manifest_sha256") != run.sha(source[1])
            or manifest.get("implementation_fingerprint") != run.implementation_fingerprint()
            or manifest.get("trainer_sha256") != run.sha(Path(__file__))
            or manifest.get("asset_hashes") != assets
            or Path(manifest.get("preflight_path", "")).resolve() != preflight_path.resolve()
            or not preflight_path.is_file()
            or run.sha(preflight_path) != manifest.get("preflight_sha256")
            or manifest.get("lambda_source_seed") != 0
            or not math.isfinite(float(manifest.get("lambda", float("nan"))))
            or run.sha(checkpoint_path) != manifest.get("checkpoint_sha256")
            or run.sha(optimizer_path) != manifest.get("optimizer_file_sha256")
            or run.sha(history_path) != manifest.get("history_sha256")
            or marker.get("status") != "training_complete"
            or marker.get("manifest_sha256") != run.sha(manifest_path)
            or marker.get("checkpoint_sha256") != run.sha(checkpoint_path)
            or marker.get("updates") != UPDATES):
        raise ValueError(f"SW0134 {arm} seed{seed} completion metadata is incomplete or mismatched")
    preflight_record, preflight_lambda, preflight_sha = _validate_preflight(seed, archive, assets)
    warm_key = "actual_joint" if arm == "actual_frozen" else arm
    expected_warm_sha = preflight_record["warm_artifacts"][warm_key]["sha256"]
    if (preflight_sha != manifest.get("preflight_sha256")
            or preflight_lambda != float(manifest["lambda"])
            or expected_warm_sha != manifest.get("warm_artifact_sha256")):
        raise ValueError("SW0134 training artifact does not bind its passed preflight/lambda/warm state")

    expected_passes = []
    expected_batches = []
    for pass_index, rows, image_ids, order_sha in _iter_passes(pool, ids, seed):
        expected_passes.append({"pass_index": pass_index, "image_ids_sha256": order_sha,
                                "first_ids": image_ids[:8].tolist(),
                                "last_ids": image_ids[-8:].tolist()})
        for start in range(0, len(rows), run.BATCH):
            expected_batches.append((pass_index, start,
                                     [int(v) for v in image_ids[start:start + run.BATCH]]))
    if manifest.get("pass_orders") != expected_passes:
        raise ValueError("SW0134 training manifest pass-order provenance does not reproduce")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history.get("records")
    if history.get("passes") != expected_passes or not isinstance(records, list) or len(records) != UPDATES:
        raise ValueError("SW0134 history does not contain the full registered 4096 updates")
    for index, (row, expected) in enumerate(zip(records, expected_batches), start=1):
        pass_index, start, batch_ids = expected
        if (row.get("update") != index or row.get("pass_index") != pass_index
                or row.get("batch_start") != start or row.get("image_ids") != batch_ids
                or not all(math.isfinite(float(row.get(key, float("nan"))))
                           for key in ("old_loss", "rgb_loss", "total_loss", "head_gradient_norm",
                                       "joint_gradient_norm"))
                or float(row["head_gradient_norm"]) <= 0
                or (arm != "actual_frozen" and float(row["joint_gradient_norm"]) <= 0)
                or (arm == "actual_frozen" and float(row["joint_gradient_norm"]) != 0)):
            raise ValueError(f"SW0134 training history update {index} violates order or finite checks")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for name in ("wrapped_state_dict", "encoder_state_dict", "binder_state_dict", "decoder_state_dict"):
        values = state.get(name)
        if not isinstance(values, dict) or not values or any(
                torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                for value in values.values()):
            raise ValueError(f"SW0134 checkpoint {name} missing or nonfinite")
    if (state.get("seed") != seed or state.get("arm") != arm
            or state.get("source_core_sha256") != manifest["source_core_sha256"]
            or float(state.get("lambda", float("nan"))) != float(manifest["lambda"])):
        raise ValueError("SW0134 checkpoint identity does not match its manifest")
    optimizers = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    head_steps = _optimizer_steps(optimizers.get("head"))
    expected_head_params = sum(1 for p in run.NativeSpikeSlotBinder().parameters()) + sum(
        1 for p in run.RelativeSlotRGBDecoder().parameters())
    if (head_steps is None or len(head_steps) != expected_head_params
            or set(head_steps) != {run.WARMUP_UPDATES + UPDATES}):
        raise ValueError("SW0134 head Adam state does not continue warm32 through update4096")
    if arm == "actual_frozen":
        if optimizers.get("joint") is not None:
            raise ValueError("SW0134 actual_frozen arm unexpectedly has a joint optimizer")
    else:
        joint_steps = _optimizer_steps(optimizers.get("joint"))
        if joint_steps is None or not joint_steps or set(joint_steps) != {UPDATES}:
            raise ValueError("SW0134 joint Adam state does not contain exactly 4096 steps")
    if optimizers.get("warmup_optimizer_sha256") != manifest.get("warm_artifact_sha256"):
        raise ValueError("SW0134 saved warm optimizer provenance mismatch")
    return manifest, state, checkpoint_path


def _iter_passes(pool, ids, seed):
    for pass_index in range(PASSES):
        if pass_index == 0:
            order = np.arange(len(ids), dtype=np.int64)
        else:
            rng = np.random.RandomState(134000 + int(seed) * 100 + pass_index)
            order = rng.permutation(len(ids))
        rows = np.asarray(pool, dtype=np.int64)[order]
        image_ids = np.asarray(ids, dtype=np.int64)[order]
        digest = hashlib.sha256(np.asarray(image_ids, dtype="<i8").tobytes()).hexdigest()
        yield pass_index, rows, image_ids, digest


def train_arm(seed, arm, *, device, output_root, archive=None):
    if seed != PILOT_SEED or arm not in run.ARMS:
        raise ValueError("SW0134 registered pilot is seed1 and the three fixed arms")
    archive = Path(archive or run.ARCHIVE)
    assets = sw130_assets = run.sw130.validate_rgb_assets()
    records = [_validate_preflight(s, archive, assets) for s in run.SEEDS]
    lambdas = [item[1] for item in records]
    if any(value != lambdas[0] for value in lambdas[1:]):
        raise RuntimeError("SW0134 seed0/1/2 preflights do not share the exact same lambda")
    fixed_lambda = lambdas[0]
    pf = records[seed][0]
    output_dir = Path(output_root) / f"{arm}_seed{seed}"
    if output_dir.exists():
        raise FileExistsError(f"preserve existing SW0134 training attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, pool, ids, source_sha = \
        run.load_models(seed, device)
    warm_optimizer = torch.optim.Adam(_head_parameters(binder, decoder), lr=run.HEAD_LR)
    warm, warm_sha = _restore_warm(pf, arm, device, binder, decoder, warm_optimizer)
    named, groups = run.sw130.joint_parameters(wrapped, encoder)
    joint_params = run._unique_params(named)
    joint_optimizer = None
    if arm != "actual_frozen":
        joint_optimizer = torch.optim.Adam(groups)
    else:
        wrapped.requires_grad_(False)
        encoder.requires_grad_(False)
    train_cache = np.load(run.sw130.TRAIN_RGB, mmap_mode="r")
    history = []
    pass_orders = []
    if arm == "actual_frozen":
        # Keep the frozen backbone in inference mode so BatchNorm buffers and
        # dropout behavior cannot drift during decoder-only fitting.
        wrapped.eval(); encoder.eval()
    else:
        wrapped.train(); encoder.train()
    binder.train(); decoder.train()
    update = 0
    try:
        for pass_index, rows, image_ids, order_sha in _iter_passes(pool, ids, seed):
            pass_orders.append({"pass_index": pass_index, "image_ids_sha256": order_sha,
                                "first_ids": image_ids[:8].tolist(),
                                "last_ids": image_ids[-8:].tolist()})
            for start in range(0, len(rows), run.BATCH):
                batch_rows = rows[start:start + run.BATCH].tolist()
                images = run.sw130.read_batch(train_cache, batch_rows, device)
                out = run._batch_forward(wrapped, encoder, patcher, mean, std, clip,
                                         binder, decoder, images, arm,
                                         checkpoint_chunks=True)
                total = out["old"] + fixed_lambda * out["rgb"]
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError(f"nonfinite SW0134 loss at update {update + 1}")
                head = _head_parameters(binder, decoder)
                head_optimizer = warm_optimizer
                head_optimizer.zero_grad(set_to_none=True)
                head_grads = torch.autograd.grad(out["rgb"], head, retain_graph=True,
                                                 allow_unused=True)
                if run._grad_norm(head_grads) <= 0:
                    raise FloatingPointError("empty SW0134 decoder/binder reconstruction gradient")
                for parameter, grad in zip(head, head_grads):
                    parameter.grad = None if grad is None else grad.detach()
                torch.nn.utils.clip_grad_norm_(head, run.CLIP)
                if joint_optimizer is not None:
                    joint_optimizer.zero_grad(set_to_none=True)
                    joint_grads = torch.autograd.grad(total, joint_params,
                                                      retain_graph=False, allow_unused=True)
                    if run._grad_norm(joint_grads) <= 0:
                        raise FloatingPointError("empty SW0134 joint reconstruction/old-loss gradient")
                    for parameter, grad in zip(joint_params, joint_grads):
                        parameter.grad = None if grad is None else grad.detach()
                    torch.nn.utils.clip_grad_norm_(joint_params, run.CLIP)
                warm_optimizer.step()
                if joint_optimizer is not None:
                    joint_optimizer.step()
                    wrapped.project_integrations_()
                update += 1
                history.append({"update": update, "pass_index": pass_index,
                                "batch_start": start,
                                "image_ids": image_ids[start:start + run.BATCH].tolist(),
                                "old_loss": float(out["old"].detach()),
                                "rgb_loss": float(out["rgb"].detach()),
                                "total_loss": float(total.detach()),
                                "head_gradient_norm": run._grad_norm(head_grads),
                                "joint_gradient_norm": (0.0 if joint_optimizer is None else
                                                        run._grad_norm(joint_grads))})
        if update != UPDATES:
            raise AssertionError(f"SW0134 fixed recipe expected {UPDATES} updates, got {update}")
        tensors = {"wrapped_state_dict": wrapped.state_dict(),
                   "encoder_state_dict": encoder.state_dict(),
                   "binder_state_dict": binder.state_dict(),
                   "decoder_state_dict": decoder.state_dict(),
                   "source_core_sha256": source_sha,
                   "seed": seed, "arm": arm, "lambda": fixed_lambda}
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint_sha = run._save_torch_once(checkpoint_path, tensors)
        optimizer_path = output_dir / "optimizers.pt"
        optimizer_sha = run._save_torch_once(optimizer_path, {
            "joint": None if joint_optimizer is None else joint_optimizer.state_dict(),
            "head": warm_optimizer.state_dict(), "warmup_optimizer_sha256": warm_sha})
        history_path = output_dir / "history.json"
        run.write_once(history_path, {"records": history, "passes": pass_orders})
        manifest = {
            "status": "training_complete", "experiment": "SW0134_native_spike_binding",
            "seed": seed, "arm": arm, "updates": update, "passes": PASSES,
            "batch_size": run.BATCH, "unique_training_images": len(ids),
            "total_image_exposures": len(ids) * PASSES, "training_ids": ids,
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "source_core_sha256": source_sha,
            "source_manifest_sha256": run.sha(run.source_contract(seed)[1]),
            "warm_artifact_sha256": warm_sha,
            "preflight_sha256": records[seed][2],
            "preflight_path": str((archive / f"preflight_seed{seed}.json").resolve()),
            "implementation_fingerprint": run.implementation_fingerprint(),
            "trainer_sha256": run.sha(Path(__file__)),
            "asset_hashes": assets,
            "lambda_source_seed": 0, "lambda": fixed_lambda,
            "pass_orders": pass_orders,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "optimizer_file": str(optimizer_path.resolve()),
            "optimizer_file_sha256": optimizer_sha,
            "history_sha256": run.sha(history_path),
            "ground_truth_used_for_training": False,
            "created_unix": time.time(),
        }
        run.write_once(output_dir / "manifest.json", manifest)
        run.write_once(output_dir / "TRAINING_COMPLETED.json", {
            "status": "training_complete", "manifest_sha256": run.sha(output_dir / "manifest.json"),
            "checkpoint_sha256": checkpoint_sha, "updates": update})
        return manifest
    except Exception as exc:
        fail = {"status": "training_failed", "experiment": "SW0134_native_spike_binding",
                "seed": seed, "arm": arm, "updates_completed": update,
                "error": repr(exc), "history_records": len(history),
                "source_core_sha256": source_sha, "warm_artifact_sha256": warm_sha,
                "implementation_fingerprint": run.implementation_fingerprint(),
                "created_unix": time.time()}
        try:
            run.write_once(output_dir / "TRAINING_FAILED.json", fail)
            run.write_once(output_dir / "history_failed.json", {"records": history,
                                                                  "passes": pass_orders})
        except FileExistsError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=run.ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "trained_models/SW0134_native_spike_binding")
    parser.add_argument("--archive", type=Path, default=run.ARCHIVE)
    args = parser.parse_args(argv)
    manifest = train_arm(args.seed, args.arm, device=torch.device(args.device),
                         output_root=args.output_root, archive=args.archive)
    print(json.dumps({"status": manifest["status"], "seed": args.seed,
                      "arm": args.arm, "updates": manifest["updates"],
                      "checkpoint_sha256": manifest["checkpoint_sha256"]}, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
