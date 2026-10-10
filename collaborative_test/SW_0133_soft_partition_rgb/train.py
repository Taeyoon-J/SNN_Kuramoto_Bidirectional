"""Train one SW0133 seed1 arm after all three frozen source preflights pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0133_soft_partition_rgb import preflight_queue, run

PILOT_SEED = 1


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_new(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def _finite(parameters):
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


def _passed_preflights(assets):
    rows = {}
    for seed in run.SEEDS:
        task = {"stage": "preflight", "seed": seed}
        if not preflight_queue.valid_result(task, assets):
            raise RuntimeError(f"seed{seed} preflight is absent or invalid; no training started")
        path = preflight_queue.artifact_path(task)
        rows[seed] = {"path": str(path.resolve()), "sha256": run.sha(path),
                      "row": json.loads(path.read_text(encoding="utf-8"))}
    return rows


def train(seed, arm, device, output):
    if seed != PILOT_SEED or arm not in run.ARMS:
        raise ValueError("the registered SW0133 pilot trains only seed1 and its three paired arms")
    device, output = torch.device(device), Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0133 training attempt: {output}")

    assets = run.sw130.validate_rgb_assets()
    preflights = _passed_preflights(assets)
    source_path, source_manifest_path, source_manifest, pool, ids, source_sha = run.source_contract(seed)
    pf_entry = preflights[seed]
    pf = pf_entry["row"]
    if (pf.get("source_core_sha256") != source_sha
            or pf.get("source_manifest_sha256") != run.sha(source_manifest_path)
            or pf.get("training_ids") != ids
            or pf.get("training_ids_sha256") != hashlib.sha256(
                np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            or pf.get("pool_indices_sha256") != hashlib.sha256(
                np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
            or pf.get("asset_hashes") != assets
            or pf.get("implementation_fingerprint") != run.implementation_fingerprint()):
        raise RuntimeError("seed1 preflight source, order, asset or implementation binding changed")
    fixed_lambda = float(preflights[0]["row"]["lambda"])
    if (pf.get("lambda_source_seed") != 0 or pf.get("lambda") != fixed_lambda
            or pf.get("seed0_lambda_record_sha256") != preflights[0]["sha256"]):
        raise RuntimeError("seed1 does not bind the immutable seed0 shared lambda")
    warm_path = run.ARCHIVE / f"preflight_decoder_seed{seed}.pt"
    if (not warm_path.is_file()
            or run.sha(warm_path) != pf.get("decoder_warmup_artifact_sha256")):
        raise RuntimeError("seed1 warmed decoder/Adam32 artifact is missing or changed")
    warm = torch.load(warm_path, map_location="cpu", weights_only=True)
    if (warm.get("source_core_sha256") != source_sha
            or warm.get("training_ids_sha256") != pf.get("training_ids_sha256")
            or warm.get("asset_hashes") != assets or warm.get("warmup_updates") != 32):
        raise RuntimeError("seed1 decoder warmup artifact provenance mismatch")
    run._validate_warm_optimizer(warm)

    started = time.time()
    owned = False
    try:
        torch.manual_seed(117 + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(117 + seed)
        wrapped, encoder, patcher, mean, std, clip, decoder, loaded_pool, loaded_ids, loaded_sha = \
            run.load_models(seed, device, arm)
        if (loaded_sha != source_sha or loaded_ids != ids
                or not np.array_equal(loaded_pool, pool)):
            raise AssertionError("training loader differs from registered source/data order")
        decoder.load_state_dict(warm["decoder_state_dict"], strict=True)
        named, groups = run.sw130.joint_parameters(wrapped, encoder)
        joint_parameters = [parameter for group in groups for parameter in group["params"]]
        if (len({id(parameter) for parameter in joint_parameters}) != len(joint_parameters)
                or {id(parameter) for parameter in joint_parameters} !=
                   {id(parameter) for _, parameter in named}):
            raise AssertionError("joint optimizer parameters differ from the unique named gradient union")
        decoder_parameters = list(decoder.parameters())
        joint_optimizer = torch.optim.Adam(groups)
        decoder_optimizer = torch.optim.Adam(decoder_parameters, lr=run.DECODER_LR)
        decoder_optimizer.load_state_dict(warm["optimizer_state_dict"])
        warm_steps = [int(state["step"].item() if torch.is_tensor(state["step"])
                           else state["step"])
                      for state in decoder_optimizer.state.values() if "step" in state]
        if len(warm_steps) != 6 or set(warm_steps) != {32}:
            raise AssertionError("SW0133 requires the exact shared six-parameter decoder Adam32 state")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        owned = True
        train_cache = np.load(run.sw130.TRAIN_RGB, mmap_mode="r")
        criterion = run.sw130.make_criterion()
        wrapped.train(); encoder.train(); decoder.train()
        history = []
        assignment_live = arm != "phase_detached"
        for update in range(run.sw130.UPDATES):
            start = update * run.BATCH
            batch_rows = pool[start:start + run.BATCH].tolist()
            if len(batch_rows) != run.BATCH:
                raise AssertionError(f"registered training order ended at update {update + 1}")
            images = run.sw130.read_batch(train_cache, batch_rows, device)
            gamma, result, q, _labels, hard, groups_found, _target = run._batch_forward(
                wrapped, encoder, patcher, mean, std, clip, images)
            _, _, _, plv, theta = result
            primary, _ = criterion(plv=plv, theta=theta)
            q_loss, _ = criterion(plv=q)
            old_loss = primary + 5.0 * q_loss
            prediction, rgb_per_image, details = run._rgb_batch(
                q, hard, gamma, _target, decoder, assignment_live=assignment_live)
            rgb_loss = rgb_per_image.mean()
            objective = old_loss + fixed_lambda * rgb_loss if assignment_live else old_loss
            if (not bool(torch.isfinite(old_loss)) or not bool(torch.isfinite(rgb_loss))
                    or not bool(torch.isfinite(objective))):
                raise FloatingPointError(f"nonfinite SW0133 objective at update {update + 1}")

            joint_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)
            joint_grads = torch.autograd.grad(objective, joint_parameters,
                                              retain_graph=True, allow_unused=True)
            decoder_grads = torch.autograd.grad(rgb_loss, decoder_parameters,
                                                allow_unused=True)
            joint_norm, decoder_norm = run._finite_norm(joint_grads), run._finite_norm(decoder_grads)
            if joint_norm <= 0 or decoder_norm <= 0:
                raise AssertionError(f"empty gradient at update {update + 1}")
            for parameter, gradient in zip(joint_parameters, joint_grads):
                parameter.grad = None if gradient is None else gradient.detach().clone()
            for parameter, gradient in zip(decoder_parameters, decoder_grads):
                parameter.grad = None if gradient is None else gradient.detach().clone()
            torch.nn.utils.clip_grad_norm_(joint_parameters, run.CLIP)
            torch.nn.utils.clip_grad_norm_(decoder_parameters, run.CLIP)
            joint_optimizer.step()
            decoder_optimizer.step()
            wrapped.project_integrations_()
            if not (_finite(wrapped.parameters()) and _finite(encoder.parameters())
                    and _finite(decoder.parameters())):
                raise FloatingPointError(f"nonfinite parameter after update {update + 1}")
            history.append({
                "update": update + 1, "total": float(objective.detach()),
                "old_objective": float(old_loss.detach()), "phase_primary": float(primary.detach()),
                "positive_product_spike": float(q_loss.detach()),
                "full_rgb_reconstruction": float(rgb_loss.detach()),
                "joint_grad_norm_preclip": joint_norm,
                "decoder_grad_norm_preclip": decoder_norm,
                "assignment_live": assignment_live,
                "mean_groups": float(np.mean([len(group) for group in groups_found])),
                "K_max": max(int(row["K"]) for row in details),
                "integration": {key: getattr(wrapped, key).detach().cpu().tolist()
                                for key in ("a_d", "a_m", "b")},
            })
            if update == 0 or (update + 1) % 16 == 0 or update + 1 == run.sw130.UPDATES:
                _atomic_json(output / "progress.json", {
                    "status": "training", "seed": seed, "arm": arm,
                    "update": update + 1, "total_updates": run.sw130.UPDATES,
                    "history": history, "updated": time.time()})

        if len(history) != run.sw130.UPDATES:
            raise AssertionError("training did not complete the registered 256 updates")
        _save_new(output / "core.pt", wrapped.core.state_dict())
        _save_new(output / "integration.pt", {
            "a_d": wrapped.a_d.detach().cpu(), "a_m": wrapped.a_m.detach().cpu(),
            "b": wrapped.b.detach().cpu(), "arm": wrapped.arm})
        _save_new(output / "encoder.pt", encoder.state_dict())
        _save_new(output / "decoder.pt", decoder.state_dict())
        _save_new(output / "optimizers.pt", {
            "joint": joint_optimizer.state_dict(),
            "decoder": decoder_optimizer.state_dict(),
            "seed": seed, "arm": arm, "updates": run.sw130.UPDATES,
            "shared_lambda": fixed_lambda})
        run.write_once(output / "history.json", history)
        hashes = {name: run.sha(output / name) for name in (
            "core.pt", "integration.pt", "encoder.pt", "decoder.pt", "optimizers.pt", "history.json")}
        manifest = {
            "status": "training_complete", "experiment": "SW0133_soft_partition_rgb",
            "seed": seed, "arm": arm, "source_model_seed": seed,
            "source_core_sha256": source_sha, "source_manifest_sha256": run.sha(source_manifest_path),
            "source_steps": int(source_manifest["steps"]),
            "preflight_path": pf_entry["path"], "preflight_sha256": pf_entry["sha256"],
            "preflight_implementation_fingerprint": pf["implementation_fingerprint"],
            "implementation_fingerprint": run.implementation_fingerprint(),
            "trainer_sha256": run.sha(HERE / "train.py"),
            "preflight_queue_sha256": run.sha(HERE / "preflight_queue.py"),
            "seed0_lambda_record_sha256": preflights[0]["sha256"],
            "lambda": fixed_lambda, "lambda_source_seed": 0,
            "decoder_warmup_path": str(warm_path.resolve()),
            "decoder_warmup_sha256": run.sha(warm_path),
            "asset_hashes": assets, "training_ids": ids,
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "pool_indices_sha256": hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest(),
            "shuffle_seed": 117 + seed, "batch_size": run.BATCH,
            "updates": run.sw130.UPDATES, "train_time_steps": run.STEPS,
            "settle_steps": run.SETTLE,
            "learning_rates": {"core_graph": run.sw130.CORE_GRAPH_LR,
                               "encoder": run.sw130.ENCODER_LR,
                               "integration": run.sw130.INTEGRATION_LR,
                               "decoder": run.DECODER_LR},
            "clip_norm": run.CLIP, "objective": "old+lambda*R" if assignment_live else "old only; decoder R",
            "artifact_sha256": hashes,
            "ground_truth_used_for_training": False,
            "started": started, "completed": time.time(),
        }
        run.write_once(output / "manifest.json", manifest)
        with (output / "TRAINING_COMPLETED").open("x", encoding="utf-8") as marker:
            marker.write("SW0133 seed1 registered 256-update paired pilot complete\n")
        return manifest
    except BaseException as exc:
        if owned:
            try:
                run.write_once(output / "failure.json", {
                    "status": "training_failed", "seed": seed, "arm": arm,
                    "error": repr(exc), "failed_at": time.time(),
                    "preflight_sha256": pf_entry["sha256"]})
            except FileExistsError:
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(PILOT_SEED,), required=True)
    parser.add_argument("--arm", choices=run.ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = train(args.seed, args.arm, torch.device(args.device), args.output)
    print(json.dumps({"status": result["status"], "seed": args.seed, "arm": args.arm,
                      "output": str(args.output), "updates": result["updates"]},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
