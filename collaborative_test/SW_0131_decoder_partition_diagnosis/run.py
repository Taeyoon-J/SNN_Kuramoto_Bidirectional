"""Read-only SW0131 TRAIN diagnostic of assignment-to-RGB decoder dependence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
sys.path[:] = [entry for entry in sys.path
               if not entry or Path(entry).resolve() != HERE]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import (
    assignment_weights, patch_centers, reconstruct_one,
)

STEPS = 64
SETTLE = 32
BATCH = 16
CONDITIONS = ("native", "row_shuffled", "image_mean_content")
DECODER_STATES = ("initial_random", "warmed32")
EXPECTED_WARM_SHA = {
    0: "f592570e8453f4bdec27d06ffeab926004999cf1eb1d05247283b2fb552a8876",
    1: "5517f7eb035a6703962d38143d1d07eb8b6c691ec253830c4818cec7f8b9d3c8",
    2: "a65c4a6240afffd84aa9e0b78ed31fd2e1d0fb226d0085c970a514f439e8f3ce",
}


def _tensor_state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (
        HERE / "run.py", HERE / "queue.py", HERE / "protocol.json",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/protocol.json",
        ROOT / "collaborative_test/SW_0106_spike_partition_rgb/partition_rgb.py",
        ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
        ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
        ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
        ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    merged = dict(sw130.implementation_fingerprint())
    merged.update({path.relative_to(ROOT).as_posix(): sw130.sha(path) for path in paths})
    audit_path = HERE / "warm_decoder_source_audit_20261009.json"
    merged[audit_path.relative_to(ROOT).as_posix()] = sw130.sha(audit_path)
    return merged


def _reconstruct_condition(q, hard, gamma_features, target, decoder, condition,
                           image_ids, image_index):
    if condition == "native":
        return reconstruct_one(q, hard, gamma_features, target, decoder, True)
    if condition == "row_shuffled":
        global_index = int(image_index)
        generator = torch.Generator(device="cpu").manual_seed(130 + global_index)
        permutation = torch.randperm(256, generator=generator, device="cpu")
        shuffled_hard = hard[permutation.to(device=hard.device)]
        return reconstruct_one(q, shuffled_hard, gamma_features, target, decoder, True)
    if condition != "image_mean_content":
        raise ValueError(f"unregistered decoder diagnostic condition: {condition}")

    weights, soft = assignment_weights(q, hard, credit=True)
    # Mirror SW0106's detached content contract, but make each slot's content
    # identical so the decoded image cannot depend on assignment membership.
    mean_content = gamma_features.detach().mean(dim=0, keepdim=True)
    content = mean_content.expand(hard.shape[1], -1)
    decoded = decoder(content, patch_centers(content.device, content.dtype))
    prediction = torch.einsum("nk,knc->nc", weights, decoded)
    loss = torch.nn.functional.mse_loss(prediction, target)
    if not torch.equal(weights.detach(), hard):
        raise AssertionError("image-mean diagnostic changed the hard forward partition")
    return prediction, loss, {"W": weights, "P": soft, "K": int(hard.shape[1]),
                             "group_sizes": hard.sum(0).detach()}


def _finite_norm(values):
    if values is None:
        return 0.0
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite diagnostic gradient")
    return float(torch.linalg.vector_norm(values.detach().double()))


def _warmup_inputs(seed):
    report_path = sw130.ARCHIVE / f"preflight_seed{seed}.json"
    warm_path = sw130.ARCHIVE / f"preflight_decoder_seed{seed}.pt"
    audit_path = HERE / "warm_decoder_source_audit_20261009.json"
    if not audit_path.is_file() or not warm_path.is_file():
        raise FileNotFoundError(f"seed{seed} registered warm decoder audit/artifact is required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audited = [row for row in audit if row.get("seed") == seed]
    if (len(audited) != 1 or audited[0].get("sha256") != EXPECTED_WARM_SHA[seed]
            or audited[0].get("schema") != ["decoder_state_dict", "optimizer_state_dict"]
            or audited[0].get("all_decoder_finite") is not True
            or len(audited[0].get("adam_steps", [])) != 6
            or set(map(float, audited[0].get("adam_steps", []))) != {32.0}):
        raise ValueError(f"seed{seed} warm decoder is not in the registered SHA audit")
    warm_sha = sw130.sha(warm_path)
    if warm_sha != EXPECTED_WARM_SHA[seed]:
        raise ValueError(f"seed{seed} warm decoder SHA differs from the registered artifact")
    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("decoder_warmup_artifact_sha256") != warm_sha:
            raise ValueError(f"seed{seed} warm decoder SHA differs from its preflight record")
    warm = torch.load(warm_path, map_location="cpu", weights_only=True)
    if set(warm) != {"decoder_state_dict", "optimizer_state_dict"}:
        raise ValueError("warm decoder artifact schema changed")
    states = warm["optimizer_state_dict"].get("state", {})
    steps = [int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
             for value in states.values() if "step" in value]
    if len(steps) != 6 or set(steps) != {32}:
        raise ValueError("warm decoder Adam state does not represent exactly 32 updates")
    audit_digest = sw130.sha(audit_path)
    return report_path if report is not None else None, report, warm_path, warm_sha, warm, audit_path, audit_digest


def diagnose_seed(seed, device, output_path):
    if seed not in sw130.SEEDS:
        raise ValueError("seed must be one of the registered SW0097 seeds 0, 1, or 2")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"preserve existing SW0131 output: {output_path}")

    assets = sw130.validate_rgb_assets()
    source_checkpoint, source_manifest_path, source_manifest, pool, ids, source_sha = \
        sw130.source_contract(seed)
    report_path, preflight_report, warm_path, warm_sha, warm, audit_path, audit_sha = _warmup_inputs(seed)
    if preflight_report is not None and (preflight_report.get("seed") != seed
            or preflight_report.get("source_core_sha256") != source_sha
            or preflight_report.get("matched_training_ids") != ids
            or preflight_report.get("asset_hashes") != assets):
        raise ValueError("preflight, source/order, or RGB asset provenance does not match current inputs")

    torch.manual_seed(117 + seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(117 + seed)
    wrapped, encoder, patcher, mean, std, clip, decoder, loaded_pool, loaded_ids, loaded_sha = \
        sw130.load_models(seed, device, "phase")
    if (loaded_sha != source_sha or loaded_ids != ids
            or not np.array_equal(loaded_pool, pool)):
        raise AssertionError("SW0130 loader differs from exact registered source/data order")
    initial_state = {key: value.detach().clone() for key, value in decoder.state_dict().items()}
    initial_sha = _tensor_state_sha(initial_state)
    warm_state = warm["decoder_state_dict"]
    if set(warm_state) != set(initial_state) or any(
            warm_state[key].shape != initial_state[key].shape
            or warm_state[key].dtype != initial_state[key].dtype
            or not torch.isfinite(warm_state[key]).all() for key in initial_state):
        raise ValueError("warm decoder tensors differ in keys/shape/dtype or are nonfinite")
    warm_sha_state = _tensor_state_sha(warm_state)
    decoder_states = {
        "initial_random": initial_state,
        "warmed32": {key: value.detach().clone() for key, value in warm_state.items()},
    }
    core_before = {key: value.detach().clone() for key, value in wrapped.state_dict().items()}
    encoder_before = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    decoder_before = {key: value.detach().clone() for key, value in decoder.state_dict().items()}
    named, _groups = sw130.joint_parameters(wrapped, encoder)
    parameters = [value for _, value in named]
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    criterion = sw130.make_criterion()
    wrapped.eval(); encoder.eval(); decoder.eval()
    rows = []
    batch_diagnostics = []
    batch_diagnostics = []
    first64_ids = ids[:64]
    if len(first64_ids) != 64:
        raise AssertionError("registered source order does not provide exactly 64 diagnostic IDs")

    for batch_index in range(4):
        start = batch_index * BATCH
        batch_indices = pool[start:start + BATCH].tolist()
        batch_ids = ids[start:start + BATCH]
        images = sw130.read_batch(train_cache, batch_indices, device)
        gamma, result, q, _labels, hard_rows, groups, target = sw130.forward_batch(
            wrapped, encoder, patcher, mean, std, clip, images, criterion)
        if tuple(gamma.shape) != (BATCH, 8, 256) or tuple(q.shape) != (BATCH, 256, 256):
            raise ValueError("registered first-64 TRAIN batch has unexpected gamma/Q shape")
        if len(hard_rows) != BATCH or len(groups) != BATCH:
            raise ValueError("native QCC hard assignment batch is incomplete")
        image_rows = []
        for decoder_name, state in decoder_states.items():
            decoder.load_state_dict(state, strict=True)
            for condition in CONDITIONS:
                losses, loss_tensors = [], []
                for image_index in range(BATCH):
                    hard = hard_rows[image_index]
                    gamma_features = gamma[image_index].transpose(0, 1)
                    _, loss, _diagnostics = _reconstruct_condition(
                        q[image_index], hard, gamma_features, target[image_index], decoder,
                        condition, batch_ids, start + image_index)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite RGB reconstruction loss")
                    losses.append(float(loss.detach()))
                    loss_tensors.append(loss)
                record = {"decoder_state": decoder_name, "condition": condition,
                          "per_image_mse": losses, "mean_mse": float(np.mean(losses))}
                if condition == "native":
                    batch_loss = torch.stack(loss_tensors).mean()
                    grads = torch.autograd.grad(batch_loss, parameters, retain_graph=True,
                                                allow_unused=True)
                    if any(grad is not None and not torch.isfinite(grad).all() for grad in grads):
                        raise FloatingPointError("nonfinite native RGB family gradient")
                    record["rgb_gradient_norms_by_family"] = sw130.family_norm(named, grads)
                    q_grad = torch.autograd.grad(batch_loss, q, retain_graph=True,
                                                 allow_unused=True)[0]
                    record["q_gradient_norm"] = _finite_norm(q_grad)
                image_rows.append(record)
        for local_index, image_id in enumerate(batch_ids):
            row = {"image_id": int(image_id), "batch_index": batch_index,
                   "K": int(hard_rows[local_index].shape[1]),
                   "native_group_sizes": [int(v) for v in hard_rows[local_index].sum(0).cpu().tolist()]}
            for variant in image_rows:
                row.setdefault("mse", {}).setdefault(variant["decoder_state"], {})[
                    variant["condition"]] = variant["per_image_mse"][local_index]
            rows.append(row)
        # Keep family/Q gradients at batch level; per-image losses remain paired.
        for variant in image_rows:
            variant["batch_index"] = batch_index
            if "rgb_gradient_norms_by_family" in variant:
                batch_diagnostics.append(variant)
        del gamma, result, q, hard_rows, groups, target, images

    if len(rows) != 64:
        raise AssertionError("diagnostic output omitted per-image or gradient records")
    if any(not torch.equal(value, wrapped.state_dict()[key]) for key, value in core_before.items()):
        raise AssertionError("read-only diagnostic changed source core/adapter state")
    if any(not torch.equal(value, encoder.state_dict()[key]) for key, value in encoder_before.items()):
        raise AssertionError("read-only diagnostic changed source encoder state")
    decoder.load_state_dict(initial_state, strict=True)
    if any(not torch.equal(value, decoder.state_dict()[key]) for key, value in decoder_before.items()):
        raise AssertionError("read-only diagnostic changed the in-memory decoder initialization")

    per_image = [row for row in rows if "image_id" in row]
    deltas = {}
    for state_name in DECODER_STATES:
        native_values = [row["mse"][state_name]["native"] for row in per_image]
        for condition in ("row_shuffled", "image_mean_content"):
            compared = [row["mse"][state_name][condition] for row in per_image]
            differences = [value - base for value, base in zip(compared, native_values)]
            deltas[f"{state_name}_{condition}_minus_native"] = {
                "mean": float(np.mean(differences)),
                "positive_count": sum(value > 0 for value in differences),
                "negative_count": sum(value < 0 for value in differences),
                "per_image": differences,
            }
    result = {
        "status": "complete", "experiment": "SW0131_decoder_partition_diagnosis",
        "seed": seed, "image_ids": first64_ids, "images": 64,
        "time_steps": STEPS, "settle": SETTLE, "batch_size": BATCH,
        "conditions": list(CONDITIONS), "decoder_states": list(DECODER_STATES),
        "source_core_sha256": source_sha,
        "source_manifest_sha256": sw130.sha(source_manifest_path),
        "source_updates": source_manifest["steps"],
        "training_ids_sha256": hashlib.sha256(np.asarray(first64_ids, dtype="<i8").tobytes()).hexdigest(),
        "pool_rows_sha256": hashlib.sha256(np.asarray(pool[:64], dtype="<i8").tobytes()).hexdigest(),
        "preflight_path": str(report_path.resolve()) if report_path else None,
        "preflight_sha256": sw130.sha(report_path) if report_path else None,
        "warm_decoder_source_audit_path": str(audit_path.resolve()),
        "warm_decoder_source_audit_sha256": audit_sha,
        "warm_decoder_path": str(warm_path.resolve()), "warm_decoder_sha256": warm_sha,
        "warm_decoder_state_sha256": warm_sha_state,
        "initial_decoder_state_sha256": initial_sha,
        "asset_hashes": assets, "implementation_fingerprint": implementation_fingerprint(),
        "per_image": per_image,
        "batch_gradient_diagnostics": batch_diagnostics,
        "paired64image_deltas": deltas,
        "ground_truth_used": False, "optimizer_updates": 0,
        "source_checkpoint_modified": False, "decoder_artifact_modified": False,
        "completed": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=sw130.SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose_seed(args.seed, torch.device(args.device), args.output)
    print(json.dumps({"status": result["status"], "seed": args.seed,
                      "output": str(args.output), "images": result["images"],
                      "deltas": {key: value["mean"] for key, value in result["paired64image_deltas"].items()}},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
