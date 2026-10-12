"""SW0148 local clone of the SW0146 update loop with durable boundaries.

The immutable SW0146 implementation is imported only for source, objective,
optimizer and preflight helpers. This module owns its training loop and paths.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0146_native32_boundary_q_pilot import run as sw146  # noqa: E402
from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (  # noqa: E402
    inspect_committed_metadata, is_only_uncommitted_temp, load_latest,
    runtime_signature, save_boundary,
)


NAME = "SW0148_checkpointable_boundary_q"
WORK = HERE / "work"
TRAIN = HERE / "training"
EXPECTED_SW146_RUN_SHA256 = "9f94c097e0f1765f32247c729523ab306f1319c508cef4160bdefef1c4dea489"
EXPECTED_SW146_PROTOCOL_SHA256 = "9d86bb93e52e84dee13e160099a3cd91899c9ec16d37dce3f1763b99d3aaca0c"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _train_source_bindings(arm: str, ctx: dict, src: tuple, ids: list[int],
                           f, parity: dict | None, preflights: dict) -> dict:
    parent_run = Path(sw146.__file__).resolve()
    parent_protocol = parent_run.with_name("protocol.json")
    if (_sha(parent_run) != EXPECTED_SW146_RUN_SHA256
            or _sha(parent_protocol) != EXPECTED_SW146_PROTOCOL_SHA256):
        raise ValueError("frozen SW0146 trainer/protocol bytes do not match the verified parent")
    return {
        "experiment": NAME,
        "parent_experiment": sw146.NAME,
        "arm": arm,
        "seed": sw146.SEED,
        "source_checkpoint_sha256": src[5],
        "source_manifest_sha256": sw146.sha(src[1]),
        "source_training_ids_sha256": f.provenance["source_training_ids_sha256"],
        "rgb_asset_validation": f.provenance["rgb_asset_validation"],
        "ordered_train_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "admission_sha256": ctx["admission_sha256"],
        "lambda_A": float(ctx["lambda_A"]),
        "scale": float(ctx["scale"]),
        "preflight_sha256": {name: sw146.sha(sw146.preflight_path(name))
                              for name in sw146.ARMS},
        "parent_implementation_sha256": sw146.implementation_sha(),
        "parent_run_sha256": _sha(parent_run),
        "parent_protocol_sha256": _sha(parent_protocol),
        "parent_implementation_fingerprint": sw146.fingerprint(),
        "trainer_sha256": _sha(HERE / "train.py"),
        "checkpoint_helper_sha256": _sha(HERE / "checkpoint.py"),
        "protocol_sha256": _sha(HERE / "protocol.json"),
        "initial_source_Q_parity": parity,
        "initial_parameter_sha256": sw146._param_sha(sw146._groups(f, arm)),
        "initial_frozen_native_sha256": sw146._frozen_native_sha(f),
        "batch": sw146.BATCH,
        "microbatch": sw146.MICRO,
        "updates": sw146.UPDATES,
        "total_steps": sw146.STEPS,
        "settle": sw146.SETTLE,
        "live_tail": sw146.TAIL,
        "optimizer_groups": list(sw146._groups(f, arm)),
        "preflight_parameter_sha256": {
            name: preflights[name]["post_parameter_sha256"] for name in sw146.ARMS
        },
        "ground_truth_used": False,
    }


def _safe_torch_save(path: Path, obj) -> None:
    with path.open("xb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _finalize(arm: str, out: Path, checkpoint_dir: Path, *, f, opt, history,
              bindings, parity, before_sha, before_frozen, started,
              completed_updates: int) -> dict:
    if completed_updates != sw146.UPDATES:
        raise ValueError("cannot finalize before all registered logical updates")
    if out.exists():
        raise FileExistsError(f"preserve existing completed SW0148 result: {out}")
    after_frozen = sw146._frozen_native_sha(f)
    if arm == "encoder_graph" and before_frozen != after_frozen:
        raise AssertionError("encoder_graph arm modified its frozen native core")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        diagnostic = sw146._first64_boundary_diagnostic(
            f, _teacher_from_source(f, bindings), np.load(sw146.sw130.TRAIN_RGB, mmap_mode="r"),
            sw146.ordered_train()[1], bindings["scale"], next(f.encoder.parameters()).device)
        diagnostic_path = staging / "first64_boundary_diagnostic.json"
        _write_once(diagnostic_path, diagnostic)
        checkpoint = staging / "checkpoint.pt"
        optimizer = staging / "optimizer.pt"
        history_path = staging / "history.json"
        _safe_torch_save(checkpoint, {"wrapped": f.wrapped.state_dict(),
                                      "encoder": f.encoder.state_dict()})
        _safe_torch_save(optimizer, opt.state_dict())
        _write_once(history_path, {"updates": history})
        after_sha = sw146._param_sha(sw146._groups(f, arm))
        manifest = {
            "status": "training_complete", "experiment": NAME,
            "parent_experiment": sw146.NAME, "arm": arm, "seed": sw146.SEED,
            "parent_run_sha256": bindings["parent_run_sha256"],
            "parent_protocol_sha256": bindings["parent_protocol_sha256"],
            "updates": sw146.UPDATES, "completed_updates": completed_updates,
            "logical_batch": sw146.BATCH, "microbatch": sw146.MICRO,
            "exposures": sw146.UPDATES * sw146.BATCH,
            "train_ids": [image_id for row in history for image_id in row["image_ids"]],
            "source_core_sha256": bindings["source_checkpoint_sha256"],
            "source_manifest_sha256": bindings["source_manifest_sha256"],
            "source_training_ids_sha256": bindings["source_training_ids_sha256"],
            "ordered_train_ids_sha256": bindings["ordered_train_ids_sha256"],
            "train_ids_sha256": bindings["ordered_train_ids_sha256"],
            "rgb_asset_validation": bindings["rgb_asset_validation"],
            "lambda_A": bindings["lambda_A"], "scale": bindings["scale"],
            "preflight_sha256": bindings["preflight_sha256"],
            "initial_source_Q_parity": parity,
            "parameter_sha256_before": before_sha,
            "parameter_sha256_after": after_sha,
            "frozen_native_core_sha256_before": before_frozen,
            "frozen_native_core_sha256_after": after_frozen,
            "checkpoint_sha256": _sha(checkpoint),
            "optimizer_sha256": _sha(optimizer),
            "history_sha256": _sha(history_path),
            "durable_boundary_sha256": _sha(checkpoint_dir / "boundary.ptx"),
            "optimizer_step_range": sw146._optimizer_range(opt),
            "implementation_sha256": bindings["trainer_sha256"],
            "checkpoint_helper_sha256": bindings["checkpoint_helper_sha256"],
            "protocol_sha256": bindings["protocol_sha256"],
            "parent_implementation_sha256": bindings["parent_implementation_sha256"],
            "bindings": bindings,
            "first64_boundary_diagnostic_sha256": _sha(diagnostic_path),
            "ground_truth_used": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_once(staging / "manifest.json", manifest)
        # Directory rename is the completion commit; staging is unique and kept
        # if the rename fails, while an existing final output is never replaced.
        os.rename(staging, out)
        return manifest
    except BaseException:
        # Preserve incomplete staging for inspection; next invocation uses a
        # new unique staging directory and never overwrites it.
        raise


def _teacher_from_source(student, bindings):
    """Construct a fresh immutable SW0097 teacher for final diagnostics."""
    device = next(student.encoder.parameters()).device
    teacher, _src = sw146._source_foundation(device)
    if teacher.provenance["source_core_sha256"] != bindings["source_checkpoint_sha256"]:
        raise ValueError("teacher source differs from checkpoint lineage")
    for p in teacher.wrapped.parameters():
        p.requires_grad_(False)
    for p in teacher.encoder.parameters():
        p.requires_grad_(False)
    teacher.wrapped.eval()
    teacher.encoder.eval()
    return teacher


def _one_logical_update(arm, update, *, f, teacher, opt, params, groups,
                        cache, rows, ordered_ids, scale, lambda_a, commit=None):
    """Frozen SW0146 update ordering, with the durable callback after projection."""
    opt.zero_grad(set_to_none=True)
    parts = []
    for micro in range(sw146.BATCH // sw146.MICRO):
        start = update * sw146.BATCH + micro * sw146.MICRO
        rgb = sw146._rgb_rows(cache, rows[start:start + sw146.MICRO],
                              next(f.encoder.parameters()).device)
        loss, boundary, source_q, _trace = sw146._loss_micro(
            f, teacher, rgb, scale, lambda_a)
        (loss / (sw146.BATCH // sw146.MICRO)).backward()
        parts.append({"loss": float(loss.detach().cpu()),
                      "boundary": float(boundary.detach().cpu()),
                      "source_Q": float(source_q.detach().cpu())})
        del loss, boundary, source_q, _trace, rgb
    gradients = sw146._norms(groups)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if arm == "joint":
        with torch.no_grad():
            f.wrapped.a_d.clamp_(0, 1)
            f.wrapped.a_m.clamp_(0, 1)
    row = {"update": update + 1,
           "image_ids": ordered_ids[update * sw146.BATCH:(update + 1) * sw146.BATCH],
           "microbatches": parts, "gradient_norms": gradients}
    if commit is not None:
        commit(row)
    return row


def _execute_remaining_updates(completed: int, maximum: int, update_fn):
    """Run only updates after the last durable boundary."""
    if not 0 <= completed <= maximum:
        raise ValueError("invalid completed-update boundary")
    for update in range(completed, maximum):
        update_fn(update)


def train_arm(arm: str, *, device="cuda:0", admission=sw146.ADMISSION,
              preflight_root=sw146.PREFLIGHT, work_root=WORK, output_root=TRAIN,
              max_updates=sw146.UPDATES) -> dict:
    """Run or exactly resume one SW0146-identical arm under SW0148 paths."""
    if arm not in sw146.ARMS or max_updates != sw146.UPDATES:
        raise ValueError("SW0148 arm and update budget are fixed to SW0146")
    ctx = sw146._admission_context(admission)
    preflights = sw146._require_both_preflights(ctx, preflight_root)
    if not ctx.get("admission_sha256"):
        raise ValueError("SW0148 requires byte-hash-bound SW0146 admission")
    dev = torch.device(device)
    f, src = sw146._source_foundation(dev)
    teacher, _ = sw146._source_foundation(dev)
    for p in teacher.wrapped.parameters():
        p.requires_grad_(False)
    for p in teacher.encoder.parameters():
        p.requires_grad_(False)
    teacher.wrapped.eval()
    teacher.encoder.eval()

    groups = sw146._groups(f, arm)
    sw146._set_trainable(f, groups)
    opt, params = sw146._optimizer(groups)
    ids, rows, _ = sw146.ordered_train()
    cache = np.load(sw146.sw130.TRAIN_RGB, mmap_mode="r")
    # Source construction is checked against the teacher only at update zero.
    # A trained student must never be required to equal the teacher on resume.
    initial_parameter_sha = sw146._param_sha(groups)
    initial_frozen_sha = sw146._frozen_native_sha(f)
    bindings = _train_source_bindings(arm, ctx, src, ids, f, None, preflights)
    runtime = runtime_signature(dev)
    checkpoint_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    out = Path(output_root) / f"{arm}_seed1"
    if out.exists():
        raise FileExistsError(f"preserve existing SW0148 training result: {out}")
    started = time.perf_counter()

    if checkpoint_dir.exists() and (checkpoint_dir / "boundary.ptx").exists():
        metadata = inspect_committed_metadata(checkpoint_dir)
        saved_bindings = metadata["bindings"]
        current_base = dict(bindings)
        saved_parity = saved_bindings.pop("initial_source_Q_parity", None)
        if saved_bindings != current_base or not isinstance(saved_parity, dict) or saved_parity.get("exact") is not True:
            raise ValueError("committed checkpoint lineage or original source parity differs")
        bindings = {**current_base, "initial_source_Q_parity": saved_parity}
        state = load_latest(checkpoint_dir, wrapped=f.wrapped, encoder=f.encoder,
                            optimizer=opt, expected_bindings=bindings,
                            expected_runtime=runtime, device=dev,
                            max_updates=sw146.UPDATES)
        completed = state["completed_updates"]
        history = state["history"]
        parity = bindings["initial_source_Q_parity"]
        # Bind the original fresh-source hash recorded at k=0, not the current
        # trained model, to the resumed lineage.
        initial_parameter_sha = bindings["initial_parameter_sha256"]
        initial_frozen_sha = bindings["initial_frozen_native_sha256"]
    else:
        if checkpoint_dir.exists():
            if not is_only_uncommitted_temp(checkpoint_dir):
                raise ValueError("uncommitted SW0148 work directory contains unknown artifacts")
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
        completed = 0
        history = []
        parity = sw146._initial_source_parity(f, teacher, cache, rows, dev)
        bindings["initial_source_Q_parity"] = parity
        save_boundary(checkpoint_dir, completed_updates=0, wrapped=f.wrapped,
                      encoder=f.encoder, optimizer=opt, history=history,
                      bindings=bindings, runtime=runtime, device=dev)

    if completed == sw146.UPDATES:
        return _finalize(arm, out, checkpoint_dir, f=f, opt=opt, history=history,
                         bindings=bindings, parity=parity,
                         before_sha=initial_parameter_sha,
                         before_frozen=initial_frozen_sha, started=started,
                         completed_updates=completed)

    f.wrapped.train()
    f.encoder.train()
    def commit_update(row):
        history.append(row)
        # The durable boundary is strictly after Adam, projection and history.
        save_boundary(checkpoint_dir, completed_updates=row["update"],
                      wrapped=f.wrapped, encoder=f.encoder, optimizer=opt,
                      history=history, bindings=bindings, runtime=runtime, device=dev)
        if row["update"] % 16 == 0:
            print(json.dumps({"stage": "train", "arm": arm,
                              "completed_updates": row["update"],
                              "total_updates": sw146.UPDATES}, allow_nan=False), flush=True)

    _execute_remaining_updates(
        completed, sw146.UPDATES,
        lambda update: _one_logical_update(
            arm, update, f=f, teacher=teacher, opt=opt, params=params,
            groups=groups, cache=cache, rows=rows, ordered_ids=ids,
            scale=ctx["scale"], lambda_a=ctx["lambda_A"], commit=commit_update))

    return _finalize(arm, out, checkpoint_dir, f=f, opt=opt, history=history,
                     bindings=bindings, parity=parity,
                     before_sha=initial_parameter_sha,
                     before_frozen=initial_frozen_sha, started=started,
                     completed_updates=sw146.UPDATES)


def validate_training(arm: str, *, output_root=TRAIN, work_root=WORK) -> dict:
    """Validate a completed SW0148 endpoint before prediction or scoring."""
    if arm not in sw146.ARMS:
        raise ValueError("unregistered SW0148 arm")
    out = Path(output_root) / f"{arm}_seed1"
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = out / "checkpoint.pt"
    optimizer = out / "optimizer.pt"
    history = out / "history.json"
    diagnostic = out / "first64_boundary_diagnostic.json"
    boundary = Path(work_root) / f"{arm}_seed1" / "checkpoints" / "boundary.ptx"
    src = sw146.source(sw146.SEED)
    ids, _rows, _ = sw146.ordered_train()
    current_ctx = sw146._admission_context()
    current_preflights = sw146._require_both_preflights(current_ctx)
    current_assets = sw146.validated_assets()
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("SW0148 training manifest is missing bound inputs")
    current_trainer_sha = _sha(HERE / "train.py")
    current_helper_sha = _sha(HERE / "checkpoint.py")
    current_protocol_sha = _sha(HERE / "protocol.json")
    expected_parent_sha = _sha(Path(sw146.__file__).resolve())
    if expected_parent_sha != EXPECTED_SW146_RUN_SHA256:
        raise ValueError("frozen SW0146 trainer changed")
    if (manifest.get("status") != "training_complete"
            or manifest.get("experiment") != NAME or manifest.get("arm") != arm
            or manifest.get("seed") != sw146.SEED
            or manifest.get("updates") != sw146.UPDATES
            or manifest.get("completed_updates") != sw146.UPDATES
            or manifest.get("logical_batch") != sw146.BATCH
            or manifest.get("microbatch") != sw146.MICRO
            or manifest.get("exposures") != sw146.UPDATES * sw146.BATCH
            or manifest.get("implementation_sha256") != current_trainer_sha
            or manifest.get("checkpoint_helper_sha256") != current_helper_sha
            or manifest.get("protocol_sha256") != current_protocol_sha
            or manifest.get("parent_run_sha256") != expected_parent_sha
            or manifest.get("parent_implementation_sha256") != sw146.implementation_sha()
            or manifest.get("checkpoint_sha256") != _sha(checkpoint)
            or manifest.get("optimizer_sha256") != _sha(optimizer)
            or manifest.get("history_sha256") != _sha(history)
            or manifest.get("first64_boundary_diagnostic_sha256") != _sha(diagnostic)
            or manifest.get("durable_boundary_sha256") != _sha(boundary)
            or manifest.get("optimizer_step_range") != [sw146.UPDATES, sw146.UPDATES]
            or manifest.get("ground_truth_used") is not False):
        raise ValueError("SW0148 completed training artifact/hash contract failed")
    if (manifest.get("source_core_sha256") != src[5]
            or manifest.get("source_manifest_sha256") != sw146.sha(src[1])
            or manifest.get("source_training_ids_sha256") != expected_ids_sha
            or manifest.get("ordered_train_ids_sha256") != expected_ids_sha
            or manifest.get("train_ids_sha256") != expected_ids_sha
            or manifest.get("train_ids") != ids):
        raise ValueError("SW0148 source/order lineage differs")
    if (manifest.get("rgb_asset_validation") != current_assets
            or manifest.get("preflight_sha256") != {
                name: sw146.sha(sw146.preflight_path(name)) for name in sw146.ARMS}
            or manifest.get("lambda_A") != current_ctx["lambda_A"]
            or manifest.get("scale") != current_ctx["scale"]
            or bindings.get("admission_sha256") != current_ctx["admission_sha256"]
            or bindings.get("source_checkpoint_sha256") != src[5]
            or bindings.get("source_manifest_sha256") != sw146.sha(src[1])
            or bindings.get("source_training_ids_sha256") != expected_ids_sha
            or bindings.get("ordered_train_ids_sha256") != expected_ids_sha
            or bindings.get("rgb_asset_validation") != current_assets
            or bindings.get("preflight_sha256") != manifest.get("preflight_sha256")
            or bindings.get("preflight_parameter_sha256") != {
                name: current_preflights[name]["post_parameter_sha256"]
                for name in sw146.ARMS}
            or bindings.get("parent_implementation_sha256") != sw146.implementation_sha()
            or bindings.get("parent_implementation_fingerprint") != sw146.fingerprint()
            or bindings.get("parent_protocol_sha256") != EXPECTED_SW146_PROTOCOL_SHA256
            or bindings.get("arm") != arm
            or bindings.get("seed") != sw146.SEED
            or bindings.get("updates") != sw146.UPDATES
            or bindings.get("batch") != sw146.BATCH
            or bindings.get("microbatch") != sw146.MICRO
            or bindings.get("total_steps") != sw146.STEPS
            or bindings.get("settle") != sw146.SETTLE
            or bindings.get("live_tail") != sw146.TAIL
            or bindings.get("lambda_A") != current_ctx["lambda_A"]
            or bindings.get("scale") != current_ctx["scale"]):
        raise ValueError("SW0148 training input/admission lineage differs")
    return {"manifest": manifest, "manifest_path": manifest_path,
            "manifest_sha256": _sha(manifest_path), "checkpoint": checkpoint,
            "optimizer": optimizer, "history": history,
            "boundary_checkpoint": boundary}
