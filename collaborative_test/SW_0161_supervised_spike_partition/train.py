"""Durable 512-update training of the SW161 actual-spike partition head."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np  # Keep NumPy before torch for the local Windows runtime.
import torch
from torch import nn

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0161_supervised_spike_partition import cache, objective

EXPERIMENT = "SW0161_supervised_spike_partition"
ARMS = ("pair_supervised_P", "mask_supervised_P")
SEED = 1
HEAD_SEED = 161001
SHUFFLE_SEED = 161002
COUNT, PASSES, UPDATES, BATCH, MICRO = 512, 16, 512, 16, 4
LR, CLIP = 3e-4, 1.0
WORK_ROOT = HERE / "results_archive" / "training_work_20261014"
TRAIN_ROOT = HERE / "results_archive" / "training_20261014"
CACHE_ROOT = cache.CACHE_ROOT
PREFLIGHT_PATH = HERE / "results_archive" / "preflight_20261014" / "preflight.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value.lower()))


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0161_supervised_spike_partition/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/objective.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/train.py",
        "collaborative_test/SW_0161_supervised_spike_partition/preflight.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/binder.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/score_queue.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/predict_evaluate.py",
    )
    paths = [ROOT / rel for rel in rels]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW161 training dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expected_cache_bindings():
    source = sw155.source_contract()
    assets = sw130.validate_rgb_assets()
    rgb = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.shape != (70000, 128, 128, 3):
        raise ValueError("registered 70k TRAIN RGB cache schema changed")
    ordered_rgb = np.asarray(rgb[source["rows"][:COUNT]]).copy()
    score_dependency = cache._current_score_dependency()
    bindings, _ = cache._source_bindings(source, ordered_rgb, assets, score_dependency)
    return bindings


def validate_cache(root=CACHE_ROOT, *, expected_bindings=None):
    """Require the complete, immutable, current-source 512-row cache."""
    expected = expected_bindings if expected_bindings is not None else _expected_cache_bindings()
    manifest = cache.validate_final_cache(root, expected_bindings=expected)
    if manifest["bindings"].get("cache_implementation_fingerprint") != cache.implementation_fingerprint():
        raise ValueError("source cache implementation lineage changed")
    return manifest


def make_training_order(count=COUNT, passes=PASSES, seed=SHUFFLE_SEED):
    if count != COUNT or passes != PASSES:
        raise ValueError("SW161 public training order is fixed to 512 rows and 16 passes")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.stack([torch.randperm(COUNT, generator=generator) for _ in range(PASSES)])
    return order.numpy().astype(np.int64, copy=False)


def make_optimizer(binder):
    return torch.optim.Adam([{"name": "binder", "params": list(binder.parameters()), "lr": LR}])


def _accumulate_microbatch_gradients(binder, arm, trace_microbatches, target_microbatches):
    if arm not in ARMS or len(trace_microbatches) != 4 or len(target_microbatches) != 4:
        raise ValueError("SW161 logical update has an invalid arm or microbatch count")
    losses = []
    for traces, targets in zip(trace_microbatches, target_microbatches):
        if traces.dtype != torch.float32 or tuple(traces.shape) != (MICRO, 4, 1024, 512):
            raise ValueError("cached spike microbatch must be float32 [4,4,1024,512]")
        if targets.dtype != torch.int64 or tuple(targets.shape) != (MICRO, 1024):
            raise ValueError("cached TRAIN targets must be int64 [4,1024]")
        probabilities, _, _ = binder(traces)
        if arm == "pair_supervised_P":
            loss, _ = objective.balanced_relation_loss(probabilities, targets)
        else:
            loss, _ = objective.mask_hungarian_loss(probabilities, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("SW161 supervised loss is nonfinite or nonscalar")
        (loss / 4.0).backward()
        losses.append(float(loss.detach().cpu()))
        del loss, probabilities
    parameters = list(binder.parameters())
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads):
        raise FloatingPointError("SW161 binder gradient is missing or nonfinite")
    grad_sq = sum(float(grad.detach().double().square().sum().cpu()) for grad in grads)
    grad_norm = math.sqrt(grad_sq)
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError("SW161 binder gradient norm is invalid")
    return losses, grad_norm


def logical_update(*, binder, optimizer, arm, trace_microbatches, target_microbatches,
                   update_index, pass_index, cache_indices, image_ids):
    """One four-microbatch B16 update, with exact class/instance loss dispatch."""
    optimizer.zero_grad(set_to_none=True)
    parameters = list(binder.parameters())
    losses, grad_norm = _accumulate_microbatch_gradients(
        binder, arm, trace_microbatches, target_microbatches)
    torch.nn.utils.clip_grad_norm_(parameters, CLIP)
    optimizer.step()
    return {"update": update_index, "pass": pass_index,
            "cache_indices": [int(v) for v in cache_indices],
            "image_ids": [int(v) for v in image_ids],
            "mean_loss": sum(losses) / 4.0,
            "binder_gradient_norm": grad_norm}


def load_microbatches(cache_store, cache_indices, device):
    """Read one registered logical B16 batch through the verified row loader."""
    if len(cache_indices) != BATCH or len(set(map(int, cache_indices))) != BATCH:
        raise ValueError("SW161 cache loader requires 16 distinct row indices")
    if any(type(index) not in (int, np.int64, np.int32) or not 0 <= int(index) < COUNT
           for index in cache_indices):
        raise ValueError("SW161 cache loader row index is outside the fixed512 prefix")
    traces_by_micro, targets_by_micro = [], []
    for start in range(0, BATCH, MICRO):
        rows, labels = [], []
        for cache_index in cache_indices[start:start + MICRO]:
            trace, target = cache_store.load_row(int(cache_index))
            rows.append(trace)
            labels.append(target)
        traces = torch.from_numpy(np.stack(rows)).to(device)
        targets = torch.from_numpy(np.stack(labels)).to(device)
        if traces.dtype != torch.float32 or tuple(traces.shape) != (MICRO, 4, 1024, 512):
            raise ValueError("verified cache row loader returned invalid trace batch")
        if targets.dtype != torch.int64 or tuple(targets.shape) != (MICRO, 1024):
            raise ValueError("verified cache row loader returned invalid modal32 targets")
        traces_by_micro.append(traces)
        targets_by_micro.append(targets)
    return traces_by_micro, targets_by_micro


def validate_preflight_admission(path=PREFLIGHT_PATH, *, cache_manifest=None,
                                 cache_root=CACHE_ROOT):
    """Require a current, strict two-update real-cache receipt before training."""
    from collaborative_test.SW_0161_supervised_spike_partition import preflight
    path = Path(path)
    if not path.is_file():
        raise ValueError("SW161 requires its completed actual-cache GPU preflight receipt")
    report = json.loads(path.read_text(encoding="utf-8"))
    current_cache = cache_manifest if cache_manifest is not None else validate_cache(cache_root)
    expected = {
        "train_implementation_fingerprint": implementation_fingerprint(),
        "preflight_implementation_fingerprint": preflight.implementation_fingerprint(),
        "cache_manifest_sha256": sha256_file(Path(cache_root) / "manifest.json"),
        "cache_ledger_sha256": sha256_file(Path(cache_root) / "ledger.json"),
        "cache_bindings_file_sha256": sha256_file(Path(cache_root) / "bindings.json"),
        "cache_bindings_sha256": current_cache["bindings_sha256"],
        "cache_implementation_fingerprint": cache.implementation_fingerprint(),
        "cache_count": current_cache["count"],
        "cache_indices": list(range(BATCH)),
        "image_ids": current_cache["bindings"]["image_ids"][:BATCH],
        "source_model_state_sha256": current_cache["source_model_state_before_sha256"],
    }
    preflight.validate_report_record(report, expected=expected)
    if report.get("training_admission") is not True:
        raise ValueError("SW161 disposable preflight did not admit full training")
    return {"report": report, "report_sha256": sha256_file(path),
            "preflight_implementation_fingerprint": expected["preflight_implementation_fingerprint"]}


def optimizer_step_summary(optimizer_state, *, expected_name="binder", max_step=UPDATES):
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if (not isinstance(groups, list) or len(groups) != 1 or groups[0].get("name") != expected_name
            or float(groups[0].get("lr", -1)) != LR
            or not isinstance(states, dict) or not states):
        raise ValueError("serialized optimizer group schema is invalid")
    param_ids = groups[0].get("params")
    expected_params = list(NativeSpikeSlotBinder(seed=HEAD_SEED).parameters())
    if (not isinstance(param_ids, list) or len(param_ids) != len(expected_params)
            or set(states) != set(param_ids)):
        raise ValueError("serialized Adam state does not cover the registered binder parameters")
    steps = []
    for param_id, expected_param in zip(param_ids, expected_params):
        value = states[param_id]
        if not isinstance(value, dict) or "step" not in value:
            raise ValueError("serialized optimizer state lacks step")
        raw = value["step"]
        step = int(raw.item()) if torch.is_tensor(raw) else int(raw)
        if not 1 <= step <= max_step:
            raise ValueError("serialized optimizer step is outside registered range")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = value.get(name)
            if (not torch.is_tensor(tensor) or tuple(tensor.shape) != tuple(expected_param.shape)
                    or not torch.isfinite(tensor).all()):
                raise ValueError("serialized Adam moments are missing or nonfinite")
        steps.append(step)
    summary = {expected_name: max(steps)}
    if summary[expected_name] != max_step or any(step != max_step for step in steps):
        raise ValueError("serialized optimizer did not reach the registered endpoint")
    return summary


def validate_training_artifact(arm, *, directory=None, cache_root=CACHE_ROOT):
    """Validate exact endpoint, source-cache lineage, history and saved Adam state."""
    if arm not in ARMS:
        raise ValueError("unregistered SW161 arm")
    output = Path(directory) if directory is not None else TRAIN_ROOT / f"{arm}_seed1"
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SW161 training manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_manifest = validate_cache(cache_root)
    preflight_binding = validate_preflight_admission(cache_root=cache_root,
                                                    cache_manifest=cache_manifest)
    fp = implementation_fingerprint()
    bindings = manifest.get("bindings")
    expected_ids = cache_manifest["bindings"]["image_ids"]
    expected_order = make_training_order().tolist()
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != EXPERIMENT
            or manifest.get("arm") != arm or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != UPDATES or manifest.get("passes") != PASSES
            or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
            or manifest.get("learning_rate") != LR or manifest.get("clip_norm") != CLIP
            or manifest.get("exposures") != PASSES * COUNT
            or manifest.get("ground_truth_used") is not True
            or manifest.get("ground_truth_scope") != "registered 512 TRAIN targets in validated SW161 source cache"
            or manifest.get("cache_manifest_sha256") != sha256_file(Path(cache_root) / "manifest.json")
            or manifest.get("cache_bindings_sha256") != cache_manifest["bindings_sha256"]
            or manifest.get("preflight_report_sha256") != preflight_binding["report_sha256"]
            or manifest.get("preflight_implementation_fingerprint") != preflight_binding["preflight_implementation_fingerprint"]
            or manifest.get("implementation_fingerprint") != fp
            or manifest.get("training_order") != expected_order
            or manifest.get("train_ids") != expected_ids
            or not isinstance(bindings, dict)
            or bindings.get("experiment") != EXPERIMENT or bindings.get("arm") != arm
            or bindings.get("seed") != SEED or bindings.get("head_seed") != HEAD_SEED
            or bindings.get("shuffle_seed") != SHUFFLE_SEED
            or bindings.get("passes") != PASSES or bindings.get("updates") != UPDATES
            or bindings.get("logical_batch") != BATCH or bindings.get("microbatch") != MICRO
            or bindings.get("learning_rate") != LR or bindings.get("clip_norm") != CLIP
            or bindings.get("ordered_train_ids_sha256") != sw155.ids_sha(expected_ids)
            or bindings.get("cache_manifest_sha256") != manifest.get("cache_manifest_sha256")
            or bindings.get("cache_bindings_sha256") != manifest.get("cache_bindings_sha256")
            or bindings.get("preflight_report_sha256") != manifest.get("preflight_report_sha256")
            or bindings.get("preflight_implementation_fingerprint") != manifest.get("preflight_implementation_fingerprint")
            or bindings.get("schedule_sha256") != hashlib.sha256(make_training_order().tobytes()).hexdigest()
            or bindings.get("training_order") != expected_order
            or bindings.get("implementation_fingerprint") != fp):
        raise ValueError("SW161 training artifact failed source/order/endpoint binding")
    if (not _valid_sha(bindings.get("initial_binder_state_sha256"))
            or manifest.get("initial_binder_state_sha256") != bindings.get("initial_binder_state_sha256")
            or not _valid_sha(manifest.get("final_binder_state_sha256"))
            or manifest.get("initial_binder_state_sha256") == manifest.get("final_binder_state_sha256")
            or manifest.get("binder_state_sha256") != manifest.get("final_binder_state_sha256")):
        raise ValueError("SW161 training binder initialization/endpoint hashes are invalid")
    for name, suffix in (("checkpoint", "pt"), ("optimizer", "pt"), ("history", "json")):
        path = output / f"{name}.{suffix}"
        if not path.is_file() or manifest.get(f"{name}_sha256") != sha256_file(path):
            raise ValueError(f"SW161 {name} artifact is missing or hash-mismatched")
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    order = make_training_order()
    ids = cache_manifest["bindings"]["image_ids"]
    if not isinstance(history, list) or len(history) != UPDATES:
        raise ValueError("SW161 history length is not 512")
    for ix, row in enumerate(history):
        indices = order[ix // 32, (ix % 32) * BATCH:(ix % 32 + 1) * BATCH].tolist()
        expected_ids = [ids[index] for index in indices]
        if (row.get("update") != ix + 1 or row.get("cache_indices") != indices
                or row.get("image_ids") != expected_ids
                or not isinstance(row.get("mean_loss"), (int, float))
                or not math.isfinite(row["mean_loss"])
                or not isinstance(row.get("binder_gradient_norm"), (int, float))
                or not math.isfinite(row["binder_gradient_norm"])
                or row["binder_gradient_norm"] <= 0):
            raise ValueError("SW161 history violates exact pass/update order")
    optimizer_state = torch.load(output / "optimizer.pt", map_location="cpu", weights_only=False)
    summary = optimizer_step_summary(optimizer_state)
    if manifest.get("optimizer_step_summary") != summary:
        raise ValueError("SW161 serialized optimizer step summary mismatch")
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("binder"), dict)
            or checkpoint.get("completed_updates") != UPDATES):
        raise ValueError("SW161 endpoint checkpoint schema is invalid")
    expected_state = NativeSpikeSlotBinder(seed=HEAD_SEED).state_dict()
    saved_state = checkpoint["binder"]
    if (set(saved_state) != set(expected_state)
            or any(not torch.is_tensor(saved_state[name])
                   or saved_state[name].shape != expected_state[name].shape
                   or saved_state[name].dtype != expected_state[name].dtype
                   or not torch.isfinite(saved_state[name]).all() for name in expected_state)):
        raise ValueError("SW161 endpoint binder state schema/nonfinite check failed")
    if (manifest.get("binder_state_sha256") != source_training.state_sha(saved_state)
            or checkpoint.get("binder_state_sha256") != manifest.get("binder_state_sha256")):
        raise ValueError("SW161 endpoint binder state hash mismatch")
    return manifest


def _fsync_dir(path):
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_training_manifest(*, arm, output_files, bindings, cache_manifest,
                            order, schedule_sha, step_summary,
                            initial_binder_state_sha256, binder_state_sha256):
    """Build the exact final manifest consumed by validate_training_artifact."""
    if arm not in ARMS or set(output_files) != {"checkpoint", "optimizer", "history"}:
        raise ValueError("invalid SW161 endpoint manifest inputs")
    cache_manifest_sha = bindings["cache_manifest_sha256"]
    cache_bindings_sha = bindings["cache_bindings_sha256"]
    preflight_sha = bindings["preflight_report_sha256"]
    preflight_fp = bindings["preflight_implementation_fingerprint"]
    return {"status": "training_complete", "experiment": EXPERIMENT,
        "arm": arm, "seed": SEED, "completed_updates": UPDATES,
        "passes": PASSES, "logical_batch": BATCH, "microbatch": MICRO,
        "learning_rate": LR, "clip_norm": CLIP, "exposures": PASSES * COUNT,
        "train_ids": cache_manifest["bindings"]["image_ids"], "bindings": bindings,
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_bindings_sha256": cache_bindings_sha,
        "preflight_report_sha256": preflight_sha,
        "preflight_implementation_fingerprint": preflight_fp,
        "training_order": order.tolist(), "schedule_sha256": schedule_sha,
        "optimizer_step_summary": step_summary,
        "initial_binder_state_sha256": initial_binder_state_sha256,
        "final_binder_state_sha256": binder_state_sha256,
        "binder_state_sha256": binder_state_sha256,
        "checkpoint_sha256": sha256_file(output_files["checkpoint"]),
        "optimizer_sha256": sha256_file(output_files["optimizer"]),
        "history_sha256": sha256_file(output_files["history"]),
        "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_used": True,
        "ground_truth_scope": "registered 512 TRAIN targets in validated SW161 source cache"}


def train_arm(arm, *, device="cuda:0", cache_root=CACHE_ROOT,
              work_root=WORK_ROOT, train_root=TRAIN_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW161 arm")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW161 full training requires its assigned CUDA device")
    torch.cuda.set_device(dev)
    cache_manifest = validate_cache(cache_root)
    preflight_binding = validate_preflight_admission(cache_root=cache_root,
                                                    cache_manifest=cache_manifest)
    ids = cache_manifest["bindings"]["image_ids"]
    cache_store = cache._AtomicTraceCache(cache_root, cache_manifest["bindings"], ids,
                                          cache_manifest["bindings"]["image_rgb_sha256"])
    order = make_training_order()
    schedule_sha = hashlib.sha256(order.tobytes()).hexdigest()
    bindings = {"experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "head_seed": HEAD_SEED, "shuffle_seed": SHUFFLE_SEED,
        "passes": PASSES, "updates": UPDATES, "logical_batch": BATCH, "microbatch": MICRO,
        "learning_rate": LR, "clip_norm": CLIP,
        "ordered_train_ids_sha256": sw155.ids_sha(ids),
        "cache_manifest_sha256": sha256_file(Path(cache_root) / "manifest.json"),
        "cache_bindings_sha256": cache_manifest["bindings_sha256"],
        "cache_implementation_fingerprint": cache.implementation_fingerprint(),
        "preflight_report_sha256": preflight_binding["report_sha256"],
        "preflight_implementation_fingerprint": preflight_binding["preflight_implementation_fingerprint"],
        "schedule_sha256": schedule_sha, "training_order": order.tolist(),
        "implementation_fingerprint": implementation_fingerprint()}
    output = Path(train_root) / f"{arm}_seed1"
    if output.exists():
        raise FileExistsError(f"preserving existing SW161 training result: {output}")
    work = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    binder = NativeSpikeSlotBinder(seed=HEAD_SEED).to(dev)
    binder.train()
    initial_binder_sha = source_training.state_sha(binder)
    bindings["initial_binder_state_sha256"] = initial_binder_sha
    encoder_sentinel = nn.Identity().to(dev)
    optimizer = make_optimizer(binder)
    runtime = durable.runtime_signature(dev)
    history = []
    if (work / "boundary.ptx").is_file():
        loaded = durable.load_latest(work, wrapped=binder, encoder=encoder_sentinel,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and not durable.is_only_uncommitted_temp(work):
            raise FileExistsError("preserving incomplete SW161 training work")
        work.mkdir(parents=True, exist_ok=True)
        completed = 0
        durable.save_boundary(work, completed_updates=0, wrapped=binder,
            encoder=encoder_sentinel, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=dev)

    for update in range(completed, UPDATES):
        pass_ix, in_pass = divmod(update, 32)
        cache_indices = order[pass_ix, in_pass * BATCH:(in_pass + 1) * BATCH].tolist()
        trace_microbatches, target_microbatches = load_microbatches(
            cache_store, cache_indices, dev)
        row = logical_update(binder=binder, optimizer=optimizer, arm=arm,
            trace_microbatches=trace_microbatches, target_microbatches=target_microbatches,
            update_index=update + 1, pass_index=pass_ix + 1,
            cache_indices=cache_indices, image_ids=[ids[index] for index in cache_indices])
        history.append(row)
        durable.save_boundary(work, completed_updates=update + 1,
            wrapped=binder, encoder=encoder_sentinel, optimizer=optimizer,
            history=history, bindings=bindings, runtime=runtime, device=dev)
        if (update + 1) % 16 == 0:
            print(json.dumps({"arm": arm, "completed_updates": update + 1,
                              "total_updates": UPDATES}, allow_nan=False), flush=True)

    optimizer_state = optimizer.state_dict()
    step_summary = optimizer_step_summary(optimizer_state)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    checkpoint_path, optimizer_path, history_path = (
        staging / "checkpoint.pt", staging / "optimizer.pt", staging / "history.json")
    binder_state = {name: value.detach().cpu().clone() for name, value in binder.state_dict().items()}
    binder_state_sha = source_training.state_sha(binder_state)
    with checkpoint_path.open("xb") as stream:
        torch.save({"binder": binder_state, "binder_state_sha256": binder_state_sha,
                    "completed_updates": UPDATES}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with optimizer_path.open("xb") as stream:
        torch.save(optimizer_state, stream); stream.flush(); os.fsync(stream.fileno())
    with history_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    manifest = build_training_manifest(arm=arm,
        output_files={"checkpoint": checkpoint_path, "optimizer": optimizer_path,
                      "history": history_path},
        bindings=bindings, cache_manifest=cache_manifest, order=order,
        schedule_sha=schedule_sha, step_summary=step_summary,
        initial_binder_state_sha256=initial_binder_sha,
        binder_state_sha256=binder_state_sha)
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    _fsync_dir(staging)
    os.rename(staging, output)
    _fsync_dir(output.parent)
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = train_arm(args.arm, device=args.device)
    print(json.dumps({"status": result["status"], "arm": args.arm,
                      "completed_updates": UPDATES, "training_admission": True}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
