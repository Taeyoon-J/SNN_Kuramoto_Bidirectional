"""Durable four-arm, 512-update SW0163 relation-context training."""
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

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0161_supervised_spike_partition import objective, train as source_train
from collaborative_test.SW_0163_relation_context_assignment import context_cache, model, preflight

EXPERIMENT = "SW0163_relation_context_assignment"
ARMS = model.ARMS
SEED, HEAD_SEED, MESSAGE_SEED, SHUFFLE_SEED = 1, 161001, 163001, 161002
COUNT, PASSES, UPDATES, BATCH, MICRO = 512, 16, 512, 16, 4
LR, CLIP = 3e-4, 1.0
CACHE_ROOT = source_cache.CACHE_ROOT
CONTEXT_ROOT = context_cache.CACHE_ROOT
PREFLIGHT_PATH = HERE / "results_archive" / "preflight_20261014" / "head_seed1.json"
WORK_ROOT = HERE / "results_archive" / "training_work_20261014"
TRAIN_ROOT = HERE / "results_archive" / "training_20261014"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _model_state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0163_relation_context_assignment/train.py",
        "collaborative_test/SW_0163_relation_context_assignment/model.py",
        "collaborative_test/SW_0163_relation_context_assignment/context_cache.py",
        "collaborative_test/SW_0163_relation_context_assignment/preflight.py",
        "collaborative_test/SW_0163_relation_context_assignment/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/objective.py",
        "collaborative_test/SW_0161_supervised_spike_partition/train.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
    )
    paths = [ROOT / rel for rel in rels]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0163 training dependency missing: {missing[0]}")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def implementation_sha256():
    payload = json.dumps(implementation_fingerprint(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def make_training_order():
    """The registered SW161 16-pass shuffle, shared exactly by all arms."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SHUFFLE_SEED)
    return torch.stack([torch.randperm(COUNT, generator=generator)
                        for _ in range(PASSES)]).numpy().astype(np.int64, copy=False)


def prepare_training_inputs_cpu(*, cache_root=CACHE_ROOT, context_root=CONTEXT_ROOT,
                               preflight_path=PREFLIGHT_PATH):
    """Deeply validate frozen inputs and the real admitted preflight before CUDA use."""
    upstream = preflight.validate_upstream(context_root)
    source_manifest_path = Path(cache_root) / "manifest.json"
    context_manifest_path = Path(context_root) / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(source_manifest_path)
    context_manifest_sha = sha256_file(context_manifest_path)
    if (source_manifest.get("status") != "complete" or source_manifest.get("count") != COUNT
            or context_manifest.get("status") != "complete" or context_manifest.get("count") != COUNT
            or source_manifest_sha != upstream.get("source_trace_manifest_sha256")
            or context_manifest_sha != upstream.get("context_cache_manifest_sha256")):
        raise ValueError("SW0163 validated cache manifests changed during CPU admission")
    report_path = Path(preflight_path)
    if not report_path.is_file():
        raise ValueError("SW0163 full training requires a completed preflight receipt")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    preflight.validate_report_record(report, expected=upstream)
    if report.get("training_admission") is not True:
        raise ValueError("SW0163 two-update preflight rejected full training")
    if source_manifest.get("count") != COUNT or context_manifest.get("count") != COUNT:
        raise ValueError("SW0163 requires both complete 512-row caches")
    return {
        "source_manifest": source_manifest,
        "context_manifest": context_manifest,
        "source_manifest_sha256": source_manifest_sha,
        "context_manifest_sha256": context_manifest_sha,
        "source_bindings_sha256": source_manifest["bindings_sha256"],
        "context_bindings_sha256": context_manifest["bindings_sha256"],
        "preflight_report": report,
        "preflight_report_sha256": sha256_file(report_path),
        "preflight_upstream": upstream,
        "preflight_implementation_fingerprint": preflight.implementation_fingerprint(),
        "implementation_fingerprint": implementation_fingerprint(),
        "image_ids": source_manifest["bindings"]["image_ids"],
    }


def _training_bindings(arm, prepared, order):
    initial = model.RelationContextAssignment(arm, seed=MESSAGE_SEED,
                                               binder_seed=HEAD_SEED).state_dict()
    return {
        "experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "head_seed": HEAD_SEED, "message_seed": MESSAGE_SEED,
        "shuffle_seed": SHUFFLE_SEED, "passes": PASSES, "updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO, "learning_rate": LR,
        "clip_norm": CLIP, "ordered_train_ids_sha256": sw155.ids_sha(prepared["image_ids"]),
        "source_trace_manifest_sha256": prepared["source_manifest_sha256"],
        "source_trace_bindings_sha256": prepared["source_bindings_sha256"],
        "context_cache_manifest_sha256": prepared["context_manifest_sha256"],
        "context_cache_bindings_sha256": prepared["context_bindings_sha256"],
        "preflight_report_sha256": prepared["preflight_report_sha256"],
        "preflight_implementation_fingerprint": prepared["preflight_implementation_fingerprint"],
        "source_checkpoint_sha256": prepared["source_manifest"]["bindings"]["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest"]["bindings"]["source_manifest_sha256"],
        "source_model_state_sha256": prepared["source_manifest"]["source_model_state_before_sha256"],
        "initial_model_state_sha256": _model_state_sha(initial),
        "schedule_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "training_order": order.tolist(),
        "implementation_fingerprint": prepared["implementation_fingerprint"],
    }


def _load_logical_batch(source_store, context_store, indices, device):
    if len(indices) != BATCH or len(set(map(int, indices))) != BATCH:
        raise ValueError("SW0163 loader requires 16 distinct registered cache rows")
    traces, targets, q_rows, a_rows = [], [], [], []
    image_ids = source_store.image_ids
    for index in map(int, indices):
        trace, target = source_store.load_row(index)
        q, a = context_store.load(index)
        traces.append(torch.from_numpy(np.asarray(trace).copy()))
        targets.append(torch.from_numpy(np.asarray(target).copy()))
        q_rows.append(torch.from_numpy(np.asarray(q).copy()))
        a_rows.append(torch.from_numpy(np.asarray(a).copy()))
    outputs = []
    for start in range(0, BATCH, MICRO):
        outputs.append((torch.stack(traces[start:start+MICRO]).to(device),
                        torch.stack(targets[start:start+MICRO]).to(device),
                        torch.stack(q_rows[start:start+MICRO]).to(device),
                        torch.stack(a_rows[start:start+MICRO]).to(device)))
    ids = [int(image_ids[int(i)]) for i in indices]
    return outputs, ids


def _logical_update(*, binder, optimizer, microbatches, update_index, pass_index,
                    cache_indices, image_ids):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    row_error = 0.0
    for spikes, targets, q, adjacency in microbatches:
        p, _, _ = binder(spikes, q, adjacency)
        loss, _ = objective.mask_hungarian_loss(p, targets)
        if loss.ndim != 0 or not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError("SW0163 TRAIN mask loss is nonfinite")
        (loss / len(microbatches)).backward()
        losses.append(float(loss.detach().cpu()))
        row_error = max(row_error, float((p.sum(-1) - 1).abs().max().detach().cpu()))
        if not bool(torch.isfinite(p).all().detach().cpu()):
            raise FloatingPointError("SW0163 assignment probabilities are nonfinite")
        del p, loss
    params = list(binder.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or any(not bool(torch.isfinite(g).all().detach().cpu()) for g in grads):
        raise FloatingPointError("SW0163 optimizer gradients are absent or nonfinite")
    norm = math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in grads))
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("SW0163 gradient norm must be finite and positive")
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    optimizer.step()
    return {"update": update_index, "pass": pass_index,
            "cache_indices": [int(i) for i in cache_indices], "image_ids": image_ids,
            "mean_loss": float(np.mean(losses)), "gradient_norm": norm,
            "row_sum_error": row_error}


def _optimizer_summary(optimizer_state, expected_parameters, *, endpoint=UPDATES):
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if (not isinstance(groups, list) or len(groups) != 1 or groups[0].get("name") != "assignment"
            or float(groups[0].get("lr", -1)) != LR or not isinstance(states, dict)
            or len(states) != len(expected_parameters)):
        raise ValueError("SW0163 serialized optimizer group/state schema is invalid")
    params = groups[0].get("params", [])
    if len(params) != len(expected_parameters) or set(params) != set(states):
        raise ValueError("SW0163 Adam states do not match all assignment parameters")
    steps = []
    for key, expected_param in zip(params, expected_parameters):
        state = states[key]
        step = int(state["step"].item()) if torch.is_tensor(state.get("step")) else int(state.get("step", 0))
        if step != endpoint:
            raise ValueError("SW0163 assignment Adam did not reach the exact update endpoint")
        if (not torch.is_tensor(state.get("exp_avg")) or not torch.is_tensor(state.get("exp_avg_sq"))
                or state["exp_avg"].shape != expected_param.shape
                or state["exp_avg_sq"].shape != expected_param.shape
                or not torch.isfinite(state["exp_avg"]).all()
                or not torch.isfinite(state["exp_avg_sq"]).all()):
            raise ValueError("SW0163 Adam moment tensors are invalid")
        steps.append(step)
    return {"assignment": max(steps), "parameter_state_count": len(steps)}


def validate_training_artifact(arm, *, directory=None, prepared=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0163 training arm")
    prepared = prepared or prepare_training_inputs_cpu()
    output = Path(directory) if directory else TRAIN_ROOT / f"{arm}_seed1"
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SW0163 training artifact is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = make_training_order()
    expected_bindings = _training_bindings(arm, prepared, order)
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != EXPERIMENT
            or manifest.get("arm") != arm or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != UPDATES or manifest.get("passes") != PASSES
            or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
            or manifest.get("learning_rate") != LR or manifest.get("clip_norm") != CLIP
            or manifest.get("implementation_fingerprint") != implementation_fingerprint()
            or manifest.get("exposures") != COUNT * PASSES
            or manifest.get("ground_truth_used") is not True
            or manifest.get("ground_truth_scope") != "registered 512 TRAIN modal32 targets"
            or manifest.get("training_order") != order.tolist()
            or manifest.get("bindings") != expected_bindings):
        raise ValueError("SW0163 training manifest/source/order binding is invalid")
    paths = {name: output / fname for name, fname in (
        ("checkpoint", "checkpoint.pt"), ("optimizer", "optimizer.pt"), ("history", "history.json"))}
    for name, path in paths.items():
        if not path.is_file() or manifest.get(f"{name}_sha256") != sha256_file(path):
            raise ValueError(f"SW0163 {name} is absent or hash-mismatched")
    history = json.loads(paths["history"].read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != UPDATES:
        raise ValueError("SW0163 history must contain exactly512 logical updates")
    ids = prepared["image_ids"]
    for ix, row in enumerate(history):
        p_ix, b_ix = divmod(ix, 32)
        indices = order[p_ix, b_ix * BATCH:(b_ix + 1) * BATCH].tolist()
        if (row.get("update") != ix + 1 or row.get("pass") != p_ix + 1
                or row.get("cache_indices") != indices
                or row.get("image_ids") != [int(ids[i]) for i in indices]
                or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                       for k in ("mean_loss", "gradient_norm", "row_sum_error"))
                or row["gradient_norm"] <= 0 or row["row_sum_error"] > 2e-5):
            raise ValueError("SW0163 update history violates registered exposure order")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    optimizer_state = torch.load(paths["optimizer"], map_location="cpu", weights_only=False)
    expected_model = model.RelationContextAssignment(arm).state_dict()
    state = checkpoint.get("assignment") if isinstance(checkpoint, dict) else None
    if (not isinstance(state, dict) or set(state) != set(expected_model)
            or any(not torch.is_tensor(state[k]) or state[k].shape != expected_model[k].shape
                   or state[k].dtype != expected_model[k].dtype or not torch.isfinite(state[k]).all()
                   for k in expected_model)
            or checkpoint.get("completed_updates") != UPDATES):
        raise ValueError("SW0163 final assignment checkpoint is invalid")
    state_hash = _model_state_sha(state)
    if (state_hash != manifest.get("final_model_state_sha256")
            or checkpoint.get("model_state_sha256") != state_hash
            or manifest.get("initial_model_state_sha256") == state_hash):
        raise ValueError("SW0163 model checkpoint digest/learning endpoint mismatch")
    summary = _optimizer_summary(optimizer_state,
                                list(model.RelationContextAssignment(arm).parameters()))
    if manifest.get("optimizer_step_summary") != summary:
        raise ValueError("SW0163 serialized Adam endpoint differs from manifest")
    return manifest


def train_arm(arm, *, device="cuda:0", cache_root=CACHE_ROOT,
              context_root=CONTEXT_ROOT, preflight_path=PREFLIGHT_PATH,
              work_root=WORK_ROOT, train_root=TRAIN_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0163 training arm")
    # Validate the immutable source/Q/A/preflight lineage before any CUDA query
    # or model allocation; CPU supervisor checks share this same public path.
    prepared = prepare_training_inputs_cpu(cache_root=cache_root,
        context_root=context_root, preflight_path=preflight_path)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0163 full training requires the assigned CUDA device")
    torch.cuda.set_device(dev)
    order = make_training_order()
    bindings = _training_bindings(arm, prepared, order)
    ids = prepared["image_ids"]
    source_store = source_cache._AtomicTraceCache(cache_root,
        prepared["source_manifest"]["bindings"], ids,
        prepared["source_manifest"]["bindings"]["image_rgb_sha256"])
    context_store = context_cache.AtomicContextCache(context_root,
        prepared["context_manifest"]["bindings"])
    output = Path(train_root) / f"{arm}_seed1"
    if output.exists():
        raise FileExistsError(f"preserving existing SW0163 training output: {output}")
    work = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    binder = model.RelationContextAssignment(arm, seed=MESSAGE_SEED,
                                              binder_seed=HEAD_SEED).to(dev)
    binder.train()
    initial_sha = _model_state_sha(binder.state_dict())
    if bindings["initial_model_state_sha256"] != initial_sha:
        raise AssertionError("SW0163 deterministic initial model hash differs from its binding")
    sentinel = nn.Identity().to(dev)
    optimizer = torch.optim.Adam([{"name": "assignment", "params": list(binder.parameters()), "lr": LR}])
    runtime = durable.runtime_signature(dev)
    history = []
    if (work / "boundary.ptx").is_file():
        loaded = durable.load_latest(work, wrapped=binder, encoder=sentinel,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and not durable.is_only_uncommitted_temp(work):
            raise FileExistsError("preserving incomplete uncommitted SW0163 training work")
        work.mkdir(parents=True, exist_ok=True)
        completed = 0
        durable.save_boundary(work, completed_updates=0, wrapped=binder, encoder=sentinel,
            optimizer=optimizer, history=history, bindings=bindings, runtime=runtime, device=dev)
    if len(history) != completed:
        raise ValueError("SW0163 durable cursor/history mismatch")
    for update in range(completed, UPDATES):
        pass_ix, in_pass = divmod(update, 32)
        indices = order[pass_ix, in_pass * BATCH:(in_pass + 1) * BATCH].tolist()
        micros, image_ids = _load_logical_batch(source_store, context_store, indices, dev)
        row = _logical_update(binder=binder, optimizer=optimizer, microbatches=micros,
            update_index=update + 1, pass_index=pass_ix + 1,
            cache_indices=indices, image_ids=image_ids)
        history.append(row)
        durable.save_boundary(work, completed_updates=update + 1, wrapped=binder,
            encoder=sentinel, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=dev)
        if (update + 1) % 16 == 0:
            print(json.dumps({"arm": arm, "completed_updates": update + 1,
                              "total_updates": UPDATES}, allow_nan=False), flush=True)
        del micros
    final_state = {k: v.detach().cpu().clone() for k, v in binder.state_dict().items()}
    final_sha = _model_state_sha(final_state)
    optimizer_state = optimizer.state_dict()
    summary = _optimizer_summary(optimizer_state, list(binder.parameters()))
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    files = {"checkpoint": staging / "checkpoint.pt", "optimizer": staging / "optimizer.pt",
             "history": staging / "history.json"}
    with files["checkpoint"].open("xb") as stream:
        torch.save({"assignment": final_state, "model_state_sha256": final_sha,
                    "completed_updates": UPDATES}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with files["optimizer"].open("xb") as stream:
        torch.save(optimizer_state, stream); stream.flush(); os.fsync(stream.fileno())
    with files["history"].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    manifest = {"status": "training_complete", "experiment": EXPERIMENT,
        "arm": arm, "seed": SEED, "completed_updates": UPDATES,
        "passes": PASSES, "logical_batch": BATCH, "microbatch": MICRO,
        "learning_rate": LR, "clip_norm": CLIP, "exposures": COUNT * PASSES,
        "train_ids": ids, "training_order": order.tolist(), "bindings": bindings,
        "initial_model_state_sha256": initial_sha, "final_model_state_sha256": final_sha,
        "optimizer_step_summary": summary,
        "checkpoint_sha256": sha256_file(files["checkpoint"]),
        "optimizer_sha256": sha256_file(files["optimizer"]),
        "history_sha256": sha256_file(files["history"]),
        "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_used": True, "ground_truth_scope": "registered 512 TRAIN modal32 targets"}
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    if output.exists():
        raise FileExistsError(f"preserving concurrent SW0163 output: {output}")
    if os.name != "nt":
        fd = os.open(staging, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    os.rename(staging, output)
    return validate_training_artifact(arm, directory=output, prepared=prepared)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    manifest = train_arm(args.arm, device=args.device)
    print(json.dumps({"status": manifest["status"], "arm": args.arm,
                      "completed_updates": UPDATES}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
