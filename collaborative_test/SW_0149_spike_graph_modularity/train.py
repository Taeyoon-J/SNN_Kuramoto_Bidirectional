"""Fixed 256-update SW0149 degree-corrected modularity pilot trainer."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0149_spike_graph_modularity.modularity import (
    degree_corrected_modularity_loss, source_adjacency,
)

from collaborative_test.SW_0149_spike_graph_modularity import preflight

NAME = preflight.NAME
SEED = 1
ARMS = ("encoder_graph", "joint")
BATCH, MICRO, UPDATES = 16, 4, 256
STEPS, SETTLE, TAIL = 1024, 512, 64
PREFLIGHT = HERE / "results_archive" / "preflight_seed1.json"
WORK = HERE / "work"
TRAIN = HERE / "training"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    result = dict(preflight.fingerprint())
    for path in (HERE / "train.py",
                 ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py"):
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def implementation_sha():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_source():
    checkpoint, manifest_path, manifest, rows, ids, core_sha = sw130.source_contract(SEED)
    ids = [int(value) for value in ids]
    rows = np.asarray(rows, dtype=np.int64)
    if len(ids) != BATCH * UPDATES or len(rows) != BATCH * UPDATES:
        raise ValueError("registered seed1 TRAIN order must contain exactly 4096 images")
    if len(set(ids)) != len(ids) or len(np.unique(rows)) != len(rows):
        raise ValueError("registered source TRAIN IDs/rows are not unique")
    return {"checkpoint": Path(checkpoint), "manifest_path": Path(manifest_path),
            "manifest": manifest, "rows": rows, "ids": ids, "core_sha256": core_sha}


def validate_preflight_admission(path=PREFLIGHT):
    """Validate both disposable arms and their source, parity, and credit evidence."""
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    src = _read_source()
    assets = sw130.validate_rgb_assets()
    expected_ids_sha = hashlib.sha256(np.asarray(src["ids"], dtype="<i8").tobytes()).hexdigest()
    if (report.get("status") != "preflight_complete" or report.get("experiment") != NAME
            or report.get("seed") != SEED or report.get("training_admission") is not False
            or report.get("ground_truth_used") is not False
            or report.get("implementation_fingerprint") != preflight.fingerprint()
            or report.get("implementation_sha256") != preflight.implementation_sha()
            or report.get("disposable_optimizer_updates_per_arm") != preflight.UPDATES):
        raise ValueError("SW0149 source preflight is incomplete or has changed implementation bytes")
    source = report.get("source", {})
    if (source.get("core_sha256") != src["core_sha256"]
            or source.get("manifest_sha256") != sha(src["manifest_path"])
            or source.get("training_ids") != src["ids"]
            or source.get("training_ids_sha256") != expected_ids_sha
            or source.get("rgb_asset_validation") != assets):
        raise ValueError("SW0149 preflight source, ordered IDs, or RGB assets changed")
    parity = report.get("existing_native32_parity", {})
    if (parity.get("sha256") != sha(preflight.PARITY_PATH)
            or len(parity.get("exact_arm_horizons", [])) != 4
            or any(row[1] is not True for row in parity["exact_arm_horizons"])):
        raise ValueError("SW0149 preflight lacks its exact frozen SW0135 parity receipt")
    credit = report.get("first64_modularity_credit", {})
    if (credit.get("images") != 64 or credit.get("microbatch") != MICRO
            or credit.get("positive_M_offdiagonal_entries", 0) <= 0
            or credit.get("negative_M_offdiagonal_entries", 0) <= 0
            or len(credit.get("per_b4_batch", [])) != 16):
        raise ValueError("SW0149 source graph lacks registered first64 positive/negative modularity credit")
    arm_rows = {row.get("arm"): row for row in report.get("arms", [])}
    if set(arm_rows) != set(ARMS):
        raise ValueError("SW0149 preflight must contain both registered arms")
    for arm, row in arm_rows.items():
        expected_groups = {"encoder", "graph"}
        if arm == "joint":
            expected_groups |= {"native_core", "integration_scalars"}
        if (row.get("trainable_groups") != sorted(expected_groups)
                or row.get("fresh_source_parity", {}).get("source_graph_exact") is not True
                or row.get("fresh_source_parity", {}).get("actual_spike_q_exact") is not True):
            raise ValueError(f"SW0149 {arm} preflight did not bind exact initialization/parity")
        updates = row.get("updates", [])
        if len(updates) != preflight.UPDATES:
            raise ValueError(f"SW0149 {arm} preflight update count differs")
        changed = set()
        positive_gradients = set()
        for number, update in enumerate(updates, 1):
            if update.get("update") != number:
                raise ValueError(f"SW0149 {arm} preflight history order differs")
            norms = update.get("gradient_norms", {})
            for group in expected_groups:
                value = norms.get(group)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"SW0149 {arm} preflight has invalid {group} gradient norm")
                if value > 0:
                    positive_gradients.add(group)
            changed.update(update.get("changed_groups", []))
        if not expected_groups.issubset(positive_gradients):
            raise ValueError(f"SW0149 {arm} preflight lacks positive gradient for a selected parameter family")
        required_changes = {"encoder", "graph"}
        if arm == "joint":
            required_changes.add("native_core")
        if not required_changes.issubset(changed):
            raise ValueError(f"SW0149 {arm} preflight did not change each required parameter family")
        row["preflight_sha256"] = hashlib.sha256(raw).hexdigest()
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
            "report": report, "arms": arm_rows, "source": src, "assets": assets}


def _groups(foundation):
    core = foundation.wrapped.core
    groups = {
        "encoder": list(foundation.encoder.parameters()),
        "graph": list(core.graph_generator.parameters()),
        "native_core": [p for name, p in core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m,
                                 foundation.wrapped.b],
    }
    flat = [p for params in groups.values() for p in params]
    if not flat or len({id(p) for p in flat}) != len(flat):
        raise ValueError("SW0149 optimizer families are empty or overlap")
    return groups


def _parameter_sha(parameters):
    digest = hashlib.sha256()
    for param in parameters:
        tensor = param.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _group_sha(groups):
    return {name: _parameter_sha(params) for name, params in groups.items()}


def _set_trainability(foundation, groups, arm):
    selected = {id(param) for name in ("encoder", "graph") for param in groups[name]}
    if arm == "joint":
        selected.update(id(param) for name in ("native_core", "integration_scalars")
                        for param in groups[name])
    for param in foundation.wrapped.parameters():
        param.requires_grad_(id(param) in selected)
    for param in foundation.encoder.parameters():
        param.requires_grad_(id(param) in selected)
    return selected


def _teacher(foundation):
    teacher_encoder = __import__("copy").deepcopy(foundation.encoder).eval()
    teacher_patcher = __import__("copy").deepcopy(foundation.patcher).eval()
    teacher_graph = __import__("copy").deepcopy(foundation.wrapped.core.graph_generator).eval()
    for module in (teacher_encoder, teacher_patcher, teacher_graph):
        module.requires_grad_(False)
    return teacher_encoder, teacher_patcher, teacher_graph


def _teacher_adjacency(teacher, foundation, rgb):
    encoder, patcher, graph = teacher
    with torch.no_grad():
        gamma = sw135.encode_native32(encoder, patcher, foundation.feature_mean,
                                      foundation.feature_std, foundation.feature_clip, rgb)
        return source_adjacency(graph(gamma))


def _student_q(foundation, rgb):
    gamma = foundation.encode(rgb)
    trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS,
                         live_tail_steps=TAIL)
    return sw145._actual_q(trace), trace


def _group_norms(groups):
    result = {}
    for name, params in groups.items():
        entries = [p.grad.detach().double().square().sum()
                   for p in params if p.grad is not None]
        value = math.sqrt(float(torch.stack(entries).sum().cpu())) if entries else 0.0
        if not math.isfinite(value):
            raise FloatingPointError(f"nonfinite SW0149 {name} gradient")
        result[name] = value
    return result


def logical_update(arm, update, *, foundation, teacher, optimizer, groups,
                   params, cache, ordered_rows, ordered_ids, commit=None):
    """One registered logical update; callback occurs after Adam and projection."""
    if arm not in ARMS or not 0 <= update < UPDATES:
        raise ValueError("invalid SW0149 arm/update index")
    optimizer.zero_grad(set_to_none=True)
    micro_rows = []
    for micro in range(BATCH // MICRO):
        start = update * BATCH + micro * MICRO
        source_rows = ordered_rows[start:start + MICRO]
        rgb_np = np.asarray(cache[source_rows]).copy()
        if rgb_np.dtype != np.uint8 or rgb_np.shape != (MICRO, 128, 128, 3):
            raise ValueError("registered SW0149 clean RGB cache returned the wrong native128 batch")
        rgb = torch.from_numpy(rgb_np).to(next(foundation.encoder.parameters()).device)
        adjacency = _teacher_adjacency(teacher, foundation, rgb)
        q, trace = _student_q(foundation, rgb)
        loss = degree_corrected_modularity_loss(adjacency, q)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SW0149 modularity loss")
        (loss / (BATCH // MICRO)).backward()
        micro_rows.append({"modularity_loss": float(loss.detach().cpu()),
                           "mean_source_degree": float(adjacency.sum((-2, -1)).mean().cpu())})
        del loss, adjacency, q, trace, rgb, rgb_np
    norms = _group_norms(groups)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    if arm == "joint":
        with torch.no_grad():
            foundation.wrapped.a_d.clamp_(0, 1)
            foundation.wrapped.a_m.clamp_(0, 1)
    if any(not torch.isfinite(param).all() for family in groups.values() for param in family):
        raise FloatingPointError("nonfinite SW0149 parameter after optimizer update")
    row = {"update": update + 1,
           "image_ids": ordered_ids[update * BATCH:(update + 1) * BATCH],
           "microbatches": micro_rows, "gradient_norms": norms,
           "parameter_sha256": _group_sha(groups)}
    if commit is not None:
        commit(row)
    return row


def _bindings(arm, admission, foundation, initial_hashes, runtime):
    parity = admission["arms"][arm]["fresh_source_parity"]
    return {"experiment": NAME, "arm": arm, "seed": SEED,
            "preflight_sha256": admission["sha256"],
            "preflight_implementation_sha256": preflight.implementation_sha(),
            "source_core_sha256": foundation.provenance["source_core_sha256"],
            "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
            "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
            "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
            "ordered_train_ids_sha256": hashlib.sha256(
                np.asarray(admission["source"]["ids"], dtype="<i8").tobytes()).hexdigest(),
            "initial_parameter_sha256": initial_hashes,
            "initial_fresh_source_parity": parity,
            "batch": BATCH, "microbatch": MICRO, "updates": UPDATES,
            "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
            "optimizer_groups": sorted(admission["arms"][arm]["trainable_groups"]),
            "optimizer": {"encoder": 3e-6, "graph": 3e-5,
                          "native_core": 3e-5, "integration_scalars": 1e-3,
                          "clip_norm": 1.0},
            "runtime_sha256": durable.canonical_sha(runtime),
            "implementation_fingerprint": implementation_fingerprint(),
            "implementation_sha256": implementation_sha(),
            "ground_truth_used": False}


def _optimizer(groups, arm):
    selected = ("encoder", "graph") if arm == "encoder_graph" else tuple(groups)
    rates = {"encoder": 3e-6, "graph": 3e-5,
             "native_core": 3e-5, "integration_scalars": 1e-3}
    param_groups = [{"params": groups[name], "lr": rates[name], "name": name}
                    for name in selected]
    params = [param for group in param_groups for param in group["params"]]
    if len({id(p) for p in params}) != len(params):
        raise ValueError("duplicate SW0149 Adam parameter")
    return torch.optim.Adam(param_groups), params


def _finalize(arm, *, foundation, optimizer, history, bindings,
              work_dir, output_dir, before_hashes, before_frozen):
    if len(history) != UPDATES or output_dir.exists():
        raise ValueError("SW0149 cannot finalize incomplete or pre-existing output")
    frozen_names = ("native_core", "integration_scalars")
    frozen_after = _parameter_sha([p for name in frozen_names
                                   for p in _groups(foundation)[name]])
    if arm == "encoder_graph" and frozen_after != before_frozen:
        raise AssertionError("SW0149 encoder_graph arm changed frozen native parameters")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint = staging / "checkpoint.pt"
        optimizer_path = staging / "optimizer.pt"
        history_path = staging / "history.json"
        with checkpoint.open("xb") as stream:
            torch.save({"wrapped": foundation.wrapped.state_dict(),
                        "encoder": foundation.encoder.state_dict()}, stream)
            stream.flush(); os.fsync(stream.fileno())
        with optimizer_path.open("xb") as stream:
            torch.save(optimizer.state_dict(), stream)
            stream.flush(); os.fsync(stream.fileno())
        with history_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(history, stream, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        manifest = {"status": "training_complete", "experiment": NAME,
                    "arm": arm, "seed": SEED, "updates": UPDATES,
                    "completed_updates": UPDATES, "logical_batch": BATCH,
                    "microbatch": MICRO, "exposures": BATCH * UPDATES,
                    "train_ids": [image_id for row in history for image_id in row["image_ids"]],
                    "source_core_sha256": bindings["source_core_sha256"],
                    "source_manifest_sha256": bindings["source_manifest_sha256"],
                    "source_training_ids_sha256": bindings["source_training_ids_sha256"],
                    "ordered_train_ids_sha256": bindings["ordered_train_ids_sha256"],
                    "rgb_asset_validation": bindings["rgb_asset_validation"],
                    "preflight_sha256": bindings["preflight_sha256"],
                    "initial_parameter_sha256": bindings["initial_parameter_sha256"],
                    "final_parameter_sha256": _group_sha(_groups(foundation)),
                    "frozen_native_sha256_before": before_frozen,
                    "frozen_native_sha256_after": frozen_after,
                    "checkpoint_sha256": sha(checkpoint),
                    "optimizer_sha256": sha(optimizer_path),
                    "history_sha256": sha(history_path),
                    "optimizer_step_range": _optimizer_step_range(optimizer),
                    "implementation_fingerprint": bindings["implementation_fingerprint"],
                    "implementation_sha256": bindings["implementation_sha256"],
                    "bindings": bindings, "ground_truth_used": False}
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        # Keep unique incomplete staging directories for forensic inspection.
        raise


def _optimizer_step_range(optimizer):
    steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is not None:
            steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return [min(steps), max(steps)] if steps else [0, 0]


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT,
              work_root=WORK, output_root=TRAIN):
    if arm not in ARMS:
        raise ValueError("unregistered SW0149 arm")
    admission = validate_preflight_admission(preflight_path)
    output_dir = Path(output_root) / f"{arm}_seed1"
    work_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    if output_dir.exists():
        raise FileExistsError(f"preserve existing SW0149 training result: {output_dir}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0149 training requires an exclusive CUDA device")
    torch.cuda.set_device(dev)
    torch.manual_seed(149001)
    torch.cuda.manual_seed_all(149001)
    foundation = sw135.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
    if (foundation.provenance.get("source_core_sha256") != admission["source"]["core_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha(admission["source"]["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != admission["assets"]
            or list(map(int, foundation.image_ids)) != admission["source"]["ids"]
            or not np.array_equal(foundation.pool_indices, admission["source"]["rows"])):
        raise ValueError("SW0149 loaded foundation no longer matches preflight source/order/assets")
    groups = _groups(foundation)
    _set_trainability(foundation, groups, arm)
    initial_hashes = _group_sha(groups)
    if initial_hashes != admission["arms"][arm]["initial_parameter_sha256"]:
        raise ValueError("SW0149 fresh source initialization differs from its preflight disposable arm")
    before_frozen = _parameter_sha([p for name in ("native_core", "integration_scalars")
                                    for p in groups[name]])
    teacher = _teacher(foundation)
    optimizer, params = _optimizer(groups, arm)
    runtime = durable.runtime_signature(dev)
    bindings = _bindings(arm, admission, foundation, initial_hashes, runtime)
    ids, rows = admission["source"]["ids"], admission["source"]["rows"]
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered SW0149 clean TRAIN cache changed shape/dtype")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.exists() and (work_dir / "boundary.ptx").exists():
        metadata = durable.inspect_committed_metadata(work_dir)
        if (metadata.get("bindings") != bindings or metadata.get("runtime") != runtime):
            raise ValueError("SW0149 committed checkpoint lineage/runtime differs")
        state = durable.load_latest(work_dir, wrapped=foundation.wrapped,
                                    encoder=foundation.encoder, optimizer=optimizer,
                                    expected_bindings=bindings, expected_runtime=runtime,
                                    device=dev, max_updates=UPDATES)
        completed, history = state["completed_updates"], state["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise ValueError("SW0149 work directory without a valid boundary has unknown files")
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=False)
        history, completed = [], 0
        durable.save_boundary(work_dir, completed_updates=0,
                              wrapped=foundation.wrapped, encoder=foundation.encoder,
                              optimizer=optimizer, history=history, bindings=bindings,
                              runtime=runtime, device=dev)
    if completed == UPDATES:
        return _finalize(arm, foundation=foundation, optimizer=optimizer,
                         history=history, bindings=bindings, work_dir=work_dir,
                         output_dir=output_dir,
                         before_hashes=initial_hashes, before_frozen=before_frozen)
    foundation.wrapped.train()
    foundation.encoder.train()
    started = time.perf_counter()

    def commit(row):
        history.append(row)
        durable.save_boundary(work_dir, completed_updates=row["update"],
                              wrapped=foundation.wrapped, encoder=foundation.encoder,
                              optimizer=optimizer, history=history, bindings=bindings,
                              runtime=runtime, device=dev)
        if row["update"] % 16 == 0:
            print(json.dumps({"stage": "train", "arm": arm,
                              "update": row["update"], "total": UPDATES,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)

    for update in range(completed, UPDATES):
        logical_update(arm, update, foundation=foundation, teacher=teacher,
                       optimizer=optimizer, groups=groups, params=params,
                       cache=cache, ordered_rows=rows, ordered_ids=ids,
                       commit=commit)
    return _finalize(arm, foundation=foundation, optimizer=optimizer,
                     history=history, bindings=bindings, work_dir=work_dir,
                     output_dir=output_dir,
                     before_hashes=initial_hashes, before_frozen=before_frozen)


def validate_training(arm, *, output_root=TRAIN, work_root=WORK):
    if arm not in ARMS:
        raise ValueError("unregistered SW0149 arm")
    out = Path(output_root) / f"{arm}_seed1"
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint, optimizer, history_path = (out / "checkpoint.pt", out / "optimizer.pt",
                                            out / "history.json")
    boundary = Path(work_root) / f"{arm}_seed1" / "checkpoints" / "boundary.ptx"
    admission = validate_preflight_admission()
    expected_manifest = admission["source"]["manifest_path"]
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != NAME
            or manifest.get("arm") != arm or manifest.get("seed") != SEED
            or manifest.get("updates") != UPDATES or manifest.get("completed_updates") != UPDATES
            or manifest.get("exposures") != BATCH * UPDATES
            or manifest.get("implementation_sha256") != implementation_sha()
            or manifest.get("checkpoint_sha256") != sha(checkpoint)
            or manifest.get("optimizer_sha256") != sha(optimizer)
            or manifest.get("history_sha256") != sha(history_path)
            or manifest.get("preflight_sha256") != admission["sha256"]
            or manifest.get("source_core_sha256") != admission["source"]["core_sha256"]
            or manifest.get("source_manifest_sha256") != sha(expected_manifest)
            or manifest.get("source_training_ids_sha256") != admission["report"]["source"]["training_ids_sha256"]
            or manifest.get("rgb_asset_validation") != admission["assets"]
            or manifest.get("train_ids") != admission["source"]["ids"]
            or (not isinstance(manifest.get("optimizer_step_range"), list)
                or len(manifest["optimizer_step_range"]) != 2
                or manifest["optimizer_step_range"][1] != UPDATES
                or not 1 <= manifest["optimizer_step_range"][0] <= UPDATES)
            or not boundary.is_file()):
        raise ValueError(f"invalid SW0149 completed training artifact: {arm}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    expected_ids = admission["source"]["ids"]
    if (len(history) != UPDATES
            or [image_id for row in history for image_id in row.get("image_ids", [])] != expected_ids
            or manifest.get("bindings", {}).get("implementation_sha256") != implementation_sha()):
        raise ValueError(f"SW0149 {arm} training history/bindings do not match the fixed endpoint")
    return {"manifest": manifest, "manifest_sha256": sha(manifest_path),
            "checkpoint": checkpoint, "checkpoint_sha256": sha(checkpoint),
            "optimizer": optimizer, "history": history_path}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = train_arm(args.arm, device=args.device)
    print(json.dumps({"status": result["status"], "arm": args.arm,
                      "training_admission": True, "updates": UPDATES}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
