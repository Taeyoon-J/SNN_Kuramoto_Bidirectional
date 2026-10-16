"""Private no-commit replay probe for SW0182 proposal 2 (seed 1 only).

The parent admits the existing cursor-1 boundary and obtains an exclusive idle
GPU lease. The child replays proposal 2 twice with the 1024 cache and twice with
the original 512 prefix cache, restoring the same boundary, Adam, and RNG before
each replay. It never saves a training boundary or touches a validation mask.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_WORKER = "--worker" in sys.argv
_ASSIGNED = os.environ.get("CUDA_VISIBLE_DEVICES") if _WORKER else None
if not _WORKER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODULE = "collaborative_testing.collaborative_private_runtime_20261008.SW0182_delivery_20261016.proposal2_replay_probe"
TRAIN_MODULE = "collaborative_test.SW_0182_unique1024_actual_descent.train"
QUEUE_MODULE = "collaborative_test.SW_0182_unique1024_actual_descent.training_queue"
CONTROL_MODULE = "collaborative_test.SW_0181_actual_descent_qce.train"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
SEED = 1
ARM = "product_q_ce"
PROPOSAL_INDEX = 1  # zero-based index for the second committed proposal
PROPOSAL_SEED = 171000 + SEED + PROPOSAL_INDEX
RUNTIME_DIR = HERE / "proposal2_replay"
STOP = threading.Event()


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _json_number(value):
    number = float(value)
    return number if np.isfinite(number) else None


def _rng_sha(state):
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def _tensor_tree(value, prefix=""):
    found = {}
    if isinstance(value, torch.Tensor):
        found[prefix] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key, child in value.items():
            found.update(_tensor_tree(child, f"{prefix}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found.update(_tensor_tree(child, f"{prefix}/{index}"))
    return found


def _difference(left, right):
    left_rows, right_rows = _tensor_tree(left), _tensor_tree(right)
    names = sorted(set(left_rows) | set(right_rows))
    rows = []
    total_different = 0
    maximum = 0.0
    squared = 0.0
    reference_squared = 0.0
    nonfinite = 0
    first = None
    for name in names:
        if name not in left_rows or name not in right_rows:
            rows.append({"name": name, "missing_from": "left" if name not in left_rows else "right"})
            total_different += 1
            first = first or name
            continue
        a, b = left_rows[name], right_rows[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append({"name": name, "left_shape": list(a.shape), "right_shape": list(b.shape),
                         "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)})
            total_different += 1; first = first or name
            continue
        aa, bb = a.double().numpy(), b.double().numpy()
        finite = np.isfinite(aa) & np.isfinite(bb)
        nonfinite += int((~finite).sum())
        diff = np.zeros(aa.shape, dtype=np.float64)
        diff[finite] = np.abs(aa[finite] - bb[finite])
        count = int(np.count_nonzero((aa != bb) | (~finite)))
        local_max = float(diff.max()) if diff.size else 0.0
        local_norm = float(np.linalg.norm(diff.ravel()))
        reference_norm = float(np.linalg.norm(aa[finite].ravel()))
        if count:
            first = first or name
            total_different += count
            maximum = max(maximum, local_max)
            squared += local_norm * local_norm
            reference_squared += reference_norm * reference_norm
            rows.append({"name": name, "different_values": count,
                "max_abs": local_max, "difference_l2": local_norm,
                "nonfinite_values": int((~finite).sum())})
    return {"exact": total_different == 0, "tensor_names_compared": len(names),
        "different_values": total_different, "first_differing_tensor": first,
        "max_abs": maximum, "difference_l2": float(np.sqrt(squared)),
        "relative_difference_l2": float(np.sqrt(squared) / max(float(np.sqrt(reference_squared)), 1e-30)),
        "nonfinite_values": nonfinite, "differences": rows[:25]}


def _named_parameters(found, optimizer):
    names = {}
    for group in optimizer.param_groups:
        group_name = group["name"]
        for parameter_name, parameter in zip(group["parameter_names"], group["params"]):
            names[id(parameter)] = f"{group_name}/{parameter_name}"
    return names


def _clip_capture(found, optimizer):
    names = _named_parameters(found, optimizer)
    original = torch.nn.utils.clip_grad_norm_
    captured = {}

    def wrapped(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured["preclip"] = {names[id(p)]: p.grad.detach().cpu().clone()
            for p in parameters if p.grad is not None}
        result = original(parameters, *args, **kwargs)
        captured["postclip"] = {names[id(p)]: p.grad.detach().cpu().clone()
            for p in parameters if p.grad is not None}
        captured["clip_calls"] = captured.get("clip_calls", 0) + 1
        return result

    return original, wrapped, captured


def _replay(train, found, optimizer, inputs, image_indices, device, start_snapshot, label):
    tx = train.tx
    modules = {"wrapped": found.wrapped, "encoder": found.encoder}
    tx.restore_transaction(start_snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
    torch.manual_seed(PROPOSAL_SEED); torch.cuda.manual_seed(PROPOSAL_SEED)
    optimizer.zero_grad(set_to_none=True)
    snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
    start_sha = tx.transaction_digest(snapshot)
    start_rng_sha = _rng_sha(snapshot["rng"])
    before_params = {f"{group['name']}/{name}": parameter.detach().cpu().clone()
        for group in optimizer.param_groups
        for name, parameter in zip(group["parameter_names"], group["params"])}

    found.encoder.train(); found.encoder.base_encoder.eval(); found.patcher.eval(); found.wrapped.train()
    before_full = train._measure_preserving_rng(ARM, found, image_indices, inputs, device,
        evaluation_rng=snapshot["rng"])
    tx.restore_rng_state(snapshot["rng"])
    original_clip, wrapped_clip, captured = _clip_capture(found, optimizer)
    torch.nn.utils.clip_grad_norm_ = wrapped_clip
    try:
        row = train.base._train_update(ARM, found, optimizer, image_indices, inputs, device,
            PROPOSAL_INDEX + 1)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    if captured.get("clip_calls") != 1:
        raise RuntimeError("proposal replay did not invoke exactly one registered global clipping operation")

    rng_after_gradient = tx.capture_rng_state()
    proposed_snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
    proposed_sha = tx.transaction_digest(proposed_snapshot)
    proposed_live_sha = train._state_digest(modules, optimizer)
    after_full = train._measure_preserving_rng(ARM, found, image_indices, inputs, device,
        evaluation_rng=snapshot["rng"], allow_numerical_rejection=True)
    if train._state_digest(modules, optimizer) != proposed_live_sha:
        raise RuntimeError("full post-loss measurement mutated the diagnostic proposal")
    accepted, threshold = train._acceptance(row["mean_registered_loss"], after_full)

    deltas = {}
    for group in optimizer.param_groups:
        family_rows = []
        for name, parameter in zip(group["parameter_names"], group["params"]):
            qualified_name = f"{group['name']}/{name}"
            old = before_params[qualified_name].double()
            new = parameter.detach().cpu().double()
            delta = new - old
            family_rows.append({"name": qualified_name, "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                "delta_l2": float(delta.norm()), "different_values": int(torch.count_nonzero(delta).item()),
                "nonfinite_values": int((~torch.isfinite(delta)).sum().item())})
        deltas[group["name"]] = family_rows

    committed, committed_threshold = train._commit_decision(snapshot, modules, optimizer,
        row["mean_registered_loss"], after_full, rng_after_gradient)
    committed_snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
    output = {"label": label, "start_transaction_sha256": start_sha,
        "start_rng_sha256": start_rng_sha, "proposed_transaction_sha256": proposed_sha,
        "proposed_live_sha256": proposed_live_sha,
        "committed_transaction_sha256": tx.transaction_digest(committed_snapshot),
        "proposal_seed": PROPOSAL_SEED, "image_indices": list(map(int, image_indices)),
        "image_ids": [int(inputs["ordered_train_ids"][int(i)]) for i in image_indices],
        "before_loss_full1024": _json_number(before_full),
        "before_loss_train_update": _json_number(row["mean_registered_loss"]),
        "after_loss_full1024": _json_number(after_full), "acceptance_threshold": _json_number(threshold),
        "accepted_by_registered_rule": bool(accepted), "committed_as_accepted": bool(committed),
        "commit_threshold": float(committed_threshold),
        "microbatch_q_z_sha256": row["microbatches"],
        "preclip_gradient_tensors": captured.get("preclip", {}),
        "postclip_gradient_tensors": captured.get("postclip", {}),
        "parameter_deltas": deltas,
        "proposed_state": proposed_snapshot, "committed_state": committed_snapshot}
    tx.restore_transaction(start_snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
    return output


def _thin_tensor_dict(tree):
    rows = {}
    for name, value in tree.items():
        norm = float(value.double().norm())
        rows[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_sha(value),
            "l2": norm if np.isfinite(norm) else None, "finite": bool(torch.isfinite(value).all())}
    return rows


def _thin_replay(row):
    return {key: value for key, value in row.items()
        if key not in {"proposed_state", "committed_state", "preclip_gradient_tensors", "postclip_gradient_tensors"}}


def run_worker(gpu, boundary_sha, output_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu) or torch.cuda.is_initialized():
        raise RuntimeError("diagnostic worker must inherit its exclusive physical GPU before CUDA init")
    train = _load(TRAIN_MODULE)
    queue = _load(QUEUE_MODULE)
    control_train = _load(CONTROL_MODULE)
    inputs, _identity = queue._cpu_admission(SEED, train)
    control_inputs = control_train.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic admission unexpectedly initialized CUDA")
    prefix_ids = list(map(int, inputs["ordered_train_ids"][:512]))
    if prefix_ids != list(map(int, control_inputs["ordered_train_ids"])):
        raise ValueError("SW0182 first512 IDs differ from SW0181 control data")
    for key in ("rgb", "modal_gt"):
        if not np.array_equal(inputs["arrays"][key][:512], control_inputs["arrays"][key]):
            raise ValueError(f"SW0182 extended TRAIN cache prefix differs from old SW0171 {key}")
    candidate_batches = train._expected_batches()
    control_batches = control_train._expected_batches()
    candidate_indices = candidate_batches[PROPOSAL_INDEX]
    control_indices = control_batches[PROPOSAL_INDEX]
    if candidate_indices != control_indices or any(index >= 512 for index in candidate_indices):
        raise ValueError("proposal 2 batch does not map to the same fixed SW0181 prefix rows")

    work, _artifact = train._artifact_paths(train.OUTPUT_ROOT, ARM, SEED)
    boundary_path = work / "boundary.ptx"
    if not boundary_path.is_file() or train._sha_file(boundary_path) != boundary_sha:
        raise ValueError("candidate cursor-1 boundary changed before diagnostic GPU load")
    meta = train._module(train.base.CHECKPOINT_MODULE).inspect_committed_metadata(work)
    if meta.get("completed_updates") != 1:
        raise ValueError("proposal replay requires the untouched cursor-1 candidate boundary")
    history = meta.get("history")
    # SW0148 metadata intentionally omits history, so inspect the checksum-validated payload.
    envelope = torch.load(boundary_path, map_location="cpu", weights_only=False)
    payload_bytes = envelope["payload_bytes"]
    if hashlib.sha256(payload_bytes).hexdigest() != envelope.get("payload_sha256"):
        raise ValueError("cursor-1 boundary payload checksum mismatch")
    payload = torch.load(__import__("io").BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    history = payload["history"]
    train._validate_cursor(1, int(payload["runtime"]["accepted_updates"]), history, seed=SEED)
    control = train._control_prefix(SEED)
    train._validate_prefix_control_row(history[0], control["history"][0], index=0)
    expected_control_start = control["history"][PROPOSAL_INDEX]["start_transaction_sha256"]
    expected_control_proposal = control["history"][PROPOSAL_INDEX]["proposal_transaction_sha256"]
    saved_runtime = payload["runtime"]
    saved_bindings = payload["bindings"]
    immutable_hashes_before = train._immutable_input_hashes(inputs, saved_bindings)

    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    found, optimizer, _families, gpu_binding, _setup = train.base._make_fresh(ARM, SEED, inputs, device)
    found._sw0181_optimizer = optimizer
    runtime = train._module(train.base.CHECKPOINT_MODULE)
    runtime_base = runtime.runtime_signature(device)
    if {key: value for key, value in saved_runtime.items() if key not in {
            "sw0182_proposal_cursor_semantics", "proposal_cursor", "accepted_updates"}} != runtime_base:
        raise ValueError("cursor-1 boundary runtime differs from current GPU environment")
    loaded = runtime.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
        optimizer=optimizer, expected_bindings=saved_bindings, expected_runtime=saved_runtime,
        device=device, max_updates=train.PROPOSALS)
    if loaded["completed_updates"] != 1 or loaded["history"] != history:
        raise ValueError("loaded cursor-1 boundary differs from its validated payload")
    if gpu_binding.get("initial_model_state_sha256") != saved_bindings.get("initial_model_state_sha256"):
        raise ValueError("GPU fresh factory initialization differs from cursor-1 source binding")

    modules = {"wrapped": found.wrapped, "encoder": found.encoder}
    tx = train.tx
    torch.manual_seed(PROPOSAL_SEED); torch.cuda.manual_seed(PROPOSAL_SEED)
    original_snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
    original_digest = tx.transaction_digest(original_snapshot)
    if original_digest != expected_control_start:
        raise ValueError("restored cursor-1 state does not match archived SW0181 proposal-2 start")
    replay_rows = {}
    try:
        for label, replay_inputs in (("control512_A", control_inputs), ("control512_B", control_inputs),
                                     ("candidate1024_A", inputs), ("candidate1024_B", inputs)):
            replay_rows[label] = _replay(train, found, optimizer, replay_inputs,
                candidate_indices, device, original_snapshot, label)
        if tx.transaction_digest(tx.capture_transaction(modules, optimizer, offload_to_cpu=True)) != original_digest:
            raise RuntimeError("diagnostic failed to restore original cursor-1 model/Adam state")
    finally:
        tx.restore_transaction(original_snapshot, modules, optimizer, restore_rng=True, clear_grad=True)

    comparisons = {}
    for left, right in (("candidate1024_A", "candidate1024_B"),
                        ("control512_A", "control512_B"),
                        ("candidate1024_A", "control512_A")):
        a, b = replay_rows[left], replay_rows[right]
        comparisons[f"{left}_vs_{right}"] = {
            "start_transaction_equal": a["start_transaction_sha256"] == b["start_transaction_sha256"],
            "start_rng_equal": a["start_rng_sha256"] == b["start_rng_sha256"],
            "proposal_state": _difference(a["proposed_state"], b["proposed_state"]),
            "committed_state": _difference(a["committed_state"], b["committed_state"]),
            "q_hashes_equal": [r["q_sha256"] for r in a["microbatch_q_z_sha256"]]
                == [r["q_sha256"] for r in b["microbatch_q_z_sha256"]],
            "z_hashes_equal": [r["z_sha256"] for r in a["microbatch_q_z_sha256"]]
                == [r["z_sha256"] for r in b["microbatch_q_z_sha256"]],
            "preclip_gradient_tensors": _difference(a["preclip_gradient_tensors"], b["preclip_gradient_tensors"]),
            "postclip_gradient_tensors": _difference(a["postclip_gradient_tensors"], b["postclip_gradient_tensors"]),
            "before_loss_delta": float(a["before_loss_train_update"] - b["before_loss_train_update"]),
            "after_loss_delta": float(a["after_loss_full1024"] - b["after_loss_full1024"]),
            "acceptance_equal": a["accepted_by_registered_rule"] == b["accepted_by_registered_rule"]}
    thin_rows = {name: _thin_replay(row) | {
        "preclip_gradient_summary": _thin_tensor_dict(row["preclip_gradient_tensors"]),
        "postclip_gradient_summary": _thin_tensor_dict(row["postclip_gradient_tensors"])}
        for name, row in replay_rows.items()}
    report = {"experiment": "SW0182_proposal2_same_state_replay_diagnostic", "status": "complete",
        "seed": SEED, "proposal": 2, "proposal_seed": PROPOSAL_SEED,
        "boundary_sha256": boundary_sha, "boundary_cursor": 1,
        "control_manifest_sha256": control["manifest_sha256"],
        "control_history_sha256": control["history_sha256"],
        "control_proposal2_start_transaction_sha256": expected_control_start,
        "control_proposal2_transaction_sha256": expected_control_proposal,
        "candidate_boundary_runtime_signature": saved_runtime,
        "SW0181_control_runtime_signature": control["manifest"].get("runtime"),
        "candidate_dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "candidate_dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "control_cache_manifest_sha256": control_inputs.get("dataset_manifest_sha256"),
        "control_cache_arrays_sha256": control_inputs.get("dataset_arrays_sha256"),
        "proposal_image_indices": list(map(int, candidate_indices)),
        "proposal_image_ids": [prefix_ids[i] for i in candidate_indices],
        "candidate_batch_rgb_sha256": hashlib.sha256(np.ascontiguousarray(
            inputs["arrays"]["rgb"][candidate_indices]).tobytes()).hexdigest(),
        "control_batch_rgb_sha256": hashlib.sha256(np.ascontiguousarray(
            control_inputs["arrays"]["rgb"][control_indices]).tobytes()).hexdigest(),
        "candidate_batch_modal_gt_sha256": hashlib.sha256(np.ascontiguousarray(
            inputs["arrays"]["modal_gt"][candidate_indices]).tobytes()).hexdigest(),
        "control_batch_modal_gt_sha256": hashlib.sha256(np.ascontiguousarray(
            control_inputs["arrays"]["modal_gt"][control_indices]).tobytes()).hexdigest(),
        "prefix512_rgb_and_modal_exact": True, "candidate_vs_control_same_indices": True,
        "training_execution_flags": {"total_steps": 1024, "live_tail_steps": 64,
            "proposal_seed": PROPOSAL_SEED, "microbatch": train.MICRO,
            "batch_size": train.BATCH, "training_mode": True},
        "validation_data_read": False, "optimizer_updates_committed": 0,
        "replays": thin_rows, "comparisons": comparisons,
        "original_boundary_state_restored": True,
        "cursor1_matches_control_proposal2_start": original_digest == expected_control_start,
        "candidate_replay_A_matches_control_proposal2_transaction":
            replay_rows["candidate1024_A"]["proposed_live_sha256"] == expected_control_proposal,
        "control_cache_replay_A_matches_control_proposal2_transaction":
            replay_rows["control512_A"]["proposed_live_sha256"] == expected_control_proposal,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": int(gpu), "initial_model_state_sha256": saved_bindings["initial_model_state_sha256"],
        "immutable_input_hashes_before": immutable_hashes_before,
        "report_generated_unix": time.time()}
    if train._immutable_input_hashes(inputs, saved_bindings) != immutable_hashes_before:
        raise RuntimeError("diagnostic changed the frozen candidate input/source files")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(output_path),
        "comparison_keys": list(comparisons)}, sort_keys=True), flush=True)
    return report


def _parent(seed=SEED):
    if seed != SEED or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("private proposal probe parent is seed1-only and must hide CUDA")
    import fcntl
    queue = _load(QUEUE_MODULE)
    train = _load(TRAIN_MODULE)
    owner = _load(OWNER_MODULE)
    inputs, _identity = queue._cpu_admission(seed, train)
    del inputs
    work, _artifact = train._artifact_paths(train.OUTPUT_ROOT, ARM, seed)
    boundary = work / "boundary.ptx"
    meta = train._module(train.base.CHECKPOINT_MODULE).inspect_committed_metadata(work)
    if meta.get("completed_updates") != 1:
        raise ValueError("diagnostic requires proposal-2 starting at committed cursor 1")
    boundary_runtime = meta.get("runtime")
    preferred_gpu = int(boundary_runtime.get("cuda_visible_devices")) if isinstance(boundary_runtime, dict) else -1
    if preferred_gpu not in range(4):
        raise ValueError("cursor-1 boundary lacks its exact physical-GPU runtime binding")
    boundary_sha = _sha_file(boundary)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME_DIR / f"seed{seed}.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = len(list(RUNTIME_DIR.glob(f"seed{seed}_attempt*.log")))
    try:
        while not STOP.is_set():
            gpu, uuid, _used, lease = owner._acquire(STOP, preferred_gpu=preferred_gpu)
            stable = owner._stable_idle(gpu, uuid, STOP,
                lambda elapsed, memory: print(json.dumps({"status": "waiting_stable_gpu",
                    "gpu": int(gpu), "seconds": float(elapsed), "memory_mib": int(memory)}), flush=True))
            if not stable:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); continue
            live_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
            if live_uuid != uuid or owners or int(used) > owner.MAX_MEMORY_MIB:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); continue
            if _sha_file(boundary) != boundary_sha:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
                raise RuntimeError("cursor-1 boundary changed while waiting for diagnostic GPU")
            attempt += 1
            output = HERE / f"proposal2_replay_seed{seed}_attempt{attempt:02d}.json"
            log = RUNTIME_DIR / f"seed{seed}_attempt{attempt:02d}.log"
            command = [sys.executable, "-u", "-m", MODULE, "--worker", "--physical-gpu",
                str(int(gpu)), "--boundary-sha256", boundary_sha, "--output", str(output)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(int(gpu)), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", USE_PYTORCH_KERNEL_CACHE="0")
            child = None
            try:
                with log.open("x", encoding="utf-8", newline="\n") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                    print(json.dumps({"status": "running_owned_probe", "pid": child.pid,
                        "gpu": int(gpu), "uuid": uuid, "attempt": attempt}, sort_keys=True), flush=True)
                    while child.poll() is None and not STOP.is_set():
                        current_uuid, _memory, active = owner.gpu_guard._gpu_info(gpu)
                        if child.poll() is not None: break
                        if current_uuid != uuid or any(int(pid) != child.pid for pid in active):
                            owner._stop_owned(child)
                            raise RuntimeError("foreign GPU owner interrupted private diagnostic; owned worker stopped")
                        record = owner._proc_record(child.pid)
                        if record and record.get("state") != "Z" and list(record.get("argv") or []) != command:
                            owner._stop_owned(child)
                            raise RuntimeError("diagnostic child argv identity changed")
                        STOP.wait(3)
                    if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                    code = child.wait()
                if code != 0:
                    raise RuntimeError(f"diagnostic worker exited {code}; see {log}")
                return json.loads(output.read_text(encoding="utf-8"))
            finally:
                if child is not None and child.poll() is None: owner._stop_owned(child)
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        raise RuntimeError("private diagnostic stopped before completion")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--boundary-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4) or args.boundary_sha256 is None or args.output is None:
            parser.error("worker requires GPU, boundary checksum, and output path")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            parser.error("worker CUDA visibility does not match assigned physical GPU")
        run_worker(args.physical_gpu, args.boundary_sha256, args.output)
        return 0
    if any(value is not None for value in (args.physical_gpu, args.boundary_sha256, args.output)):
        parser.error("parent does not accept worker-only arguments")
    result = _parent()
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"]},
        sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
