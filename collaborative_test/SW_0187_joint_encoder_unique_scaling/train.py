"""SW0187: one shared 32-proposal prefix, then exact-state 512/1024 forks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0182_unique1024_actual_descent import train as prior
from collaborative_test.SW_0182_unique1024_actual_descent import schedule as schedule1024
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx
from collaborative_test.SW_0186_full_encoder_qce import source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0187_shared_prefix_unique_scaling"
SEEDS = (0, 1, 2)
ARMS = ("control512", "candidate1024")
PREFIX_PROPOSALS = 32
BRANCH_PROPOSALS = 32
PROPOSALS_PER_ARM = 64
OUTPUT_ROOT = HERE / "results_archive" / "training_20261017"
QUEUE_MODULE = "collaborative_test.SW_0187_joint_encoder_unique_scaling.training_queue"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _rng_sha(state) -> str:
    # Canonicalize content: pickle can encode tensor storage identity.
    digest = hashlib.sha256()
    def visit(value, name=""):
        digest.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str): visit(value[key], f"{name}/{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value): visit(item, f"{name}/{index}")
        else:
            digest.update(repr(value).encode("utf-8"))
    visit(state)
    return digest.hexdigest()


def _module(name):
    return prior._module(name)


def implementation_fingerprint():
    relatives = (
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/DESIGN_KO.md",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/train.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/training_queue.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/test_train.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/test_training_queue.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/data.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/schedule.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/train.py",
        "collaborative_test/SW_0186_full_encoder_qce/train.py",
        "collaborative_test/SW_0181_actual_descent_qce/state_transaction.py",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
    )
    files = {name: _sha_file(ROOT / name) for name in relatives}
    return {"experiment": EXPERIMENT, "files": files, "seeds": list(SEEDS),
        "arms": list(ARMS), "shared_prefix_proposals": PREFIX_PROPOSALS,
        "branch_proposals": BRANCH_PROPOSALS, "actual_proposals_per_seed": 96,
        "logical_proposals_per_arm": PROPOSALS_PER_ARM,
        "source_factory": "fresh SW0186 joint original RGB encoder plus zero-context, five-family empty Adam",
        "source_factory_fingerprint": source.implementation_fingerprint(),
        "loss": "SW0179 product-Q class-balanced leave-one-out CE, temperature 0.1",
        "training_mode": "mask-supervised TRAIN modal labels; not unsupervised leaderboard eligible",
        "schedule": "SW0179 fixed two permutations; first32 shared; second32 index or index+512",
        "forward": "T1024, settle512, live-tail64, B16/micro4, global clip1 and registered projection",
        "acceptance": "finite full-forward post-loss < before-max(1e-7,1e-6*abs(before)); no retry"}


def fingerprint_sha256():
    return _json_sha(implementation_fingerprint())


def prepare_training_inputs_cpu():
    inputs = prior.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0187 cache admission must precede CUDA initialization")
    if len(inputs["ordered_train_ids"]) != 1024:
        raise ValueError("SW0187 requires the admitted ordered TRAIN1024 cache")
    control = source._expected_batches()
    candidate = schedule1024.expected_batches_1024()
    if candidate[:32] != control[:32]:
        raise ValueError("registered first32 schedule is not shared across arms")
    if any(index >= 512 for row in control for index in row):
        raise ValueError("control512 schedule includes a non-prefix row")
    return inputs


def _batches(arm):
    if arm == "control512":
        return source._expected_batches()
    if arm == "candidate1024":
        return schedule1024.expected_batches_1024()
    if arm == "shared_prefix":
        return schedule1024.expected_batches_1024()[:PREFIX_PROPOSALS]
    raise ValueError(f"unknown SW0187 arm: {arm}")


def _history_sha(history):
    return _json_sha(history)


def _validate_history(stage, cursor, accepted_updates, history, seed):
    expected = _batches(stage) if stage != "shared_prefix" else _batches("shared_prefix")
    limit = PREFIX_PROPOSALS if stage == "shared_prefix" else PROPOSALS_PER_ARM
    if type(cursor) is not int or not 0 <= cursor <= limit or len(history) != cursor:
        raise ValueError("SW0187 branch cursor/history mismatch")
    if type(accepted_updates) is not int or not 0 <= accepted_updates <= cursor:
        raise ValueError("SW0187 accepted-update count is invalid")
    accepted = 0
    for index, row in enumerate(history):
        if row.get("proposal") != index + 1 or row.get("image_indices") != expected[index]:
            raise ValueError(f"SW0187 history differs from {stage} registered schedule at row {index + 1}")
        if row.get("proposal_seed") != 171000 + int(seed) + index:
            raise ValueError("SW0187 proposal seed differs from registered schedule")
        if type(row.get("accepted")) is not bool:
            raise ValueError("SW0187 history lacks a boolean accept/reject decision")
        before, after = row.get("before_loss"), row.get("after_loss")
        threshold = row.get("acceptance_threshold")
        if not isinstance(before, (int, float)) or not math.isfinite(float(before)):
            raise ValueError("SW0187 history has nonfinite pre-loss")
        after_value = math.nan if after is None else float(after)
        expected_accept, expected_threshold = prior._acceptance(float(before), after_value)
        if (row["accepted"] is not expected_accept
                or not isinstance(threshold, (int, float))
                or float(threshold) != expected_threshold):
            raise ValueError("SW0187 acceptance row differs from registered threshold")
        if row.get("accepted_update_count") != accepted + int(row["accepted"]):
            raise ValueError("SW0187 accepted-update cursor is inconsistent")
        accepted += int(row["accepted"])
    if accepted != accepted_updates:
        raise ValueError("SW0187 accepted count differs from proposal history")


def _paths(root, stage, seed):
    work = Path(root) / "work" / f"{stage}_seed{seed}"
    output = Path(root) / f"{stage}_seed{seed}"
    return work, output


def _branch_bindings(arm, seed, inputs, fresh, control, prefix_state_sha,
                     prefix_rng_sha, prefix_history_sha):
    parent = fresh["bindings"]
    return {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
        "objective": "SW0179 product_q_ce; no auxiliary loss",
        "source_binding": parent["source_binding"],
        "source_state_sha256": parent["source_state_sha256"],
        "base_encoder_sha256": parent["base_encoder_sha256"],
        "initial_model_state_sha256": parent["initial_model_state_sha256"],
        "factory_binding": parent,
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "ordered_train_ids": list(map(int, inputs["ordered_train_ids"])),
        "prefix_state_sha256": prefix_state_sha,
        "prefix_rng_sha256": prefix_rng_sha,
        "prefix_history_sha256": prefix_history_sha,
        "control_prefix_manifest_sha256": control["manifest_sha256"],
        "control_prefix_history_sha256": control["history_sha256"],
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "validation_data_read": False, "training_fingerprint": implementation_fingerprint()}


def _archived_prefix_comparison(row, old_row, index):
    """Historical SW0186 comparison only; it never gates SW0187 execution."""
    return {"proposal": int(index) + 1,
        "batch_match": row.get("image_indices") == old_row.get("image_indices"),
        "acceptance_match": row.get("accepted") == old_row.get("accepted"),
        "start_state_sha_match": row.get("start_transaction_sha256") == old_row.get("start_transaction_sha256"),
        "proposal_state_sha_match": row.get("proposal_transaction_sha256") == old_row.get("proposal_transaction_sha256"),
        "before_loss_abs_delta": abs(float(row["before_loss"]) - float(old_row["before_loss"])),
        "after_loss_abs_delta": (None if row.get("after_loss") is None or old_row.get("after_loss") is None
            else abs(float(row["after_loss"]) - float(old_row["after_loss"])))}


def _runtime_base(checkpoint, device):
    return checkpoint.runtime_signature(device)


def _save_boundary(checkpoint, work, stage, cursor, found, optimizer, history,
                   bindings, runtime_base, device, accepted_updates):
    runtime = {**runtime_base, "sw0187_stage": stage, "sw0187_cursor": int(cursor),
        "accepted_updates": int(accepted_updates), "prefix_shared": True}
    checkpoint.save_boundary(work, completed_updates=int(cursor), wrapped=found.wrapped,
        encoder=found.encoder, optimizer=optimizer, history=history,
        bindings=bindings, runtime=runtime, device=device)


def _load_boundary(checkpoint, work, stage, found, optimizer, bindings, runtime_base, device, limit):
    metadata = checkpoint.inspect_committed_metadata(work)
    runtime = metadata.get("runtime")
    custom = {"sw0187_stage", "sw0187_cursor", "accepted_updates", "prefix_shared"}
    if (not isinstance(runtime, dict)
            or {key: value for key, value in runtime.items() if key not in custom} != runtime_base
            or runtime.get("sw0187_stage") != stage
            or runtime.get("sw0187_cursor") != metadata.get("completed_updates")
            or runtime.get("prefix_shared") is not True):
        raise ValueError("SW0187 checkpoint runtime/stage binding mismatch")
    loaded = checkpoint.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
        optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
        device=device, max_updates=limit)
    accepted = sum(int(row["accepted"]) for row in loaded["history"])
    if runtime.get("accepted_updates") != accepted:
        raise ValueError("SW0187 accepted-update counter differs from its history")
    _validate_history(stage, loaded["completed_updates"], accepted, loaded["history"],
                      bindings["seed"])
    return loaded["completed_updates"], accepted, loaded["history"]


def _prefix_control(seed):
    from collaborative_test.SW_0186_full_encoder_qce import train as archived
    path=archived.OUTPUT_ROOT / f"product_q_ce_seed{int(seed)}"
    manifest=archived.validate_training_artifact("product_q_ce",path,seed=seed)
    history=json.loads((path/"history.json").read_text(encoding="utf-8"))["proposals"]
    if len(history)!=64:raise ValueError("SW0187 requires complete archived SW0186 history")
    return dict(manifest=manifest,history=history,manifest_sha256=_sha_file(path/"manifest.json"),
                history_sha256=_sha_file(path/"history.json"))


def _do_proposal(stage, index, seed, batch, found, optimizer, inputs, device,
                 history, accepted_updates, bindings, work, checkpoint, runtime_base):
    torch.manual_seed(171000 + seed + index)
    torch.cuda.manual_seed(171000 + seed + index)
    optimizer.zero_grad(set_to_none=True)
    modules = {"wrapped": found.wrapped, "encoder": found.encoder}
    snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
    start_state_sha = tx.transaction_digest(snapshot)
    before_full = prior._measure_preserving_rng("product_q_ce", found, np.asarray(batch, dtype=np.int64),
        inputs, device, evaluation_rng=snapshot["rng"]) if index < 2 else None
    if before_full is not None:
        tx.restore_rng_state(snapshot["rng"])
    row = source._train_update("product_q_ce", found, optimizer, np.asarray(batch, dtype=np.int64),
        inputs, device, index + 1)
    before = float(row["mean_registered_loss"])
    if before_full is not None:
        tolerance = max(1e-7, 1e-6 * abs(before))
        if not math.isfinite(before_full) or abs(before - before_full) >= tolerance:
            tx.restore_transaction(snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
            raise RuntimeError("SW0187 before-loss parity failed on first two shared proposals")
        row["before_loss_full_forward"] = before_full
        row["before_loss_parity_abs_error"] = abs(before - before_full)
    rng_after_gradient = tx.capture_rng_state()
    proposal_sha = prior._state_digest(modules, optimizer)
    after = prior._measure_preserving_rng("product_q_ce", found, np.asarray(batch, dtype=np.int64),
        inputs, device, evaluation_rng=snapshot["rng"], allow_numerical_rejection=True)
    if prior._state_digest(modules, optimizer) != proposal_sha:
        raise RuntimeError("SW0187 post-loss measurement mutated the proposal")
    accepted, threshold = prior._commit_decision(snapshot, modules, optimizer,
        before, after, rng_after_gradient)
    row.update({"proposal": index + 1, "proposal_seed": 171000 + seed + index,
        "image_indices": list(map(int, batch)), "accepted": bool(accepted),
        "before_loss": before, "after_loss": float(after) if math.isfinite(float(after)) else None,
        "acceptance_threshold": float(threshold), "start_transaction_sha256": start_state_sha,
        "proposal_transaction_sha256": proposal_sha,
        "accepted_update_count": accepted_updates + int(accepted),
        "proposal_cursor": index + 1, "shared_prefix_stage": stage == "shared_prefix"})
    if stage == "shared_prefix" and index < 2:
        row["first_two_credit_failures"] = source._credit_failures(row)
    history = history + [row]
    accepted_updates += int(accepted)
    _save_boundary(checkpoint, work, stage, index + 1, found, optimizer, history,
                   bindings, runtime_base, device, accepted_updates)
    return history, accepted_updates


def train_seed(seed, physical_gpu, output_root=OUTPUT_ROOT):
    seed = int(seed)
    if seed not in SEEDS or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("SW0187 worker requires registered seed and leased physical GPU")
    inputs = prepare_training_inputs_cpu()
    fresh_cpu = source.prepare_fresh_training_state_cpu(seed, inputs)
    initial_sha = fresh_cpu["proof"]["initial_model_state_sha256"]
    control = _prefix_control(seed)
    if initial_sha != control["manifest"]["initial_model_state_sha256"]:
        raise ValueError("SW0187 fresh source differs from SW0186 initialized control")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0187 source/data admission must finish before CUDA initialization")
    torch.set_num_threads(1)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    found, optimizer, _families, fresh_binding, setup = source._make_fresh(
        "product_q_ce", seed, inputs, device)
    found._sw0181_optimizer = optimizer
    source_utils, base_encoder, frozen_before = setup[1], setup[2], setup[3]
    checkpoint = prior._module(source.CHECKPOINT_MODULE)
    runtime_base = _runtime_base(checkpoint, device)
    if fresh_binding["initial_model_state_sha256"] != initial_sha:
        raise ValueError("SW0187 GPU source initialization differs from CPU proof")
    fresh = {"bindings": fresh_binding}
    input_hash_binding = {"source_binding": fresh_binding["source_binding"]}
    input_hashes_before = prior._immutable_input_hashes(inputs, input_hash_binding)
    control_result = _prefix_control(seed)
    root = Path(output_root)
    prefix_work, prefix_output = _paths(root, "shared_prefix", seed)
    prefix_bind = _branch_bindings("shared_prefix", seed, inputs, fresh, control_result,
        initial_sha, "uncommitted", "uncommitted")
    if prefix_output.exists():
        shared = validate_prefix_artifact(prefix_output, seed=seed,
            expected_fingerprint=fingerprint_sha256())
        prefix_history = shared["history"]
        prefix_state_sha = shared["prefix_state_sha256"]
        prefix_rng_sha = shared["prefix_rng_sha256"]
    else:
        prefix_work.mkdir(parents=True, exist_ok=True)
        if (prefix_work / "boundary.ptx").is_file():
            cursor, accepted, prefix_history = _load_boundary(checkpoint, prefix_work,
                "shared_prefix", found, optimizer, prefix_bind, runtime_base, device, PREFIX_PROPOSALS)
        else:
            cursor, accepted, prefix_history = 0, 0, []
            _save_boundary(checkpoint, prefix_work, "shared_prefix", 0, found, optimizer,
                prefix_history, prefix_bind, runtime_base, device, accepted)
        _validate_history("shared_prefix", cursor, accepted, prefix_history, seed)
        rejection_path = prefix_work / "credit_rejection.json"
        prior_failures = [reason for row in prefix_history[:2]
                          for reason in row.get("first_two_credit_failures", [])]
        if prior_failures:
            if rejection_path.is_file():
                rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
                if rejection.get("history_sha256") != _history_sha(prefix_history):
                    raise ValueError("SW0187 prefix credit rejection no longer matches history")
                return rejection
            rejection = {"experiment": EXPERIMENT, "status": "scientific_rejection",
                "seed": seed, "proposal_cursor": cursor, "accepted_updates": accepted,
                "history_sha256": _history_sha(prefix_history),
                "training_fingerprint": fingerprint_sha256(),
                "rejection_reasons": prior_failures, "validation_data_read": False}
            prior._atomic_json_create(rejection_path, rejection)
            return rejection
        found.encoder.train(); found.encoder.base_encoder.eval(); found.patcher.eval(); found.wrapped.train()
        for index in range(cursor, PREFIX_PROPOSALS):
            prefix_history, accepted = _do_proposal("shared_prefix", index, seed,
                _batches("shared_prefix")[index], found, optimizer, inputs, device,
                prefix_history, accepted, prefix_bind, prefix_work, checkpoint, runtime_base)
            # The archived SW0181 prefix is provenance only, not an execution gate.
            if prefix_history[-1].get("first_two_credit_failures"):
                rejection = {"experiment": EXPERIMENT, "status": "scientific_rejection",
                    "seed": seed, "proposal_cursor": index + 1, "accepted_updates": accepted,
                    "history_sha256": _history_sha(prefix_history),
                    "training_fingerprint": fingerprint_sha256(),
                    "rejection_reasons": prefix_history[-1]["first_two_credit_failures"],
                    "validation_data_read": False}
                prior._atomic_json_create(prefix_work / "credit_rejection.json", rejection)
                return rejection
        prefix_state = tx.capture_transaction({"wrapped": found.wrapped, "encoder": found.encoder},
                                              optimizer, offload_to_cpu=True)
        prefix_state_sha = tx.transaction_digest(prefix_state)
        prefix_rng_sha = _rng_sha(prefix_state["rng"])
        if len(prefix_history) != PREFIX_PROPOSALS:
            raise RuntimeError("SW0187 shared prefix did not complete all32 proposals")
        _publish_prefix(prefix_output, prefix_work, seed, prefix_history, prefix_bind,
            prefix_state_sha, prefix_rng_sha, inputs, accepted, control_result)

    branch_results = {}
    for arm in ARMS:
        work, output = _paths(root, arm, seed)
        branch_binding = _branch_bindings(arm, seed, inputs, fresh, control_result,
            prefix_state_sha, prefix_rng_sha, _history_sha(prefix_history))
        if output.exists():
            branch_results[arm] = validate_training_artifact(arm, output, seed=seed,
                expected_bindings=branch_binding)
            continue
        # Rebuild a clean source instance, then load the immutable shared boundary.
        found, optimizer, _families, gpu_binding, setup = source._make_fresh(
            "product_q_ce", seed, inputs, device)
        found._sw0181_optimizer = optimizer
        source_utils, base_encoder, branch_frozen = setup[1], setup[2], setup[3]
        if branch_frozen != frozen_before:
            raise ValueError("SW0187 source foundation changed across branch factory calls")
        shared_work, _ = _paths(root, "shared_prefix", seed)
        shared_bind = prefix_bind
        cursor, accepted, loaded_history = _load_boundary(checkpoint, shared_work,
            "shared_prefix", found, optimizer, shared_bind, runtime_base, device, PREFIX_PROPOSALS)
        if cursor != PREFIX_PROPOSALS or _history_sha(loaded_history) != _history_sha(prefix_history):
            raise ValueError("SW0187 branch fork source is not the completed shared prefix")
        fork_snapshot = tx.capture_transaction({"wrapped": found.wrapped, "encoder": found.encoder},
                                               optimizer, offload_to_cpu=True)
        fork_state_sha = tx.transaction_digest(fork_snapshot)
        fork_rng_sha = _rng_sha(fork_snapshot["rng"])
        if fork_state_sha != prefix_state_sha or fork_rng_sha != prefix_rng_sha:
            raise ValueError("SW0187 branch fork did not restore exact prefix model/Adam/RNG state")
        if gpu_binding["initial_model_state_sha256"] != initial_sha:
            raise ValueError("SW0187 fork factory source differs from registered source")
        work.mkdir(parents=True, exist_ok=True)
        if (work / "boundary.ptx").is_file():
            branch_cursor, branch_accepted, branch_history = _load_boundary(checkpoint, work,
                arm, found, optimizer, branch_binding, runtime_base, device, PROPOSALS_PER_ARM)
            if _history_sha(branch_history[:PREFIX_PROPOSALS]) != _history_sha(prefix_history):
                raise ValueError("SW0187 resumed branch prefix differs from immutable shared history")
        else:
            if any(work.iterdir()):
                raise FileExistsError("preserving unexpected partial SW0187 branch work directory")
            branch_cursor, branch_accepted, branch_history = PREFIX_PROPOSALS, accepted, list(prefix_history)
            _save_boundary(checkpoint, work, arm, branch_cursor, found, optimizer, branch_history,
                branch_binding, runtime_base, device, branch_accepted)
        _validate_history(arm, branch_cursor, branch_accepted, branch_history, seed)
        found.encoder.train(); found.encoder.base_encoder.eval(); found.patcher.eval(); found.wrapped.train()
        arm_batches = _batches(arm)
        for index in range(branch_cursor, PROPOSALS_PER_ARM):
            batch = arm_batches[index]
            branch_history, branch_accepted = _do_proposal(arm, index, seed, batch,
                found, optimizer, inputs, device, branch_history, branch_accepted,
                branch_binding, work, checkpoint, runtime_base)
        branch_results[arm] = _publish_branch(arm, seed, output, work, found,
            optimizer, branch_history, branch_binding, runtime_base, inputs,
            branch_accepted, prefix_state_sha)
        if source_utils._frozen_source_components(found, base_encoder) != frozen_before:
            raise ValueError(f"SW0187 {arm} changed frozen SourceT foundation components")
    if prior._immutable_input_hashes(inputs, input_hash_binding) != input_hashes_before:
        raise RuntimeError("SW0187 TRAIN/source input cache changed during the experiment")
    return {"status": "training_complete", "seed": seed, "prefix_state_sha256": prefix_state_sha,
        "arms": branch_results, "actual_proposals": 96, "validation_data_read": False}


def _publish_prefix(output, work, seed, history, bindings, state_sha, rng_sha,
                    inputs, accepted, archived_control):
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        _write_state_bundle(stage, "shared_prefix", seed, history, bindings, work, inputs,
            accepted, PREFIX_PROPOSALS, state_sha, rng_sha, None)
        manifest_path = stage / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["archived_186_prefix_comparison"] = [
            _archived_prefix_comparison(row, old, index)
            for index, (row, old) in enumerate(zip(history, archived_control["history"][:PREFIX_PROPOSALS]))]
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.rename(stage, output)
    except Exception:
        # Preserve any stage for diagnosis; no recursive deletion is performed.
        raise


def _publish_branch(arm, seed, output, work, found, optimizer, history, bindings,
                    runtime, inputs, accepted, prefix_state_sha):
    if output.exists():
        raise FileExistsError("preserving existing SW0187 branch artifact")
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        _write_state_bundle(stage, arm, seed, history, bindings, work, inputs,
            accepted, PROPOSALS_PER_ARM, prefix_state_sha,
            bindings["prefix_rng_sha256"], (found, optimizer, runtime))
        os.rename(stage, output)
    except Exception:
        raise
    return validate_training_artifact(arm, output, seed=seed, expected_bindings=bindings)


def _write_state_bundle(stage, arm, seed, history, bindings, work, inputs,
                        accepted, proposals, prefix_state_sha, prefix_rng_sha, model_parts):
    checkpoint = None
    if model_parts is not None:
        found, optimizer, _runtime = model_parts
        checkpoint = {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
            "bindings": bindings,
            "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
            "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
        checkpoint["model_state_sha256"] = source._state_sha(found.wrapped, found.encoder)
        torch.save(checkpoint, stage / "checkpoint.pt")
        torch.save(optimizer.state_dict(), stage / "optimizer.pt")
    shutil.copyfile(work / "boundary.ptx", stage / "boundary.ptx")
    history_path = stage / "history.json"
    with history_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"proposals": history}, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
        "status": "shared_prefix_complete" if arm == "shared_prefix" else "training_complete",
        "proposal_budget": proposals, "proposal_cursor": proposals,
        "completed_proposals": proposals, "accepted_updates": int(accepted),
        "optimizer_steps": int(accepted),
        "unique_train_images": 1024 if arm == "candidate1024" else 512,
        "logical_proposal_image_exposures": proposals * 16,
        "actual_parent_proposal_count": 96,
        "actual_parent_image_exposures": 96 * 16,
        "full_forward_batch_equivalents_per_parent": 194,
        "logical_arm_full_forward_batch_equivalents": 130,
        "actual_proposals_per_seed": 96, "shared_prefix_proposals": PREFIX_PROPOSALS,
        "branch_proposals": BRANCH_PROPOSALS, "batch": 16, "microbatch": 4,
        "prefix_state_sha256": prefix_state_sha,
        "prefix_rng_sha256": prefix_rng_sha,
        "boundary_sha256": _sha_file(stage / "boundary.ptx"),
        "history_sha256": _sha_file(history_path), "bindings": bindings,
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "train_image_indices": [int(i) for row in history for i in row["image_indices"]],
        "train_ids": [int(inputs["ordered_train_ids"][i])
                      for row in history for i in row["image_indices"]],
        "proposal_train_ids": [[int(inputs["ordered_train_ids"][i]) for i in row["image_indices"]]
                                for row in history],
        "unique_train_ids": list(dict.fromkeys(int(inputs["ordered_train_ids"][i])
            for row in history for i in row["image_indices"])),
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "validation_data_read": False, "ground_truth_scope": "TRAIN modal labels only",
        "training_fingerprint": implementation_fingerprint()}
    if checkpoint is not None:
        manifest["checkpoint_sha256"] = _sha_file(stage / "checkpoint.pt")
        manifest["optimizer_sha256"] = _sha_file(stage / "optimizer.pt")
        manifest["model_state_sha256"] = checkpoint["model_state_sha256"]
        final_encoder=source._base_encoder_sha(found)
        manifest.update(source_base_trainable=True,
            source_base_encoder_initial_sha256=bindings["base_encoder_sha256"],
            source_base_encoder_final_sha256=final_encoder,
            source_base_encoder_changed=final_encoder!=bindings["base_encoder_sha256"])
        manifest["optimizer_step_summary"] = source._module(source.SOURCE_UTIL_MODULE)._optimizer_summary(optimizer)
    path = stage / "manifest.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validate_prefix_artifact(path, *, seed, expected_fingerprint):
    path = Path(path); manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    history_obj = json.loads((path / "history.json").read_text(encoding="utf-8"))
    history = history_obj["proposals"]
    _validate_history("shared_prefix", PREFIX_PROPOSALS, manifest.get("accepted_updates"), history, seed)
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("arm") != "shared_prefix"
            or manifest.get("seed") != int(seed)
            or manifest.get("training_fingerprint") != implementation_fingerprint()
            or expected_fingerprint != fingerprint_sha256()
            or manifest.get("history_sha256") != _sha_file(path / "history.json")
            or manifest.get("boundary_sha256") != _sha_file(path / "boundary.ptx")):
        raise ValueError("SW0187 prefix fingerprint changed")
    if (len(manifest.get("archived_186_prefix_comparison", [])) != PREFIX_PROPOSALS
            or manifest.get("accepted_updates") != sum(int(row["accepted"]) for row in history)):
        raise ValueError("SW0187 shared-prefix manifest counters/comparison are incomplete")
    return {"manifest": manifest, "history": history,
        "prefix_state_sha256": manifest["prefix_state_sha256"],
        "prefix_rng_sha256": manifest["prefix_rng_sha256"]}


def _assert_trained_encoder_manifest(manifest,encoder_state):
    from collaborative_test.SW_0186_full_encoder_qce import score as encoder_guard
    encoder_guard._assert_source_encoder_learning(manifest)
    if (manifest.get("source_base_encoder_initial_sha256")!=manifest.get("bindings",{}).get("base_encoder_sha256")
            or manifest.get("source_base_encoder_final_sha256")!=source._base_encoder_sha_from_state(encoder_state)):
        raise ValueError("SW0187 declared encoder change differs from actual checkpoint tensors")


def validate_training_artifact(arm, root, *, seed, expected_bindings=None):
    if arm not in ARMS:
        raise ValueError("unknown SW0187 branch")
    root = Path(root)
    manifest_path = root / "manifest.json"
    history_path = root / "history.json"
    checkpoint_path = root / "checkpoint.pt"
    optimizer_path = root / "optimizer.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))["proposals"]
    if manifest.get("experiment") != EXPERIMENT or manifest.get("arm") != arm or manifest.get("seed") != int(seed):
        raise ValueError("SW0187 manifest identity mismatch")
    if manifest.get("training_fingerprint") != implementation_fingerprint():
        raise ValueError("SW0187 artifact fingerprint mismatch")
    if expected_bindings is not None and manifest.get("bindings") != expected_bindings:
        raise ValueError("SW0187 artifact bindings differ from current inputs")
    _validate_history(arm, PROPOSALS_PER_ARM, manifest.get("accepted_updates"), history, seed)
    ordered_ids = manifest.get("bindings", {}).get("ordered_train_ids")
    if not isinstance(ordered_ids, list) or len(ordered_ids) != 1024:
        raise ValueError("SW0187 artifact lacks its ordered TRAIN1024 identity binding")
    expected_indices = [int(i) for row in history for i in row["image_indices"]]
    expected_ids = [int(ordered_ids[i]) for row in history for i in row["image_indices"]]
    expected_proposal_ids = [[int(ordered_ids[i]) for i in row["image_indices"]] for row in history]
    if (manifest.get("train_image_indices") != expected_indices
            or manifest.get("train_ids") != expected_ids
            or manifest.get("proposal_train_ids") != expected_proposal_ids
            or manifest.get("logical_proposal_image_exposures") != 1024
            or manifest.get("unique_train_images") != (1024 if arm == "candidate1024" else 512)
            or len(set(manifest.get("unique_train_ids", []))) != manifest["unique_train_images"]):
        raise ValueError("SW0187 artifact exposures/ordered IDs differ from its bound history")
    if _sha_file(history_path) != manifest.get("history_sha256"):
        raise ValueError("SW0187 history checksum mismatch")
    if _sha_file(checkpoint_path) != manifest.get("checkpoint_sha256") or _sha_file(optimizer_path) != manifest.get("optimizer_sha256"):
        raise ValueError("SW0187 model/optimizer artifact checksum mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("bindings") != manifest.get("bindings")
            or payload.get("model_state_sha256") != manifest.get("model_state_sha256")
            or source._state_sha_from_payload(payload) != manifest.get("model_state_sha256")):
        raise ValueError("SW0187 checkpoint and manifest state differ")
    _assert_trained_encoder_manifest(manifest,payload["encoder"])
    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    source._module(source.SOURCE_UTIL_MODULE)._validate_optimizer_state(
        optimizer_state, manifest.get("optimizer_step_summary"))
    parameter_steps = [int(state["step"].item()) for state in optimizer_state["state"].values()]
    if (len(history) != PROPOSALS_PER_ARM
            or (manifest["accepted_updates"] == 0 and parameter_steps)
            or any(step != manifest["accepted_updates"] for step in parameter_steps)):
        raise ValueError("SW0187 Adam moments/steps differ from accepted proposal history")
    if _sha_file(root / "boundary.ptx") != manifest.get("boundary_sha256"):
        raise ValueError("SW0187 final boundary checksum mismatch")
    return {"status": "training_complete", "manifest": manifest, "history": history,
        "manifest_sha256": _sha_file(manifest_path), "checkpoint_sha256": _sha_file(checkpoint_path),
        "model_state_sha256": manifest["model_state_sha256"]}
