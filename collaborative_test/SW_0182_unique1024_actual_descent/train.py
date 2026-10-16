"""SW0182 bounded actual-descent proposals over unique TRAIN1024 inputs."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0179_qce_seed_replication import train as base
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx
from collaborative_test.SW_0182_unique1024_actual_descent import data as data_module
from collaborative_test.SW_0182_unique1024_actual_descent import schedule

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0182_unique1024_actual_descent"
ARMS = ("product_q_ce",)
SEEDS = (0, 1, 2)
COUNT, PASSES, PROPOSALS = 1024, 1, 64
BATCH, MICRO = base.BATCH, base.MICRO
OUTPUT_ROOT = HERE / "results_archive" / "training_20261016"
CONTROL_ROOT = ROOT / "collaborative_test" / "SW_0181_actual_descent_qce" / "results_archive" / "training_20261016"
CONTROL_SCORE_CANDIDATES = (
    ROOT / "collaborative_test" / "SW_0181_actual_descent_qce" / "results_archive" / "score_v1" / "evaluation.json",
    ROOT / "collaborative_test" / "SW_0181_actual_descent_qce" / "results_archive" / "evaluation_seedmean_20261016.json",
)
CONTROL_SCORE_SHA256 = "3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54"
QUEUE_MODULE = "collaborative_test.SW_0182_unique1024_actual_descent.training_queue"


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def implementation_fingerprint():
    paths = (
        "collaborative_test/SW_0182_unique1024_actual_descent/DESIGN_KO.md",
        "collaborative_test/SW_0182_unique1024_actual_descent/data.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/schedule.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/train.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/test_data.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/test_train.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/training_queue.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/test_training_queue.py",
        "collaborative_test/SW_0181_actual_descent_qce/DESIGN_KO.md",
        "collaborative_test/SW_0181_actual_descent_qce/state_transaction.py",
        "collaborative_test/SW_0181_actual_descent_qce/train.py",
        "collaborative_test/SW_0181_actual_descent_qce/training_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
    )
    files = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0182 fingerprint dependency missing: {relative}")
        files[relative] = _sha_file(path)
    return {
        "experiment": EXPERIMENT,
        "files": files,
        "parent_training_fingerprint": base.fingerprint_sha256(),
        "source_factory": "fresh seed-specific SW0179 SourceT + zero-context; no trained checkpoint reuse",
        "dataset_size": COUNT,
        "batch_schedule_sha256": _history_sha([{"indices": row} for row in _expected_batches()]),
        "seeds": list(SEEDS),
        "proposal_budget": PROPOSALS,
        "acceptance": "finite post < before - max(1e-7,1e-6*abs(before))",
        "gradient_mode": "original SW0179 full-1024 forward; settle512; live-tail64 backward",
        "mask_supervised": True,
        "unsupervised_leaderboard_eligible": False,
    }


def fingerprint_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def prepare_training_inputs_cpu():
    raw = data_module.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0182 TRAIN cache must be admitted before CUDA initialization")
    ids = schedule.validate_ordered_ids(raw["ordered_train_ids"])
    if len(raw["arrays"]["rgb"]) != COUNT or len(raw["arrays"]["modal_gt"]) != COUNT:
        raise ValueError("SW0182 prepared cache does not contain 1024 rows")
    prefix = base.prepare_training_inputs_cpu()
    if (raw["prefix_manifest_sha256"] != prefix["dataset_manifest_sha256"]
            or raw["prefix_arrays_sha256"] != prefix["dataset_arrays_sha256"]):
        raise ValueError("SW0182 cache is not bound to the frozen SW0171 prefix")
    producer_proof = {"experiment": EXPERIMENT,
        "prefix_512_manifest_sha256": raw["prefix_manifest_sha256"],
        "prefix_512_arrays_sha256": raw["prefix_arrays_sha256"],
        "dataset_manifest_sha256": raw["manifest_sha256"],
        "dataset_arrays_sha256": raw["arrays_sha256"],
        "source_checkpoint_sha256": raw["source_checkpoint_sha256"],
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(ids, dtype="<i8").tobytes()),
        "validation_data_read": False, "ground_truth_scope": "TRAIN modal32 only"}
    return {**raw, "ordered_train_ids": ids,
        "dataset_manifest_sha256": raw["manifest_sha256"],
        "dataset_arrays_sha256": raw["arrays_sha256"],
        "dataset_producer_proof": producer_proof,
        "dataset_prepared": {"cache_root": raw["cache_dir"]}}


def prepare_fresh_training_state_cpu(seed, inputs=None):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0182 accepts registered source seeds 0, 1, or 2")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0182 fresh source factory must run before CUDA initialization")
    inputs = prepare_training_inputs_cpu() if inputs is None else inputs
    proof = base.prepare_fresh_training_state_cpu(seed, inputs)
    if proof["proof"]["adam_state_entries"] != 0:
        raise ValueError("SW0182 fresh Adam is not empty")
    proof["proof"]["experiment"] = EXPERIMENT
    proof["proof"]["accept_reject_protocol"] = "one proposal per fixed batch; no retries"
    proof["proof"]["training_fingerprint"] = implementation_fingerprint()
    return proof


def _expected_batches():
    return schedule.expected_batches_1024()


def _history_sha(history):
    return _sha_bytes(json.dumps(history, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _validate_cursor(proposal_cursor, accepted_updates, history, *, seed=None):
    if (type(proposal_cursor) is not int or not 0 <= proposal_cursor <= PROPOSALS
            or len(history) != proposal_cursor or type(accepted_updates) is not int
            or not 0 <= accepted_updates <= proposal_cursor):
        raise ValueError("SW0182 proposal cursor/accepted count/history mismatch")
    batches = _expected_batches()
    actual_accepted = 0
    for index, row in enumerate(history):
        if (row.get("proposal") != index + 1
                or row.get("image_indices") != batches[index]
                or type(row.get("accepted")) is not bool):
            raise ValueError("SW0182 proposal history differs from registered order")
        before = row.get("before_loss")
        after = row.get("after_loss")
        threshold = row.get("acceptance_threshold")
        if (not isinstance(before, (int, float)) or not math.isfinite(float(before))
                or not isinstance(threshold, (int, float))
                or float(threshold) != max(1e-7, 1e-6 * abs(float(before)))):
            raise ValueError("SW0182 proposal loss/threshold evidence is malformed")
        expected = (isinstance(after, (int, float)) and math.isfinite(float(after))
                    and float(after) < float(before) - float(threshold))
        if row["accepted"] is not bool(expected):
            raise ValueError("SW0182 accepted flag differs from registered actual-loss rule")
        actual_accepted += int(row["accepted"])
        if row.get("accepted_update_count") != actual_accepted:
            raise ValueError("SW0182 proposal history accepted-count cursor is inconsistent")
        if seed is not None and row.get("proposal_seed") != 171000 + int(seed) + index:
            raise ValueError("SW0182 proposal RNG seed differs from registered sequence")
    if actual_accepted != accepted_updates:
        raise ValueError("SW0182 accepted-update count differs from proposal history")


def _state_digest(modules, optimizer):
    digest = hashlib.sha256()

    def visit(prefix, value):
        if isinstance(value, torch.Tensor):
            digest.update(prefix.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(f"{prefix}/{key}", value[key])
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(f"{prefix}/{index}", child)
        else:
            digest.update(prefix.encode())
            digest.update(repr(value).encode())

    for name, module in modules.items():
        visit("module/" + name, module.state_dict())
    visit("optimizer", optimizer.state_dict())
    return digest.hexdigest()


def _full_batch_loss(arm, found, image_indices, inputs, device):
    """Full T=1024 no-grad objective, preserving all caller RNG streams."""
    flatten_targets = _module(
        "collaborative_test.SW_0178_temporal_class_competition.preflight").flatten_modal_targets
    util = _module(base.SOURCE_UTIL_MODULE)
    arrays = inputs["arrays"]
    losses = []
    for offset in range(0, base.BATCH, base.MICRO):
        indices = image_indices[offset:offset + base.MICRO]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][indices])).to(device)
        labels_batch = flatten_targets(arrays["modal_gt"][indices]).to(device)
        with torch.no_grad():
            _gamma, _trace, _adjacency, _z, q, rho = util._native_forward(
                found, rgb, live_tail_steps=0)
            losses.extend(float(base._loss_for_image(arm, q[i], rho[i], labels_batch[i]).cpu())
                          for i in range(base.MICRO))
        del rgb, labels_batch, q, rho
    if len(losses) != base.BATCH:
        raise ValueError("SW0182 full-forward batch loss is incomplete")
    if not all(math.isfinite(value) for value in losses):
        return math.nan
    return float(np.mean(losses))


def _measure_preserving_rng(arm, found, image_indices, inputs, device, *, evaluation_rng=None,
                            allow_numerical_rejection=False):
    rng = tx.capture_rng_state()
    if evaluation_rng is not None:
        tx.restore_rng_state(evaluation_rng)
    modules = {"wrapped": found.wrapped, "encoder": found.encoder}
    before_state = _state_digest(modules, found._sw0181_optimizer)
    try:
        try:
            value = _full_batch_loss(arm, found, image_indices, inputs, device)
        except FloatingPointError as exc:
            if not allow_numerical_rejection or "nonfinite" not in str(exc).lower():
                raise
            value = math.nan
    finally:
        tx.restore_rng_state(rng)
    if _state_digest(modules, found._sw0181_optimizer) != before_state:
        raise RuntimeError("SW0182 no-grad loss measurement mutated model or Adam state")
    return value


def _acceptance(before, after):
    if not math.isfinite(float(before)):
        raise FloatingPointError("SW0182 before loss is nonfinite")
    threshold = max(1e-7, 1e-6 * abs(float(before)))
    accepted = math.isfinite(float(after)) and float(after) < float(before) - threshold
    return bool(accepted), float(threshold)


def _commit_decision(snapshot, modules, optimizer, before, after, rng_after_gradient):
    accepted, threshold = _acceptance(before, after)
    if accepted:
        optimizer.zero_grad(set_to_none=True)
        tx.restore_rng_state(rng_after_gradient)
    else:
        tx.restore_transaction(snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
    return accepted, threshold


def _make_bindings(arm, seed, inputs, fresh, control):
    parent = fresh["bindings"]
    return {
        "experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
        "objective": "product_q_ce", "acceptance_rule": "strict actual full-forward descent",
        "factory_binding": parent,
        "source_binding": parent["source_binding"],
        "source_state_sha256": parent["source_state_sha256"],
        "base_encoder_sha256": parent["base_encoder_sha256"],
        "initial_model_state_sha256": parent["initial_model_state_sha256"],
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "ordered_train_ids": inputs["ordered_train_ids"],
        "parent_optimizer_groups": parent["optimizer_groups"],
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "control_experiment": "SW0181_actual_descent_qce",
        "control_manifest_sha256": control["manifest_sha256"],
        "control_history_sha256": control["history_sha256"],
        "control_score_sha256": control["score_sha256"],
        "control_score_path": str(control["score_path"]),
        "control_manifest_path": str(control["manifest_path"]),
        "control_history_path": str(control["history_path"]),
        "prefix_control_updates": 32,
        "training_fingerprint": implementation_fingerprint(),
    }


def _control_prefix(seed):
    """Validate and pin the completed SW0181 first-pass control history."""
    module = _module("collaborative_test.SW_0181_actual_descent_qce.train")
    manifest_path = CONTROL_ROOT / f"product_q_ce_seed{int(seed)}" / "manifest.json"
    history_path = manifest_path.parent / "history.json"
    if not manifest_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"missing completed SW0181 seed-{seed} control artifact")
    manifest = module.validate_training_artifact("product_q_ce", manifest_path.parent, seed=seed)
    history = json.loads(history_path.read_text(encoding="utf-8"))["proposals"]
    score_path = _select_control_score(CONTROL_SCORE_CANDIDATES, CONTROL_SCORE_SHA256)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if score.get("experiment") != "SW0181_actual_descent_qce_three_seed_score":
        raise ValueError("SW0181 comparison score has an unexpected experiment identity")
    if score.get("training_manifest_sha256", {}).get(str(seed)) != _sha_file(manifest_path):
        raise ValueError(f"SW0181 seed-{seed} score does not bind its actual training manifest")
    if len(history) != 64 or len(manifest.get("train_image_indices", [])) != 1024:
        raise ValueError("SW0181 control lacks its complete 64-proposal artifact")
    return {"manifest": manifest, "history": history,
            "manifest_path": manifest_path, "history_path": history_path,
            "manifest_sha256": _sha_file(manifest_path),
            "history_sha256": _sha_file(history_path), "score_sha256": CONTROL_SCORE_SHA256,
            "score_path": score_path}


def _validate_prefix_control_row(row, control_row, *, index):
    exact_fields = ("proposal", "proposal_seed", "image_indices", "accepted",
                    "accepted_update_count", "start_transaction_sha256",
                    "proposal_transaction_sha256")
    for key in exact_fields:
        if row.get(key) != control_row.get(key):
            raise ValueError(f"SW0182 first32 control mismatch at proposal {index + 1}: {key}")
    for key in ("before_loss", "after_loss", "acceptance_threshold"):
        left, right = row.get(key), control_row.get(key)
        if (left is None) != (right is None):
            raise ValueError(f"SW0182 first32 control mismatch at proposal {index + 1}: {key}")
        if left is not None:
            tolerance = max(1e-7, 1e-6 * abs(float(right)))
            if not math.isfinite(float(left)) or abs(float(left) - float(right)) >= tolerance:
                raise ValueError(f"SW0182 first32 control numeric mismatch at proposal {index + 1}: {key}")


def _select_control_score(candidates, expected_sha256, *, is_file=None, sha_file=None):
    """Select only a registered score location with the exact pinned bytes."""
    is_file = Path.is_file if is_file is None else is_file
    sha_file = _sha_file if sha_file is None else sha_file
    matches = [Path(path) for path in candidates
               if is_file(Path(path)) and sha_file(Path(path)) == expected_sha256]
    if len(matches) != 1:
        raise ValueError("SW0181 pinned three-seed score is missing, changed, or ambiguous")
    return matches[0]


def _artifact_paths(output_root, arm, seed):
    label = f"{arm}_seed{int(seed)}"
    return Path(output_root) / "work" / label, Path(output_root) / label


def _immutable_input_hashes(inputs, binding):
    cache_root = Path(inputs["dataset_prepared"]["cache_root"])
    source = binding["source_binding"]
    return {
        "dataset_manifest": _sha_file(cache_root / "manifest.json"),
        "dataset_arrays": _sha_file(cache_root / "arrays.npz"),
        "source_checkpoint": _sha_file(Path(source["source_core_path"])),
        "source_manifest": _sha_file(Path(source["source_manifest_path"])),
    }


def _atomic_json_create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _credit_rejection_record(arm, seed, binding, cursor, history, failures):
    return {"experiment": EXPERIMENT, "status": "scientific_rejection",
        "arm": arm, "seed": int(seed), "proposal_cursor": int(cursor),
        "accepted_updates": sum(bool(row["accepted"]) for row in history),
        "history_sha256": _history_sha(history), "bindings": binding,
        "rejection_reasons": list(failures), "optimizer_updates": sum(bool(row["accepted"]) for row in history),
        "validation_data_read": False}


def train(arm, *, seed, physical_gpu, output_root=OUTPUT_ROOT):
    seed = int(seed)
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("SW0182 received an unregistered arm or source seed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0182 worker visibility differs from leased physical GPU")
    inputs = prepare_training_inputs_cpu()
    fresh_cpu = prepare_fresh_training_state_cpu(seed, inputs)
    cpu_initial_state_sha = fresh_cpu["proof"]["initial_model_state_sha256"]
    control = _control_prefix(seed)
    if cpu_initial_state_sha != control["manifest"]["initial_model_state_sha256"]:
        raise ValueError("SW0182 fresh seed initialization differs from completed SW0181 control")
    del fresh_cpu
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0182 CUDA initialized before source/data/control admission")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    found, optimizer, families, parent_binding, setup = base._make_fresh(
        "product_q_ce", seed, inputs, device)
    found._sw0181_optimizer = optimizer
    source_cal, source_utils, base_encoder, frozen_before, _dataset = setup
    binding = _make_bindings(arm, seed, inputs,
                             {"bindings": parent_binding}, control)
    if binding["initial_model_state_sha256"] != cpu_initial_state_sha:
        raise ValueError("SW0182 initialized model differs from CPU fresh factory")
    runtime_module = _module(base.CHECKPOINT_MODULE)
    runtime_base = runtime_module.runtime_signature(device)
    work, output = _artifact_paths(output_root, arm, seed)
    if output.exists():
        return validate_training_artifact(arm, output, seed=seed, expected_bindings=binding)
    if work.exists() and (work / "boundary.ptx").is_file():
        metadata = runtime_module.inspect_committed_metadata(work)
        saved_runtime = metadata.get("runtime")
        if (not isinstance(saved_runtime, dict)
                or {k: v for k, v in saved_runtime.items() if k not in {
                    "sw0182_proposal_cursor_semantics", "proposal_cursor", "accepted_updates"}}
                   != runtime_base
                or saved_runtime.get("sw0182_proposal_cursor_semantics") is not True
                or saved_runtime.get("proposal_cursor") != metadata.get("completed_updates")):
            raise ValueError("SW0182 boundary runtime/cursor binding mismatch")
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=binding, expected_runtime=saved_runtime,
            device=device, max_updates=PROPOSALS)
        cursor, accepted_updates, history = loaded["completed_updates"], None, loaded["history"]
        accepted_updates = sum(bool(row["accepted"]) for row in history)
        if saved_runtime.get("accepted_updates") != accepted_updates:
            raise ValueError("SW0182 boundary accepted count differs from committed proposal history")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving unexpected partial SW0182 work directory")
        cursor, accepted_updates, history = 0, 0, []
        work.mkdir(parents=True, exist_ok=True)
        _save_boundary(runtime_module, work, cursor, found, optimizer, history,
                       binding, runtime_base, device, accepted_updates)
    _validate_cursor(cursor, accepted_updates, history, seed=seed)
    for index, row in enumerate(history[:32]):
        _validate_prefix_control_row(row, control["history"][index], index=index)
    rejection_path = work / "credit_rejection.json"
    if cursor >= 1 and len(history) >= 1:
        failures = []
        for row in history[:2]:
            failures.extend(base._credit_failures(row))
        if failures:
            if rejection_path.is_file():
                saved = json.loads(rejection_path.read_text(encoding="utf-8"))
                if (saved.get("bindings") != binding or saved.get("proposal_cursor") > cursor
                        or saved.get("history_sha256") != _history_sha(history)):
                    raise ValueError("SW0182 credit rejection receipt does not match committed history")
                return saved
            saved = _credit_rejection_record(arm, seed, binding, cursor, history, failures)
            _atomic_json_create(rejection_path, saved)
            return saved
    found.encoder.train(); found.encoder.base_encoder.eval()
    found.patcher.eval(); found.wrapped.train()
    cache_hashes_before = _immutable_input_hashes(inputs, binding)
    expected = _expected_batches()
    for proposal_index in range(cursor, PROPOSALS):
        image_indices = np.asarray(expected[proposal_index], dtype=np.int64)
        proposal_seed = 171000 + seed + proposal_index
        torch.manual_seed(proposal_seed)
        torch.cuda.manual_seed(proposal_seed)
        optimizer.zero_grad(set_to_none=True)
        modules = {"wrapped": found.wrapped, "encoder": found.encoder}
        snapshot = tx.capture_transaction(modules, optimizer, offload_to_cpu=True)
        start_state_sha = tx.transaction_digest(snapshot)
        before_full = None
        if proposal_index < 2:
            before_full = _measure_preserving_rng(
                arm, found, image_indices, inputs, device, evaluation_rng=snapshot["rng"])
            tx.restore_rng_state(snapshot["rng"])
        row = base._train_update("product_q_ce", found, optimizer, image_indices,
                                 inputs, device, proposal_index + 1)
        before = float(row["mean_registered_loss"])
        parity_tolerance = max(1e-7, 1e-6 * abs(before))
        if before_full is not None:
            parity_error = abs(before - before_full) if math.isfinite(before_full) else math.inf
            if not math.isfinite(before_full) or parity_error >= parity_tolerance:
                tx.restore_transaction(snapshot, modules, optimizer, restore_rng=True, clear_grad=True)
                raise RuntimeError("SW0182 helper before-loss differs from same-RNG full-forward value")
            row["before_loss_parity_abs_error"] = parity_error
        rng_after_gradient = tx.capture_rng_state()
        proposal_sha = _state_digest(modules, optimizer)
        after = _measure_preserving_rng(arm, found, image_indices, inputs, device,
                                        evaluation_rng=snapshot["rng"],
                                        allow_numerical_rejection=True)
        if _state_digest(modules, optimizer) != proposal_sha:
            raise RuntimeError("SW0182 post-loss measurement mutated proposed model/Adam state")
        accepted, threshold = _commit_decision(snapshot, modules, optimizer, before, after,
                                               rng_after_gradient)
        row.update({"proposal": proposal_index + 1, "proposal_seed": proposal_seed,
            "accepted": accepted, "before_loss": before,
            "before_full_nograd_loss": before_full,
            "after_loss": after if math.isfinite(float(after)) else None,
            "acceptance_threshold": threshold,
            "proposal_transaction_sha256": proposal_sha,
            "start_transaction_sha256": start_state_sha,
            "proposal_parameter_delta_l2": row["family_parameter_delta_l2"],
            "committed_parameter_delta_l2": (row["family_parameter_delta_l2"] if accepted
                                                else {name: 0.0 for name in row["family_parameter_delta_l2"]}),
            "accepted_update_count": accepted_updates + int(accepted),
            "proposal_cursor": proposal_index + 1})
        row["pass"] = 1
        row["data_block"] = proposal_index // 32
        if proposal_index < 32:
            _validate_prefix_control_row(row, control["history"][proposal_index],
                                         index=proposal_index)
        if proposal_index < 2:
            row["first_two_credit_failures"] = base._credit_failures(row)
        if not accepted and not math.isfinite(float(after)):
            row["rejection_reason"] = "nonfinite full-forward post-loss"
        elif not accepted:
            row["rejection_reason"] = "post-loss did not clear registered descent threshold"
        history.append(row)
        accepted_updates += int(accepted)
        _save_boundary(runtime_module, work, proposal_index + 1, found, optimizer,
                       history, binding, runtime_base, device, accepted_updates)
        if proposal_index < 2:
            failures = base._credit_failures(row)
            if failures:
                saved = _credit_rejection_record(arm, seed, binding,
                    proposal_index + 1, history, failures)
                _atomic_json_create(rejection_path, saved)
                return saved
    cache_hashes_after = _immutable_input_hashes(inputs, binding)
    if cache_hashes_after != cache_hashes_before:
        raise RuntimeError("SW0182 TRAIN/source cache files changed during proposals")
    frozen_after = source_utils._frozen_source_components(found, base_encoder)
    if frozen_after != frozen_before:
        raise ValueError("SW0182 frozen SourceT/context foundation components changed")
    return _publish_training(arm, seed, output, work, found, optimizer, history,
                             binding, runtime_base, inputs, accepted_updates, frozen_before,
                             source_utils, base_encoder, cache_hashes_before)


def _save_boundary(runtime_module, work, cursor, found, optimizer, history,
                   bindings, runtime, device, accepted_updates):
    # SW0148's completed_updates field is deliberately the proposal cursor here;
    # the binding and history carry accepted optimizer steps independently.
    runtime = {**runtime, "sw0182_proposal_cursor_semantics": True,
        "proposal_cursor": int(cursor), "accepted_updates": int(accepted_updates)}
    runtime_module.save_boundary(work, completed_updates=cursor, wrapped=found.wrapped,
        encoder=found.encoder, optimizer=optimizer, history=history, bindings=bindings,
        runtime=runtime, device=device)


def _publish_training(arm, seed, output, work, found, optimizer, history, bindings,
                      runtime, inputs, accepted_updates, frozen_before, source_utils,
                      base_encoder, cache_hashes):
    if output.exists():
        raise FileExistsError("preserving existing SW0182 output artifact")
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
            "bindings": bindings,
            "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
            "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
        checkpoint["model_state_sha256"] = base._state_sha(found.wrapped, found.encoder)
        torch.save(checkpoint, stage / "checkpoint.pt")
        torch.save(optimizer.state_dict(), stage / "optimizer.pt")
        shutil.copyfile(work / "boundary.ptx", stage / "boundary.ptx")
        with (stage / "history.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"proposals": history}, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        manifest = {"experiment": EXPERIMENT, "status": "training_complete",
            "arm": arm, "seed": int(seed), "proposal_budget": PROPOSALS,
            "proposal_cursor": PROPOSALS, "completed_proposals": PROPOSALS,
            "accepted_updates": int(accepted_updates),
            "optimizer_steps": int(accepted_updates), "batch": base.BATCH,
            "microbatch": MICRO, "passes": PASSES,
            "training_images_per_pass": COUNT,
            "proposal_exposures": PROPOSALS * base.BATCH,
            "train_image_indices": [i for batch in _expected_batches() for i in batch],
            "train_ids": [int(inputs["ordered_train_ids"][i])
                          for batch in _expected_batches() for i in batch],
            "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
            "validation_data_read": False, "ground_truth_scope": "TRAIN modal labels only",
            "bindings": bindings, "runtime": runtime,
            "initial_model_state_sha256": bindings["initial_model_state_sha256"],
            "final_model_state_sha256": checkpoint["model_state_sha256"],
            "optimizer_step_summary": source_utils._optimizer_summary(optimizer),
            "dataset_file_sha256": cache_hashes,
            "history_sha256": _sha_file(stage / "history.json"),
            "checkpoint_sha256": _sha_file(stage / "checkpoint.pt"),
            "optimizer_sha256": _sha_file(stage / "optimizer.pt"),
            "boundary_sha256": _sha_file(stage / "boundary.ptx"),
            "training_fingerprint": bindings["training_fingerprint"],
            "frozen_source_components_before": frozen_before,
            "frozen_source_components_after": source_utils._frozen_source_components(found, base_encoder)}
        with (stage / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        validate_training_artifact(arm, stage, seed=seed, expected_bindings=bindings)
        os.rename(stage, output)
    except BaseException:
        raise
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def validate_training_artifact(arm, root, *, seed=None, expected_bindings=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "training_complete"
            or manifest.get("arm") != arm or manifest.get("proposal_cursor") != PROPOSALS
            or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0182 artifact is not a complete TRAIN-only proposal run")
    if seed is not None and manifest.get("seed") != int(seed):
        raise ValueError("SW0182 artifact source seed mismatch")
    bindings = manifest.get("bindings")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("SW0182 artifact bindings mismatch")
    if (not isinstance(bindings, dict) or bindings.get("training_fingerprint") != implementation_fingerprint()
            or manifest.get("training_fingerprint") != bindings.get("training_fingerprint")):
        raise ValueError("SW0182 artifact fingerprint mismatch")
    for filename, key in (("history.json", "history_sha256"), ("checkpoint.pt", "checkpoint_sha256"),
                          ("optimizer.pt", "optimizer_sha256"), ("boundary.ptx", "boundary_sha256")):
        if _sha_file(root / filename) != manifest.get(key):
            raise ValueError(f"SW0182 artifact digest mismatch: {filename}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))["proposals"]
    accepted = sum(bool(row.get("accepted")) for row in history)
    _validate_cursor(PROPOSALS, accepted, history, seed=int(manifest["seed"]))
    if accepted != manifest.get("accepted_updates") or accepted != manifest.get("optimizer_steps"):
        raise ValueError("SW0182 accepted proposal and Adam step counts differ")
    ordered_ids = bindings.get("ordered_train_ids")
    batches = _expected_batches()
    if (not isinstance(ordered_ids, list) or len(ordered_ids) != COUNT
            or manifest.get("train_image_indices") != [i for batch in batches for i in batch]
            or manifest.get("train_ids") != [int(ordered_ids[i]) for batch in batches for i in batch]
            or [row.get("image_ids") for row in history]
               != [[int(ordered_ids[i]) for i in batch] for batch in batches]):
        raise ValueError("SW0182 artifact TRAIN IDs/order do not match bound exposures")
    for row in history[:2]:
        if row.get("first_two_credit_failures") or base._credit_failures(row):
            raise ValueError("SW0182 first-two-proposal credit validation failed")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("bindings") != bindings
            or base._state_sha_from_payload(checkpoint) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0182 model payload digest mismatch")
    optimizer_state = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    util = _module(base.SOURCE_UTIL_MODULE)
    util._validate_optimizer_state(optimizer_state, manifest.get("optimizer_step_summary"))
    parameter_steps = [int(state["step"].item()) for state in optimizer_state["state"].values()]
    if ((accepted == 0 and parameter_steps)
            or any(step != accepted for step in parameter_steps)):
        raise ValueError("SW0182 Adam per-parameter steps differ from accepted proposal count")
    return manifest

