"""SW0181 bounded actual-descent proposals for the supervised product-QCE model."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0181_actual_descent_qce"
ARMS = ("product_q_ce",)
SEEDS = (0, 1, 2)
PROPOSALS = 64
OUTPUT_ROOT = HERE / "results_archive" / "training_20261016"
PREFLIGHT_MODULE = "collaborative_test.SW_0177_learned_coupling_polarity.preflight"
QUEUE_MODULE = "collaborative_test.SW_0181_actual_descent_qce.training_queue"


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
        "collaborative_test/SW_0181_actual_descent_qce/DESIGN_KO.md",
        "collaborative_test/SW_0181_actual_descent_qce/state_transaction.py",
        "collaborative_test/SW_0181_actual_descent_qce/test_state_transaction.py",
        "collaborative_test/SW_0181_actual_descent_qce/train.py",
        "collaborative_test/SW_0181_actual_descent_qce/test_train.py",
        "collaborative_test/SW_0181_actual_descent_qce/training_queue.py",
        "collaborative_test/SW_0181_actual_descent_qce/test_training_queue.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/DESIGN_KO.md",
        "collaborative_test/SW_0177_learned_coupling_polarity/preflight.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/resource_queue_v2.py",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md",
        "collaborative_test/SW_0178_temporal_class_competition/class_competition.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
    )
    files = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0181 fingerprint dependency missing: {relative}")
        files[relative] = _sha_file(path)
    return {
        "experiment": EXPERIMENT,
        "files": files,
        "parent_training_fingerprint": base.fingerprint_sha256(),
        "source_preflight_module": PREFLIGHT_MODULE,
        "seeds": list(SEEDS),
        "proposal_budget": PROPOSALS,
        "acceptance": "finite post < before - max(1e-7,1e-6*abs(before))",
        "gradient_mode": "original SW0179 full-1024 forward; live-tail64 backward",
        "mask_supervised": True,
        "unsupervised_leaderboard_eligible": False,
    }


def fingerprint_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def prepare_training_inputs_cpu():
    inputs = base.prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0181 TRAIN cache must be admitted before CUDA initialization")
    return inputs


def prepare_fresh_training_state_cpu(seed, inputs=None):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0181 accepts registered source seeds 0, 1, or 2")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0181 fresh source factory must run before CUDA initialization")
    inputs = prepare_training_inputs_cpu() if inputs is None else inputs
    proof = base.prepare_fresh_training_state_cpu(seed, inputs)
    if proof["proof"]["adam_state_entries"] != 0:
        raise ValueError("SW0181 fresh Adam is not empty")
    proof["proof"]["experiment"] = EXPERIMENT
    proof["proof"]["accept_reject_protocol"] = "one proposal per fixed batch; no retries"
    proof["proof"]["training_fingerprint"] = implementation_fingerprint()
    return proof


def _expected_batches():
    return base._expected_batches()


def _history_sha(history):
    return _sha_bytes(json.dumps(history, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _validate_cursor(proposal_cursor, accepted_updates, history, *, seed=None):
    if (type(proposal_cursor) is not int or not 0 <= proposal_cursor <= PROPOSALS
            or len(history) != proposal_cursor or type(accepted_updates) is not int
            or not 0 <= accepted_updates <= proposal_cursor):
        raise ValueError("SW0181 proposal cursor/accepted count/history mismatch")
    batches = _expected_batches()
    actual_accepted = 0
    for index, row in enumerate(history):
        if (row.get("proposal") != index + 1
                or row.get("image_indices") != batches[index]
                or type(row.get("accepted")) is not bool):
            raise ValueError("SW0181 proposal history differs from registered order")
        before = row.get("before_loss")
        after = row.get("after_loss")
        threshold = row.get("acceptance_threshold")
        if (not isinstance(before, (int, float)) or not math.isfinite(float(before))
                or not isinstance(threshold, (int, float))
                or float(threshold) != max(1e-7, 1e-6 * abs(float(before)))):
            raise ValueError("SW0181 proposal loss/threshold evidence is malformed")
        expected = (isinstance(after, (int, float)) and math.isfinite(float(after))
                    and float(after) < float(before) - float(threshold))
        if row["accepted"] is not bool(expected):
            raise ValueError("SW0181 accepted flag differs from registered actual-loss rule")
        actual_accepted += int(row["accepted"])
        if row.get("accepted_update_count") != actual_accepted:
            raise ValueError("SW0181 proposal history accepted-count cursor is inconsistent")
        if seed is not None and row.get("proposal_seed") != 171000 + int(seed) + index:
            raise ValueError("SW0181 proposal RNG seed differs from registered sequence")
    if actual_accepted != accepted_updates:
        raise ValueError("SW0181 accepted-update count differs from proposal history")


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
        raise ValueError("SW0181 full-forward batch loss is incomplete")
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
        raise RuntimeError("SW0181 no-grad loss measurement mutated model or Adam state")
    return value


def _acceptance(before, after):
    if not math.isfinite(float(before)):
        raise FloatingPointError("SW0181 before loss is nonfinite")
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


def _make_bindings(arm, seed, inputs, fresh, preflight_sha, preflight_queue_sha):
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
        "preflight_report_sha256": preflight_sha,
        "preflight_queue_fingerprint_sha256": preflight_queue_sha,
        "training_fingerprint": implementation_fingerprint(),
    }


def _preflight_receipt(seed):
    queue = _module("collaborative_test.SW_0177_learned_coupling_polarity.resource_queue_v2")
    state_path, report_path = queue.STATE, queue.REPORT
    if not state_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("SW0181 requires a completed SW0177 source-preflight receipt")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = _sha_file(report_path)
    if (state.get("status") != "complete"
            or state.get("report_file_sha256") != report_sha
            or state.get("report_status") != "complete_source_preflight"
            or report.get("status") != "complete_source_preflight"
            or state.get("implementation_fingerprint") != report.get("implementation_fingerprint")
            or report.get("optimizer_updates") != 0
            or report.get("validation_data_read") is not False):
        raise ValueError("SW0181 SW0177 preflight is not a successful immutable zero-update receipt")
    preflight = _module(PREFLIGHT_MODULE)
    queue_sha = queue.fingerprint_sha256()
    current_preflight = preflight.implementation_fingerprint()
    if (state.get("queue_fingerprint_sha256") != queue_sha
            or state.get("implementation_fingerprint") != current_preflight
            or report.get("implementation_fingerprint") != current_preflight):
        raise ValueError("SW0181 SW0177 preflight source or queue fingerprint changed")
    return {"status": "complete", "report_sha256": report_sha,
            "queue_state_sha256": _sha_file(state_path),
            "queue_fingerprint_sha256": queue_sha,
            "preflight_fingerprint": state.get("implementation_fingerprint"),
            "report": report}


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
        raise ValueError("SW0181 received an unregistered arm or source seed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0181 worker visibility differs from leased physical GPU")
    inputs = prepare_training_inputs_cpu()
    fresh_cpu = prepare_fresh_training_state_cpu(seed, inputs)
    cpu_initial_state_sha = fresh_cpu["proof"]["initial_model_state_sha256"]
    receipt = _preflight_receipt(seed)
    preflight_sha = receipt["report_sha256"]
    del fresh_cpu
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0181 CUDA initialized before source/data admission")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    found, optimizer, families, parent_binding, setup = base._make_fresh(
        "product_q_ce", seed, inputs, device)
    found._sw0181_optimizer = optimizer
    source_cal, source_utils, base_encoder, frozen_before, _dataset = setup
    binding = _make_bindings(arm, seed, inputs,
                             {"bindings": parent_binding}, preflight_sha,
                             receipt["queue_fingerprint_sha256"])
    if binding["initial_model_state_sha256"] != cpu_initial_state_sha:
        raise ValueError("SW0181 initialized model differs from successful preflight")
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
                    "sw0181_proposal_cursor_semantics", "proposal_cursor", "accepted_updates"}}
                   != runtime_base
                or saved_runtime.get("sw0181_proposal_cursor_semantics") is not True
                or saved_runtime.get("proposal_cursor") != metadata.get("completed_updates")):
            raise ValueError("SW0181 boundary runtime/cursor binding mismatch")
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=binding, expected_runtime=saved_runtime,
            device=device, max_updates=PROPOSALS)
        cursor, accepted_updates, history = loaded["completed_updates"], None, loaded["history"]
        accepted_updates = sum(bool(row["accepted"]) for row in history)
        if saved_runtime.get("accepted_updates") != accepted_updates:
            raise ValueError("SW0181 boundary accepted count differs from committed proposal history")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving unexpected partial SW0181 work directory")
        cursor, accepted_updates, history = 0, 0, []
        work.mkdir(parents=True, exist_ok=True)
        _save_boundary(runtime_module, work, cursor, found, optimizer, history,
                       binding, runtime_base, device, accepted_updates)
    _validate_cursor(cursor, accepted_updates, history, seed=seed)
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
                    raise ValueError("SW0181 credit rejection receipt does not match committed history")
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
                raise RuntimeError("SW0181 helper before-loss differs from same-RNG full-forward value")
            row["before_loss_parity_abs_error"] = parity_error
        rng_after_gradient = tx.capture_rng_state()
        proposal_sha = _state_digest(modules, optimizer)
        after = _measure_preserving_rng(arm, found, image_indices, inputs, device,
                                        evaluation_rng=snapshot["rng"],
                                        allow_numerical_rejection=True)
        if _state_digest(modules, optimizer) != proposal_sha:
            raise RuntimeError("SW0181 post-loss measurement mutated proposed model/Adam state")
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
        row["pass"] = proposal_index // 32 + 1
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
        raise RuntimeError("SW0181 TRAIN/source cache files changed during proposals")
    frozen_after = source_utils._frozen_source_components(found, base_encoder)
    if frozen_after != frozen_before:
        raise ValueError("SW0181 frozen SourceT/context foundation components changed")
    return _publish_training(arm, seed, output, work, found, optimizer, history,
                             binding, runtime_base, inputs, accepted_updates, frozen_before,
                             source_utils, base_encoder, cache_hashes_before)


def _save_boundary(runtime_module, work, cursor, found, optimizer, history,
                   bindings, runtime, device, accepted_updates):
    # SW0148's completed_updates field is deliberately the proposal cursor here;
    # the binding and history carry accepted optimizer steps independently.
    runtime = {**runtime, "sw0181_proposal_cursor_semantics": True,
        "proposal_cursor": int(cursor), "accepted_updates": int(accepted_updates)}
    runtime_module.save_boundary(work, completed_updates=cursor, wrapped=found.wrapped,
        encoder=found.encoder, optimizer=optimizer, history=history, bindings=bindings,
        runtime=runtime, device=device)


def _publish_training(arm, seed, output, work, found, optimizer, history, bindings,
                      runtime, inputs, accepted_updates, frozen_before, source_utils,
                      base_encoder, cache_hashes):
    if output.exists():
        raise FileExistsError("preserving existing SW0181 output artifact")
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
            "microbatch": base.MICRO, "passes": base.PASSES,
            "training_images_per_pass": base.COUNT,
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
        raise ValueError("SW0181 artifact is not a complete TRAIN-only proposal run")
    if seed is not None and manifest.get("seed") != int(seed):
        raise ValueError("SW0181 artifact source seed mismatch")
    bindings = manifest.get("bindings")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("SW0181 artifact bindings mismatch")
    if (not isinstance(bindings, dict) or bindings.get("training_fingerprint") != implementation_fingerprint()
            or manifest.get("training_fingerprint") != bindings.get("training_fingerprint")):
        raise ValueError("SW0181 artifact fingerprint mismatch")
    for filename, key in (("history.json", "history_sha256"), ("checkpoint.pt", "checkpoint_sha256"),
                          ("optimizer.pt", "optimizer_sha256"), ("boundary.ptx", "boundary_sha256")):
        if _sha_file(root / filename) != manifest.get(key):
            raise ValueError(f"SW0181 artifact digest mismatch: {filename}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))["proposals"]
    accepted = sum(bool(row.get("accepted")) for row in history)
    _validate_cursor(PROPOSALS, accepted, history, seed=int(manifest["seed"]))
    if accepted != manifest.get("accepted_updates") or accepted != manifest.get("optimizer_steps"):
        raise ValueError("SW0181 accepted proposal and Adam step counts differ")
    ordered_ids = bindings.get("ordered_train_ids")
    batches = _expected_batches()
    if (not isinstance(ordered_ids, list) or len(ordered_ids) != base.COUNT
            or manifest.get("train_image_indices") != [i for batch in batches for i in batch]
            or manifest.get("train_ids") != [int(ordered_ids[i]) for batch in batches for i in batch]
            or [row.get("image_ids") for row in history]
               != [[int(ordered_ids[i]) for i in batch] for batch in batches]):
        raise ValueError("SW0181 artifact TRAIN IDs/order do not match bound exposures")
    for row in history[:2]:
        if row.get("first_two_credit_failures") or base._credit_failures(row):
            raise ValueError("SW0181 first-two-proposal credit validation failed")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("bindings") != bindings
            or base._state_sha_from_payload(checkpoint) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0181 model payload digest mismatch")
    optimizer_state = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    util = _module(base.SOURCE_UTIL_MODULE)
    util._validate_optimizer_state(optimizer_state, manifest.get("optimizer_step_summary"))
    parameter_steps = [int(state["step"].item()) for state in optimizer_state["state"].values()]
    if ((accepted == 0 and parameter_steps)
            or any(step != accepted for step in parameter_steps)):
        raise ValueError("SW0181 Adam per-parameter steps differ from accepted proposal count")
    return manifest

