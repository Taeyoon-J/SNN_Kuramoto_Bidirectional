"""Checkpointable paired SW0177 training over the frozen SourceT input cache."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np  # Initialize NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0177_learned_coupling_polarity import preflight
from collaborative_test.SW_0177_learned_coupling_polarity.polarity_operator import (
    install_polarity_prepare, seeded_edge_multiplier,
)

EXPERIMENT = "SW0177_learned_coupling_polarity_training"
ARMS = ("signed", "nonnegative")
TRAIN_COUNT, PASSES, UPDATES = 512, 2, 64
BATCH, MICRO = 16, 4
SEED = 177001
OUTPUT_ROOT = preflight.HERE / "results_archive" / "training_v1_20261016"
HERE, ROOT = preflight.HERE, preflight.ROOT
LEGACY = "collaborative_test.SW_0175_critical_component_binding.train_v2"
PREFLIGHT_QUEUE = "collaborative_test.SW_0177_learned_coupling_polarity.resource_queue_v2"
CHECKPOINT = "collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name):
    import importlib
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _legacy():
    return _module(LEGACY)


def _preflight_receipt():
    queue = _module(PREFLIGHT_QUEUE)
    state_path, report_path = queue.STATE, queue.REPORT
    if not state_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("SW0177 requires a completed v2 source-preflight receipt")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    report_bytes = report_path.read_bytes()
    report_sha = _sha_bytes(report_bytes)
    if state.get("status") != "complete":
        raise ValueError(f"SW0177 requires successful source_preflight_v2, got {state.get('status')!r}")
    if state.get("report_file_sha256") != report_sha:
        raise ValueError("SW0177 v2 source-preflight report hash differs from terminal state")
    queue_fp = queue.fingerprint_sha256()
    if state.get("queue_fingerprint_sha256") != queue_fp:
        raise ValueError("SW0177 v2 queue fingerprint differs from terminal state")
    report = json.loads(report_bytes)
    prep_fp = preflight.implementation_fingerprint()
    if state.get("implementation_fingerprint") != prep_fp:
        raise ValueError("SW0177 original preflight fingerprint differs from v2 terminal state")
    proof = state.get("source_proof")
    terminal = queue._validate_worker_report(report, preflight_fingerprint=prep_fp,
        source_proof=proof, physical_gpu=state.get("physical_gpu"))
    if terminal != "complete" or report.get("status") != "complete_source_preflight":
        raise ValueError("SW0177 scientific rejection cannot admit training")
    return {"state": state, "report": report, "report_sha256": report_sha,
            "queue_fingerprint_sha256": queue_fp, "preflight_fingerprint": prep_fp,
            "source_proof": proof}


def _load_admitted_inputs():
    """Require v2 preflight success, then load the exact SW0175 TRAIN cache."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 CPU admission must precede CUDA initialization")
    receipt = _preflight_receipt()
    prepared = preflight.prepare_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 source/cache preparation unexpectedly initialized CUDA")
    if prepared.get("source_proof") != receipt["source_proof"]:
        raise ValueError("SW0177 preflight and fresh CPU source proof differ")
    if prepared.get("implementation_fingerprint") != receipt["preflight_fingerprint"]:
        raise ValueError("SW0177 preflight source fingerprint changed after admission")
    arrays = prepared["arrays"]
    proof = prepared["source_proof"]
    admitted = prepared["admitted"]
    cache_proof = admitted["prepared"]["cache_proof"]
    if (len(arrays["image_ids"]) != TRAIN_COUNT
            or not np.array_equal(np.asarray(arrays["image_ids"], dtype=np.int64),
                                  np.asarray(cache_proof["image_ids"], dtype=np.int64))):
        raise ValueError("SW0177 ordered TRAIN IDs differ from admitted input cache")
    return {"prepared": prepared, "proof": proof, "v2_receipt": receipt,
            "cache_root": Path(admitted["prepared"]["cache_root"])}


def _training_fingerprint():
    paths = (
        "collaborative_test/SW_0177_learned_coupling_polarity/DESIGN_KO.md",
        "collaborative_test/SW_0177_learned_coupling_polarity/polarity_operator.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/preflight.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/resource_queue_v2.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/train.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/training_queue.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/test_train.py",
        "collaborative_test/SW_0177_learned_coupling_polarity/test_training_queue.py",
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
    )
    files = {rel: _sha_file(ROOT / rel) for rel in paths}
    return {"experiment": EXPERIMENT, "files": files,
        "preflight_fingerprint": preflight.implementation_fingerprint(),
        "v2_queue_fingerprint_sha256": _module(PREFLIGHT_QUEUE).fingerprint_sha256(),
        "arms": list(ARMS), "seed": SEED, "updates": UPDATES, "passes": PASSES,
        "training_images_per_pass": TRAIN_COUNT, "logical_batch": BATCH,
        "microbatch": MICRO, "optimizer": "Adam", "global_clip_norm": 1.0,
        "learning_rates": {"context_encoder": 3e-4, "graph": 3e-5,
            "native_core": 3e-5, "integration": 1e-3, "polarity_head": 1e-3}}


def _fingerprint_sha():
    return _sha_bytes(json.dumps(_training_fingerprint(), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _expected_batches():
    base = _legacy()
    return base._expected_batches()


def _state_sha(found):
    base = _legacy()
    return base._state_sha(found.wrapped, found.encoder)


def _make_optimizer(found, head):
    base = _legacy()
    optimizer, families = base._make_optimizer(found)
    # Register the head under the encoder container only after the established
    # source families have been selected, preserving disjoint family ownership.
    if hasattr(found.encoder, "polarity_head"):
        raise ValueError("SW0177 polarity head already registered on this fresh source")
    found.encoder.add_module("polarity_head", head)
    entries = list(head.named_parameters())
    if not entries or any(not parameter.requires_grad for _, parameter in entries):
        raise ValueError("SW0177 polarity head has no trainable parameters")
    optimizer.add_param_group({"params": [p for _, p in entries],
        "parameter_names": [name for name, _ in entries], "name": "polarity_head",
        "lr": 1e-3})
    families["polarity_head"] = entries
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("SW0177 optimizer families overlap")
    return optimizer, families


def _optimizer_contract(optimizer):
    """JSON-safe exact ordered optimizer groups used by the paired arms."""
    result = []
    for group in optimizer.param_groups:
        row = {"name": group.get("name"),
            "parameter_names": list(group.get("parameter_names", [])),
            "parameter_shapes": [list(p.shape) for p in group["params"]],
            "parameter_dtypes": [str(p.dtype) for p in group["params"]]}
        for key in ("lr", "eps", "weight_decay"):
            row[key] = float(group[key])
        row["betas"] = [float(x) for x in group["betas"]]
        for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
            row[key] = group.get(key)
        result.append(row)
    return result


def _build_training_source(admitted, device):
    """Verify the CPU factory, then construct the GPU model through its factory."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 source factory admission must begin before CUDA initialization")
    report = admitted["v2_receipt"]["report"]
    prepared = admitted["prepared"]
    cpu_found, cpu_base, cpu_source_sha, cpu_state_sha = preflight.build_source(prepared, "cpu")
    if cpu_state_sha != report.get("fresh_initial_model_state_sha256"):
        raise ValueError("SW0177 CPU source/context initialization differs from v2 preflight")
    source_sha = _legacy()._source_sha(cpu_found, cpu_base)
    module_api = preflight._module("collaborative_test.SW_0171_contextual_instance_graph.preflight")
    cpu_base_sha = module_api._module_sha(cpu_base)
    torch.cuda.set_device(torch.device(device))
    gpu_found, gpu_base, gpu_source_sha, gpu_state_sha = preflight._build_source(prepared, device)
    if (gpu_source_sha != cpu_source_sha or gpu_state_sha != cpu_state_sha
            or module_api._module_sha(gpu_base) != cpu_base_sha
            or _legacy()._source_sha(gpu_found, gpu_base) != source_sha):
        raise ValueError("SW0177 GPU source factory differs from validated fresh CPU initialization")
    return gpu_found, gpu_base, gpu_source_sha, gpu_state_sha


def _native_forward(found, head, rgb, *, live_tail_steps):
    graph = found.wrapped.core.graph_generator
    kuramoto = found.wrapped.core.kuramoto
    head_capture, negative = {}, {"sum_sq": 0.0, "frames": 0}

    def capture_head(_module, inputs, output):
        if len(inputs) != 2:
            raise ValueError("SW0177 head must receive actual projection and structural graph")
        head_capture["graph"] = inputs[1]
        head_capture["multiplier"] = output

    head_hook = head.register_forward_hook(capture_head)

    def capture_step(_module, args, kwargs):
        if live_tail_steps != 0:
            return
        theta = args[0]
        coupling = kwargs.get("coupling")
        if coupling is None or "multiplier" not in head_capture:
            return
        _, lag_cos, lag_sin = coupling
        mask = (head_capture["multiplier"] < 0).to(lag_cos.dtype)
        cos_kernel, sin_kernel = lag_cos * mask, lag_sin * mask
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        interaction = (cos_theta * torch.bmm(cos_kernel, sin_theta)
            - sin_theta * torch.bmm(cos_kernel, cos_theta)
            - cos_theta * torch.bmm(sin_kernel, cos_theta)
            - sin_theta * torch.bmm(sin_kernel, sin_theta))
        term = (kuramoto.K / float(theta.shape[1])) * interaction
        negative["sum_sq"] += float(term.detach().double().square().sum().cpu())
        negative["frames"] += int(term.numel())

    step_hook = None
    try:
        if live_tail_steps == 0:
            step_hook = kuramoto.register_forward_pre_hook(capture_step, with_kwargs=True)
        with install_polarity_prepare(graph, kuramoto, head):
            result = _legacy()._native_forward(found, rgb, live_tail_steps=live_tail_steps)
    finally:
        head_hook.remove()
        if step_hook is not None:
            step_hook.remove()
    if "graph" not in head_capture or "multiplier" not in head_capture:
        raise RuntimeError("SW0177 actual coupling did not execute the registered head")
    gamma, trace, adjacency, z, q, rho = result
    rms = (math.sqrt(negative["sum_sq"] / negative["frames"])
           if negative["frames"] else 0.0)
    return gamma, trace, adjacency, z, q, rho, head_capture["graph"], head_capture["multiplier"], rms


def _atomic_json_create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _binding_payload(arm, admitted, found, base_encoder, head, optimizer):
    base = _legacy()
    arrays = admitted["prepared"]["arrays"]
    cache_proof = admitted["prepared"]["admitted"]["prepared"]["cache_proof"]
    receipt = admitted["v2_receipt"]
    return {"experiment": EXPERIMENT, "arm": arm, "seed": 1,
        "v2_preflight_report_sha256": receipt["report_sha256"],
        "v2_queue_fingerprint_sha256": receipt["queue_fingerprint_sha256"],
        "preflight_implementation_fingerprint": receipt["preflight_fingerprint"],
        "source_proof": admitted["proof"],
        "source_state_sha256": admitted["proof"].get("source_cache_state_sha256"),
        "source_checkpoint_sha256": admitted["prepared"]["admitted"]["prepared"]["source_checkpoint_sha256"],
        "cache_manifest_sha256": admitted["prepared"]["admitted"]["prepared"]["cache_manifest_sha256"],
        "cache_arrays_sha256": admitted["prepared"]["admitted"]["prepared"]["arrays_sha256"],
        "ordered_train_ids": list(map(int, arrays["image_ids"])),
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(arrays["image_ids"], dtype="<i8").tobytes()),
        "initial_model_state_sha256": _state_sha(found),
        "initial_source_state_sha256": base._source_sha(found, base_encoder),
        "base_encoder_sha256": preflight._module("collaborative_test.SW_0171_contextual_instance_graph.preflight")._module_sha(base_encoder),
        "initial_head_state_sha256": _module("collaborative_test.SW_0171_contextual_instance_graph.preflight")._module_sha(head),
        "optimizer_groups": _optimizer_contract(optimizer),
        "optimizer_defaults": {key: ([float(x) for x in value] if key == "betas" else value)
            for key, value in optimizer.defaults.items()},
        "head_mode": arm, "head_seed": SEED,
        "training_fingerprint": _training_fingerprint()}


def _history_sha(history):
    return _sha_bytes(json.dumps(history, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _validate_checkpoint_model_state(payload, declared_sha):
    base = _legacy()
    if (payload.get("model_state_sha256") != declared_sha
            or base._state_sha_from_state(payload) != declared_sha):
        raise ValueError("SW0177 checkpoint tensor bytes differ from declared final model state")


def _named_gradients(optimizer):
    norms, raw = {}, {}
    for group in optimizer.param_groups:
        squared = 0.0
        for name, parameter in zip(group["parameter_names"], group["params"]):
            grad = parameter.grad
            if grad is not None:
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError(f"nonfinite SW0177 gradient {group['name']}/{name}")
                norm = float(grad.detach().double().norm().cpu())
                raw[(group["name"], name)] = norm
                squared += norm * norm
        norms[group["name"]] = math.sqrt(squared)
    return norms, raw


def _train_update(arm, found, head, optimizer, indices, admitted, device, update):
    arrays = admitted["prepared"]["arrays"]
    loss_api = _module("collaborative_test.SW_0175_critical_component_binding.critical_component_objective")
    base = _legacy()
    optimizer.zero_grad(set_to_none=True)
    before = {id(p): p.detach().clone() for group in optimizer.param_groups for p in group["params"]}
    all_losses, q_losses, micro_rows = [], [], []
    emitted_credit_sq = final_credit_sq = hidden_credit_sq = 0.0
    for offset in range(0, BATCH, MICRO):
        ids = np.asarray(indices[offset:offset + MICRO], dtype=np.int64)
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][ids])).to(device)
        _gamma, trace, adjacency, z, q, rho, graph_a, multiplier, _neg_rms = _native_forward(found, head, rgb,
            live_tail_steps=preflight.TAIL)
        values = []
        for local, index in enumerate(ids.tolist()):
            loss = loss_api.critical_component_loss(q[local], rho[local],
                arrays["feasible_targets"][index], base._support_mask(arrays["source_support"][index]))
            qloss = loss_api.native_q_loss(q[local], arrays["feasible_targets"][index],
                base._support_mask(arrays["source_support"][index]))
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(qloss)):
                raise FloatingPointError("SW0177 actual critical/Q loss is nonfinite")
            if update <= 2:
                grads = torch.autograd.grad(loss,
                    (trace["component_spikes"], head.layers[-1].weight,
                     head.layers[-1].bias, head.layers[0].weight, head.layers[0].bias),
                    retain_graph=True, allow_unused=True)
                emitted, final_w, final_b, hidden_w, hidden_b = grads
                if emitted is not None:
                    emitted_credit_sq += float(emitted.detach().double().square().sum().cpu())
                for grad, accum in ((final_w, "final"), (final_b, "final"),
                                    (hidden_w, "hidden"), (hidden_b, "hidden")):
                    if grad is not None:
                        norm_sq = float(grad.detach().double().square().sum().cpu())
                        if accum == "final": final_credit_sq += norm_sq
                        else: hidden_credit_sq += norm_sq
            values.append(loss)
            all_losses.append(float(loss.detach().cpu()))
            q_losses.append(float(qloss.detach().cpu()))
        base._backward_microbatch_loss(torch.stack(values), BATCH)
        micro_rows.append({"indices": ids.tolist(), "q_sha256": base._tensor_sha(q),
                           "z_sha256": base._tensor_sha(z)})
        del rgb, trace, adjacency, z, q, rho, graph_a, multiplier, values, _gamma, _neg_rms
    norms, raw = _named_gradients(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    clipped = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(clipped)):
        raise FloatingPointError("SW0177 global gradient norm is nonfinite")
    optimizer.step()
    found.wrapped.project_integrations_()
    deltas = {}
    named = {}
    for group in optimizer.param_groups:
        sq, rows = 0.0, []
        for name, parameter in zip(group["parameter_names"], group["params"]):
            delta = float((parameter.detach().double() - before[id(parameter)].double()).norm().cpu())
            sq += delta * delta
            rows.append({"name": name, "gradient_l2": raw.get((group["name"], name), 0.0),
                         "gradient_present": parameter.grad is not None, "parameter_delta_l2": delta})
        deltas[group["name"]] = math.sqrt(sq); named[group["name"]] = rows
    return {"update": int(update), "image_indices": list(map(int, indices)),
        "image_ids": [int(admitted["prepared"]["arrays"]["image_ids"][i]) for i in indices],
        "images_seen": BATCH, "microbatch": MICRO,
        "mean_loss": float(np.mean(all_losses)), "mean_q_loss": float(np.mean(q_losses)),
        "family_gradient_norms": norms, "family_parameter_delta_l2": deltas,
        "named_parameter_credit": named, "microbatches": micro_rows,
        "emitted_spike_gradient_l2": math.sqrt(emitted_credit_sq) if update <= 2 else None,
        "head_final_layer_gradient_l2": math.sqrt(final_credit_sq) if update <= 2 else None,
        "head_hidden_gradient_l2": math.sqrt(hidden_credit_sq) if update <= 2 else None,
        "clip_scale": min(1.0, 1.0 / (float(clipped.detach().cpu()) + 1e-12))}


def _credit_failures(row):
    result = []
    for key in ("emitted_spike_gradient_l2", "head_final_layer_gradient_l2"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            result.append(f"{key} is not finite-positive")
    if int(row["update"]) == 2:
        value = row.get("head_hidden_gradient_l2")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            result.append("head hidden layer did not receive second-update gradient credit")
    return result


def _save_rejection(work, arm, bindings, completed, history, reasons):
    path = Path(work) / "credit_rejection.json"
    data = {"experiment": EXPERIMENT, "status": "scientific_rejection", "arm": arm,
        "completed_updates": completed, "history_sha256": _history_sha(history),
        "boundary_sha256": _sha_file(Path(work) / "boundary.ptx"),
        "bindings": bindings, "rejection_reasons": list(reasons), "validation_data_read": False}
    _atomic_json_create(path, data)
    return data


def _diagnose_train8(found, head, admitted, phase):
    arrays = admitted["prepared"]["arrays"]
    base = _legacy()
    rows = []
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval(); head.eval()
    try:
        with torch.no_grad():
            for start in (256, 260):
                ids = np.arange(start, start + 4)
                rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][ids])).to(next(found.encoder.parameters()).device)
                _, trace, adjacency, z, q, rho, graph_a, multiplier, negative_kernel_rms = _native_forward(
                    found, head, rgb, live_tail_steps=0)
                for local, index in enumerate(ids):
                    support = base._support_mask(arrays["source_support"][index])
                    target = arrays["feasible_targets"][index]
                    rows.append({"position": int(index), "image_id": int(arrays["image_ids"][index]),
                        "q_sha256": base._tensor_sha(q[local]), "z_sha256": base._tensor_sha(z[local]),
                        "loss": float(_module("collaborative_test.SW_0175_critical_component_binding.critical_component_objective").critical_component_loss(
                            q[local], rho[local], target, support).cpu()),
                        "negative_edge_fraction": float(((multiplier[local] < 0) & (graph_a[local] > 0)).float().sum().cpu()
                            / max(1, int((graph_a[local] > 0).sum().cpu()))),
                        "negative_graph_weight": float((graph_a[local] * (multiplier[local] < 0)).sum().cpu()),
                        "negative_lag_kernel_term_rms_postsettle": negative_kernel_rms,
                        "negative_kernel_scope": "B4 batch aggregate over settled frames 512-1023; shared by batch rows"})
                del rgb, trace, adjacency, z, q, rho, multiplier, graph_a
    finally:
        found.encoder.train(); found.encoder.base_encoder.eval(); found.patcher.eval(); found.wrapped.train(); head.train()
    return {"phase": phase, "positions": list(range(256, 264)), "rows": rows,
        "mean_loss": float(np.mean([r["loss"] for r in rows]))}


def train(arm, *, physical_gpu, output_root=OUTPUT_ROOT):
    if arm not in ARMS:
        raise ValueError("SW0177 training arm must be signed or nonnegative")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("SW0177 trainer visibility differs from leased physical GPU")
    admitted = _load_admitted_inputs()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0177 source admission must precede CUDA initialization")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    found, base_encoder, source_sha, source_initial_state = _build_training_source(admitted, device)
    base = _legacy()
    head = seeded_edge_multiplier(seed=SEED, mode=arm).to(device)
    if base._state_sha(torch.nn.Identity(), head) != \
            admitted["v2_receipt"]["report"].get("head_initial_state_sha256"):
        raise ValueError("SW0177 initial polarity-head state differs from v2 preflight")
    optimizer, _families = _make_optimizer(found, head)
    base_encoder_sha = preflight._module("collaborative_test.SW_0171_contextual_instance_graph.preflight")._module_sha(base_encoder)
    bindings = _binding_payload(arm, admitted, found, base_encoder, head, optimizer)
    bindings["base_encoder_sha256"] = base_encoder_sha
    bindings["training_fingerprint"] = _training_fingerprint()
    runtime_module = _module(CHECKPOINT)
    runtime = runtime_module.runtime_signature(device)
    output_root = Path(output_root)
    output = output_root / f"{arm}_seed1"
    work = output_root / "work" / f"{arm}_seed1"
    if output.exists():
        return validate_training_artifact(arm, output, expected_bindings=bindings)
    if work.exists() and (work / "boundary.ptx").is_file():
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=device, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving unexpected SW0177 work files")
        completed, history = 0, []
        work.mkdir(parents=True, exist_ok=True)
        _atomic_json_create(work / "bindings.json", bindings)
        runtime_module.save_boundary(work, completed_updates=0, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
    if completed != len(history) or not 0 <= completed <= UPDATES:
        raise ValueError("SW0177 durable training cursor/history mismatch")
    expected = _expected_batches()
    if [row.get("image_indices") for row in history] != expected[:completed]:
        raise ValueError("SW0177 resumed cursor differs from registered NumPy order")
    rejection_path = work / "credit_rejection.json"
    if rejection_path.is_file():
        saved = json.loads(rejection_path.read_text(encoding="utf-8"))
        if (saved.get("experiment") != EXPERIMENT or saved.get("status") != "scientific_rejection"
                or not saved.get("rejection_reasons") or saved.get("bindings") != bindings
                or saved.get("completed_updates") != completed
                or saved.get("history_sha256") != _history_sha(history)
                or saved.get("boundary_sha256") != _sha_file(work / "boundary.ptx")):
            raise ValueError("SW0177 terminal credit rejection is not bound to boundary")
        return saved
    # Crash window: infer a first-two-update rejection from the committed
    # history before exposing another optimizer update.
    for row in history[:2]:
        reasons = _credit_failures(row)
        if reasons:
            return _save_rejection(work, arm, bindings, int(row["update"]), history, reasons)
    initial_path = work / "initial_train8.json"
    if initial_path.is_file():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    elif completed == 0:
        initial = _diagnose_train8(found, head, admitted, "initial_no_update")
        _atomic_json_create(initial_path, initial)
    else:
        raise FileNotFoundError("SW0177 resumed work lacks initial TRAIN8 diagnostics")
    cache_root = admitted["cache_root"]
    cache_before = {"manifest": _sha_file(cache_root / "manifest.json"),
                    "arrays": _sha_file(cache_root / "arrays.npz")}
    found.encoder.train(); found.encoder.base_encoder.eval(); found.patcher.eval(); found.wrapped.train(); head.train()
    frozen_before = base._frozen_source_components(found, base_encoder)
    for update in range(completed, UPDATES):
        # Reuse SW0175's registered per-update random seed and exact NumPy order.
        torch.manual_seed(171001 + update)
        torch.cuda.manual_seed(171001 + update)
        row = _train_update(arm, found, head, optimizer, expected[update], admitted,
                            device, update + 1)
        row["pass"] = update // 32 + 1
        history.append(row)
        runtime_module.save_boundary(work, completed_updates=update + 1,
            wrapped=found.wrapped, encoder=found.encoder, optimizer=optimizer,
            history=history, bindings=bindings, runtime=runtime, device=device)
        if update < 2:
            reasons = _credit_failures(row)
            if reasons:
                return _save_rejection(work, arm, bindings, update + 1, history, reasons)
    if cache_before != {"manifest": _sha_file(cache_root / "manifest.json"),
                        "arrays": _sha_file(cache_root / "arrays.npz")}:
        raise RuntimeError("SW0177 immutable input cache changed during training")
    final = _diagnose_train8(found, head, admitted, "after_update64")
    frozen_after = base._frozen_source_components(found, base_encoder)
    if frozen_after != frozen_before:
        raise ValueError("SW0177 frozen source components changed")
    final_source_sha = base._source_sha(found, base_encoder)
    return _publish(arm, output, work, found, head, optimizer, history, initial,
                    final, bindings, runtime, source_sha, base_encoder_sha,
                    cache_before, frozen_before, frozen_after, final_source_sha)


def _publish(arm, output, work, found, head, optimizer, history, initial, final,
             bindings, runtime, source_sha, base_encoder_sha, cache_sha,
             frozen_before, frozen_after, final_source_sha):
    if output.exists():
        raise FileExistsError("preserving existing SW0177 training output")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir()
    checkpoint = {"experiment": EXPERIMENT, "arm": arm, "bindings": bindings,
        "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
        "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
    checkpoint["model_state_sha256"] = _state_sha(found)
    with (stage / "checkpoint.pt").open("xb") as stream:
        torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
    with (stage / "optimizer.pt").open("xb") as stream:
        torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
    shutil.copyfile(work / "boundary.ptx", stage / "boundary.ptx")
    _atomic_json_create(stage / "history.json", {"updates": history})
    _atomic_json_create(stage / "initial_train8.json", initial)
    _atomic_json_create(stage / "final_train8.json", final)
    manifest = {"experiment": EXPERIMENT, "status": "training_complete", "arm": arm,
        "seed": 1, "updates": UPDATES, "completed_updates": UPDATES,
        "passes": PASSES, "training_images_per_pass": TRAIN_COUNT,
        "logical_batch": BATCH, "microbatch": MICRO, "exposures": UPDATES * BATCH,
        "validation_data_read": False, "bindings": bindings, "runtime": runtime,
        "train_image_indices": [i for batch in _expected_batches() for i in batch],
        "train_ids": [int(bindings["ordered_train_ids"][i])
                      for batch in _expected_batches() for i in batch],
        "source_model_state_before_sha256": source_sha,
        "source_model_state_after_sha256": final_source_sha,
        "base_encoder_sha256": base_encoder_sha,
        "frozen_source_components_before": frozen_before,
        "frozen_source_components_after": frozen_after,
        "cache_sha256": cache_sha,
        "initial_model_state_sha256": bindings["initial_model_state_sha256"],
        "final_model_state_sha256": checkpoint["model_state_sha256"],
        "optimizer_step_summary": _legacy()._optimizer_summary(optimizer),
        "checkpoint_sha256": _sha_file(stage / "checkpoint.pt"),
        "optimizer_sha256": _sha_file(stage / "optimizer.pt"),
        "boundary_sha256": _sha_file(stage / "boundary.ptx"),
        "history_sha256": _sha_file(stage / "history.json"),
        "initial_train8_sha256": _sha_file(stage / "initial_train8.json"),
        "final_train8_sha256": _sha_file(stage / "final_train8.json"),
        "training_fingerprint": bindings["training_fingerprint"]}
    _atomic_json_create(stage / "manifest.json", manifest)
    validate_training_artifact(arm, stage, expected_bindings=bindings)
    os.rename(stage, output)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def validate_training_artifact(arm, root, *, expected_bindings=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "training_complete"
            or manifest.get("arm") != arm or manifest.get("completed_updates") != UPDATES
            or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0177 training manifest experiment/cursor/scope mismatch")
    bindings = manifest.get("bindings")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("SW0177 training binding mismatch")
    if bindings.get("arm") != arm or bindings.get("head_mode") != arm:
        raise ValueError("SW0177 arm/head binding mismatch")
    if (manifest.get("frozen_source_components_before")
            != manifest.get("frozen_source_components_after")):
        raise ValueError("SW0177 frozen-source component hashes changed")
    for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("optimizer.pt", "optimizer_sha256"),
                      ("boundary.ptx", "boundary_sha256"), ("history.json", "history_sha256"),
                      ("initial_train8.json", "initial_train8_sha256"), ("final_train8.json", "final_train8_sha256")):
        if _sha_file(root / name) != manifest.get(key):
            raise ValueError(f"SW0177 training artifact hash mismatch: {name}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))["updates"]
    if len(history) != UPDATES or [r.get("update") for r in history] != list(range(1, UPDATES + 1)):
        raise ValueError("SW0177 training history cursor mismatch")
    expected = _expected_batches()
    if [r.get("image_indices") for r in history] != expected:
        raise ValueError("SW0177 training exposure order mismatch")
    for row in history:
        if (row.get("images_seen") != BATCH or len(row.get("microbatches", [])) != BATCH // MICRO
                or set(row.get("family_gradient_norms", {})) !=
                {"context_encoder", "graph", "native_core", "integration", "polarity_head"}):
            raise ValueError("SW0177 training update family/geometry mismatch")
        if not math.isfinite(float(row.get("mean_loss", math.nan))):
            raise ValueError("SW0177 training loss is nonfinite")
    for row in history[:2]:
        if _credit_failures(row):
            raise ValueError("SW0177 first-two-update actual credit failed")
    payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if payload.get("arm") != arm or payload.get("bindings") != bindings:
        raise ValueError("SW0177 final checkpoint lineage mismatch")
    _validate_checkpoint_model_state(payload, manifest.get("final_model_state_sha256"))
    optimizer_payload = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    _legacy()._validate_optimizer_state(optimizer_payload, manifest.get("optimizer_step_summary"))
    return manifest


def validate_training_rejection(arm, work):
    """Validate a terminal first-two-update rejection against its durable boundary."""
    import io
    from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as checkpoint_api

    work = Path(work)
    report = json.loads((work / "credit_rejection.json").read_text(encoding="utf-8"))
    metadata = checkpoint_api.inspect_committed_metadata(work)
    raw = (work / "boundary.ptx").read_bytes()
    outer = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    payload_bytes = outer.get("payload_bytes")
    if not isinstance(payload_bytes, bytes) or _sha_bytes(payload_bytes) != outer.get("payload_sha256"):
        raise ValueError("SW0177 rejection boundary payload hash mismatch")
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    bindings, history = payload.get("bindings"), payload.get("history")
    if (report.get("experiment") != EXPERIMENT or report.get("status") != "scientific_rejection"
            or report.get("arm") != arm or not report.get("rejection_reasons")
            or report.get("completed_updates") != payload.get("completed_updates")
            or report.get("boundary_sha256") != _sha_file(work / "boundary.ptx")
            or report.get("history_sha256") != _history_sha(history)
            or report.get("bindings") != bindings or metadata.get("bindings") != bindings
            or report.get("validation_data_read") is not False
            or not isinstance(bindings, dict)
            or bindings.get("experiment") != EXPERIMENT or bindings.get("arm") != arm
            or len(history) != payload.get("completed_updates")):
        raise ValueError("SW0177 scientific rejection is not bound to its committed boundary")
    return report

