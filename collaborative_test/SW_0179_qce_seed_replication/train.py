"""Fresh-source SW0179 supervised product-QCE seed replication."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0178_temporal_class_competition.preflight import flatten_modal_targets
from collaborative_test.SW_0178_temporal_class_competition.class_competition import leave_one_out_class_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0179_qce_seed_replication"
ARMS = ("product_q_ce",)
COUNT, PASSES, UPDATES, BATCH, MICRO = 512, 2, 64, 16, 4
SHUFFLE_SEED = 171001
MODEL_SEED_BASE = 171000
CONTEXT_SEED_BASE = 171000
ENCODER_LR, GRAPH_CORE_LR, INTEGRATION_LR = 3e-4, 3e-5, 1e-3
CLIP_NORM = 1.0
MASK_SUPERVISED = True
UNSUPERVISED_LEADERBOARD_ELIGIBLE = False
SEED1_REFERENCE_INITIAL_STATE_SHA256 = "f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5"
OUTPUT_ROOT = HERE / "results_archive" / "training_20261016"
DATA_CACHE_MODULE = "collaborative_test.SW_0171_contextual_instance_graph.preflight"
SOURCE_CAL_MODULE = "collaborative_test.SW_0169_normalized_hard_reset.calibration_v2"
SOURCE_UTIL_MODULE = "collaborative_test.SW_0175_critical_component_binding.train_v2"
CHECKPOINT_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint"


def _module(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(tensor):
    return _sha_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _state_sha(wrapped, encoder):
    digest = hashlib.sha256()
    for section, module in (("wrapped", wrapped), ("encoder", encoder)):
        for name, value in sorted(module.state_dict().items()):
            digest.update((section + "." + name).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_sha_from_payload(payload):
    digest = hashlib.sha256()
    for section in ("wrapped", "encoder"):
        for name, value in sorted(payload[section].items()):
            digest.update((section + "." + name).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def implementation_fingerprint():
    utils = _module(SOURCE_UTIL_MODULE)
    paths = (
        "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md",
        "collaborative_test/SW_0178_temporal_class_competition/class_competition.py",
        "collaborative_test/SW_0178_temporal_class_competition/preflight.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    files = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0179 fingerprint dependency missing: {relative}")
        files[relative] = _sha_file(path)
    return {"experiment": EXPERIMENT, "files": files,
            "source_trainer_dependency": utils._training_fingerprint(),
            "objectives": list(ARMS), "source_seeds": [0, 2], "context_seed_base": CONTEXT_SEED_BASE,
            "shuffle_seed": SHUFFLE_SEED, "count": COUNT, "passes": PASSES,
            "updates": UPDATES, "batch": BATCH, "micro": MICRO,
            "encoder_lr": ENCODER_LR, "graph_core_lr": GRAPH_CORE_LR,
            "integration_lr": INTEGRATION_LR, "clip_norm": CLIP_NORM,
            "temperature": 0.1,
            "mask_supervised": MASK_SUPERVISED,
            "unsupervised_leaderboard_eligible": UNSUPERVISED_LEADERBOARD_ELIGIBLE}


def fingerprint_sha256():
    return _sha_bytes(json.dumps(implementation_fingerprint(), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8"))


def prepare_training_inputs_cpu():
    """Load only the common, immutable SW171 TRAIN512 RGB/modal dataset."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 data preparation must precede CUDA initialization")
    data_preflight = _module(DATA_CACHE_MODULE)
    prepared = data_preflight.prepare_inputs_cpu()
    arrays = prepared["arrays"]
    ids = list(map(int, prepared["cache_proof"]["image_ids"]))
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise ValueError("SW171 admitted ordered IDs are not a unique TRAIN512 set")
    if arrays["rgb"].dtype != np.uint8 or arrays["rgb"].shape != (COUNT, 128, 128, 3):
        raise ValueError("SW171 RGB cache has invalid native128 schema")
    modal = np.asarray(arrays["modal_gt"])
    if modal.dtype != np.int32 or modal.shape != (COUNT, 32, 32):
        raise ValueError("SW171 modal TRAIN cache has invalid native32 schema")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 data preparation initialized CUDA")
    return {"arrays": arrays, "dataset_prepared": prepared,
            "ordered_train_ids": ids, "dataset_manifest_sha256": prepared["cache_manifest_sha256"],
            "dataset_arrays_sha256": prepared["arrays_sha256"],
        "dataset_producer_proof": prepared["source_proof"],
            "cuda_initialized": False}


def _expected_batches():
    rng = np.random.RandomState(SHUFFLE_SEED)
    batches = []
    for _ in range(PASSES):
        order = rng.permutation(COUNT).astype(np.int64)
        batches.extend([order[i:i+BATCH].tolist() for i in range(0, COUNT, BATCH)])
    if len(batches) != UPDATES or any(len(row) != BATCH for row in batches):
        raise AssertionError("SW0179 registered order does not produce 64 complete B16 updates")
    return batches


def _validate_cursor(completed, history):
    expected = _expected_batches()
    if type(completed) is not int or completed != len(history) or not 0 <= completed <= UPDATES:
        raise ValueError("SW0179 durable update cursor/history mismatch")
    if any(row.get("update") != index + 1 or row.get("image_indices") != expected[index]
           for index, row in enumerate(history)):
        raise ValueError("SW0179 durable cursor differs from registered training order")


def _vectorized_class_loss(affinity, labels, *, temperature=0.1):
    """Vectorized form of the registered LOO objective for one image.

    It has the same class prototypes/averaging as the clear reference helper,
    while avoiding a per-pixel autograd graph and per-pixel device syncs.
    """
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square [N,N]")
    if labels.ndim != 1 or labels.shape[0] != affinity.shape[0]:
        raise ValueError("labels must be [N]")
    if labels.device != affinity.device:
        raise ValueError("labels and affinity must share a device")
    classes, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True,
                                             return_counts=True)
    if classes.numel() < 2:
        return affinity.sum() * 0.0
    node_count, class_count = labels.numel(), classes.numel()
    membership = torch.nn.functional.one_hot(inverse, num_classes=class_count).to(affinity.dtype)
    totals = affinity @ membership
    prototypes = totals / counts.to(affinity.dtype).unsqueeze(0)
    own_counts = counts.index_select(0, inverse)
    own_total = totals.gather(1, inverse[:, None]).squeeze(1)
    diagonal = torch.diagonal(affinity)
    own_loo = (own_total - diagonal) / (own_counts - 1).clamp_min(1).to(affinity.dtype)
    logits = prototypes.scatter(1, inverse[:, None], own_loo[:, None]) / float(temperature)
    per_query = torch.nn.functional.cross_entropy(logits, inverse, reduction="none")
    valid = own_counts >= 2
    if not bool(valid.any()):
        return affinity.sum() * 0.0
    valid_losses = per_query[valid]
    valid_classes = inverse[valid]
    sums = torch.zeros(class_count, dtype=affinity.dtype, device=affinity.device)
    sums.scatter_add_(0, valid_classes, valid_losses)
    query_counts = torch.zeros(class_count, dtype=affinity.dtype, device=affinity.device)
    query_counts.scatter_add_(0, valid_classes, torch.ones_like(valid_losses))
    per_class = sums[query_counts > 0] / query_counts[query_counts > 0]
    return per_class.mean()


def _backward_microbatch_losses(losses):
    """Backpropagate one shared B4 forward graph exactly once."""
    if len(losses) != MICRO or any(value.ndim != 0 for value in losses):
        raise ValueError("SW0179 expects four scalar image losses from the shared microbatch graph")
    stacked = torch.stack(losses)
    if not bool(torch.isfinite(stacked).all()):
        raise FloatingPointError("SW0179 microbatch has nonfinite losses")
    (stacked.sum() / BATCH).backward()


def _source_parts(seed, device, inputs):
    source_cal = _module(SOURCE_CAL_MODULE)
    source_utils = _module(SOURCE_UTIL_MODULE)
    foundation = _module("collaborative_test.SW_0135_native32_spike_binding.foundation")
    geodesic = _module("collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    dev = torch.device(device)
    fork_devices = [dev.index if dev.index is not None else 0] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(149001)
        if dev.type == "cuda":
            torch.cuda.manual_seed(149001)
        found = foundation.load_native32_foundation(seed, device, verify_rgb_assets=False)
    geodesic.install_operator(found.wrapped.core.graph_generator, "T")
    if (getattr(found.wrapped.core.graph_generator, "_sw0143_geodesic_arm", None) != "T"
            or found.wrapped.core.kuramoto_backend != "factorized"):
        raise ValueError("SW0179 requires the seed-matched original pretrained SourceT dynamics")
    source_sha = source_cal._source_digest(found)
    contract = _module("collaborative_test.SW_0130_phase_state_integration.run")
    checkpoint, source_manifest_path, source_manifest, _, _, checkpoint_sha = contract.source_contract(seed)
    if found.provenance.get("source_core_sha256") != checkpoint_sha:
        raise ValueError("loaded native32 foundation does not match seed-specific SW0097 source contract")
    base_encoder = found.encoder
    base_sha = source_utils.preflight._module_sha(base_encoder)
    components_before = source_utils._frozen_source_components(found, base_encoder)
    source_binding = {"seed": int(seed), "source_state_sha256": source_sha,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_manifest_sha256": _sha_file(source_manifest_path),
        "source_manifest_path": str(source_manifest_path),
        "source_steps": int(source_manifest["steps"]),
        "source_core_path": str(checkpoint),
        "encoder_sha256": found.provenance["encoder_sha256"],
        "feature_preprocessing_sha256": found.provenance["feature_preprocessing_sha256"]}
    return found, source_cal, source_utils, base_encoder, source_sha, base_sha, components_before, source_binding


def _make_fresh(arm, seed, inputs, device):
    if arm not in ARMS:
        raise ValueError("unregistered SW0179 objective")
    source_utils = _module(SOURCE_UTIL_MODULE)
    data = inputs["dataset_prepared"]
    found, source_cal, source_utils, base_encoder, source_sha, base_sha, frozen_before, source_binding = _source_parts(seed, device, inputs)
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
    found.encoder = ContextualFeatureEncoder(base_encoder, seed=CONTEXT_SEED_BASE + int(seed)).to(device)
    found.encoder.eval()
    final_context = found.encoder.context[-1]
    if (not torch.equal(final_context.weight.detach(), torch.zeros_like(final_context.weight))
            or not torch.equal(final_context.bias.detach(), torch.zeros_like(final_context.bias))):
        raise ValueError("SW0179 contextual residual must start at exact zero output")
    optimizer, families = source_utils._make_optimizer(found)
    state_sha = _state_sha(found.wrapped, found.encoder)
    if int(seed) == 1 and state_sha != SEED1_REFERENCE_INITIAL_STATE_SHA256:
        raise ValueError("SW0179 seed-1 fresh factory no longer matches the frozen seed-1 reference state")
    binding = _bindings(arm, seed, inputs, source_binding, base_sha, state_sha, families)
    return found, optimizer, families, binding, (source_cal, source_utils, base_encoder, frozen_before, data)


def prepare_fresh_training_state_cpu(seed: int, inputs=None):
    """Prove seed-specific source/context/optimizer initialization before CUDA."""
    seed = int(seed)
    if seed not in (0, 1, 2):
        raise ValueError("SW0179 factory admission accepts only registered source seeds 0, 1, or 2")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 fresh-model admission must precede CUDA initialization")
    inputs = prepare_training_inputs_cpu() if inputs is None else inputs
    found, optimizer, families, bindings, _setup = _make_fresh("product_q_ce", seed, inputs, "cpu")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 CPU source/context factory initialized CUDA")
    family_names = list(families)
    names = [name for rows in families.values() for name, _ in rows]
    if len(names) != len(set(names)) or len(names) != sum(len(rows) for rows in families.values()):
        raise ValueError("SW0179 optimizer families overlap or contain duplicate parameter names")
    adam_state_size = len(optimizer.state)
    if adam_state_size != 0:
        raise ValueError("SW0179 fresh Adam must have an empty state before its first update")
    result = {"experiment": EXPERIMENT, "seed": seed,
        "context_seed": CONTEXT_SEED_BASE + seed,
        "source_binding": bindings["source_binding"],
        "source_state_sha256": bindings["source_state_sha256"],
        "base_encoder_sha256": bindings["base_encoder_sha256"],
        "initial_model_state_sha256": bindings["initial_model_state_sha256"],
        "optimizer_groups": bindings["optimizer_groups"],
        "family_names": family_names, "parameter_count": len(names),
        "adam_state_entries": adam_state_size,
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "ordered_train_ids_sha256": _sha_bytes(np.asarray(inputs["ordered_train_ids"], dtype="<i8").tobytes()),
        "validation_data_read": False, "ground_truth_scope": "TRAIN modal labels only",
        "cuda_initialized": False, "training_fingerprint": implementation_fingerprint()}
    return {"proof": result, "found": found, "optimizer": optimizer,
            "families": families, "bindings": bindings, "inputs": inputs}


def _optimizer_contract(optimizer):
    return [{"name": group["name"], "parameter_names": list(group["parameter_names"]),
             "lr": float(group["lr"]), "betas": list(group["betas"]),
             "eps": float(group["eps"]), "weight_decay": float(group["weight_decay"])}
            for group in optimizer.param_groups]


def _bindings(arm, seed, inputs, source_binding, base_sha, state_sha, families):
    if set(families) != {"context_encoder", "graph", "native_core", "integration"}:
        raise ValueError("SW0179 optimizer family schema changed")
    return {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed),
        "objective": arm, "context_seed": CONTEXT_SEED_BASE + int(seed),
        "model_update_seed_base": MODEL_SEED_BASE + int(seed), "temperature": 0.1,
        "mask_supervised": MASK_SUPERVISED,
        "unsupervised_leaderboard_eligible": UNSUPERVISED_LEADERBOARD_ELIGIBLE,
        "source_binding": source_binding, "source_state_sha256": source_binding["source_state_sha256"],
        "base_encoder_sha256": base_sha,
        "source_checkpoint_sha256": source_binding["source_checkpoint_sha256"],
        "dataset_producer_proof": inputs["dataset_producer_proof"],
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
        "dataset_arrays_sha256": inputs["dataset_arrays_sha256"],
        "ordered_train_ids": inputs["ordered_train_ids"],
        "initial_model_state_sha256": state_sha,
        "optimizer_groups": _optimizer_contract(_make_optimizer_from_families(families)),
        "training_fingerprint": implementation_fingerprint()}


def _make_optimizer_from_families(families):
    groups = []
    for name, entries in families.items():
        groups.append({"name": name, "params": [p for _, p in entries],
            "parameter_names": [n for n, _ in entries],
            "lr": INTEGRATION_LR if name == "integration" else ENCODER_LR if name == "context_encoder" else GRAPH_CORE_LR})
    return torch.optim.Adam(groups)


def _loss_for_image(arm, q, rho, labels):
    if arm != "product_q_ce":
        raise ValueError("SW0179 registers only the product-QCE objective")
    return _vectorized_class_loss(q, labels, temperature=0.1)


def _train_update(arm, found, optimizer, image_indices, inputs, device, update_number):
    from collaborative_test.SW_0175_critical_component_binding import train_v2 as util
    arrays = inputs["arrays"]
    optimizer.zero_grad(set_to_none=True)
    before = {group["name"]: [p.detach().clone() for p in group["params"]]
              for group in optimizer.param_groups}
    losses, q_losses, emitted_grad_sq = [], [], 0.0
    micro_records = []
    for offset in range(0, BATCH, MICRO):
        indices = image_indices[offset:offset+MICRO]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][indices])).to(device)
        _gamma, trace, adjacency, z, q, rho = util._native_forward(
            found, rgb, live_tail_steps=64)
        labels_batch = flatten_modal_targets(arrays["modal_gt"][indices]).to(device)
        micro_losses = []
        for local in range(MICRO):
            labels = labels_batch[local]
            loss = _loss_for_image(arm, q[local], rho[local], labels)
            qloss = _vectorized_class_loss(q[local], labels, temperature=0.1)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(qloss)):
                raise FloatingPointError(f"nonfinite SW0179 loss at update {update_number}")
            if update_number <= 2:
                grad = torch.autograd.grad(loss, trace["component_spikes"], retain_graph=True,
                                           allow_unused=True)[0]
                if grad is not None:
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError("nonfinite SW0179 loss-to-emitted-S gradient")
                    emitted_grad_sq += float(grad.detach().double().square().sum().cpu())
            micro_losses.append(loss)
            losses.append(float(loss.detach().cpu()))
            q_losses.append(float(qloss.detach().cpu()))
        _backward_microbatch_losses(micro_losses)
        micro_records.append({"image_indices": list(map(int, indices)),
                              "q_sha256": _tensor_sha(q), "z_sha256": _tensor_sha(z)})
        del rgb, trace, adjacency, z, q, rho, labels_batch

    family_norms, raw_param_norms = {}, {}
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    for group in optimizer.param_groups:
        sq = 0.0
        for name, parameter in zip(group["parameter_names"], group["params"]):
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"nonfinite SW0179 gradient {group['name']}/{name}")
                norm = float(parameter.grad.detach().double().norm().cpu())
                raw_param_norms[id(parameter)] = norm
                sq += norm * norm
        family_norms[group["name"]] = math.sqrt(sq)
    total_norm = math.sqrt(sum(value * value for value in family_norms.values()))
    clipped = torch.nn.utils.clip_grad_norm_(all_params, CLIP_NORM)
    if not bool(torch.isfinite(clipped)):
        raise FloatingPointError("nonfinite SW0179 global gradient norm")
    optimizer.step()
    found.wrapped.project_integrations_()
    deltas, named = {}, {}
    for group in optimizer.param_groups:
        sq, rows = 0.0, []
        for name, old, parameter in zip(group["parameter_names"], before[group["name"]], group["params"]):
            delta = float((parameter.detach().double() - old.detach().double()).norm().cpu())
            if not math.isfinite(delta):
                raise FloatingPointError("nonfinite SW0179 optimizer delta")
            sq += delta * delta
            rows.append({"name": name, "gradient_l2": raw_param_norms.get(id(parameter), 0.0),
                         "gradient_present": parameter.grad is not None, "parameter_delta_l2": delta})
        deltas[group["name"]], named[group["name"]] = math.sqrt(sq), rows
    emitted_grad = math.sqrt(emitted_grad_sq) if update_number <= 2 else None
    return {"update": int(update_number), "image_indices": list(map(int, image_indices)),
        "image_ids": [int(inputs["ordered_train_ids"][i]) for i in image_indices],
        "images_seen": BATCH, "microbatches": micro_records,
        "mean_registered_loss": float(np.mean(losses)), "mean_q_loss": float(np.mean(q_losses)),
        "gradient_norm_before_clip": total_norm,
        "clip_scale": min(1.0, CLIP_NORM / (float(clipped.detach().cpu()) + 1e-12)),
        "family_gradient_norms": family_norms, "family_parameter_delta_l2": deltas,
        "named_parameter_credit": named, "loss_to_emitted_component_spikes_gradient_l2": emitted_grad}


def _credit_failures(row):
    failures = []
    emitted = row.get("loss_to_emitted_component_spikes_gradient_l2")
    if not isinstance(emitted, (int, float)) or not math.isfinite(emitted) or emitted <= 0:
        failures.append("loss-to-emitted-S gradient is not finite-positive")
    groups = {"context_encoder", "graph", "native_core", "integration"}
    grads, deltas = row.get("family_gradient_norms", {}), row.get("family_parameter_delta_l2", {})
    if set(grads) != groups or set(deltas) != groups:
        failures.append("four optimizer-family gradient/delta records are incomplete")
    else:
        for name in groups:
            if (not math.isfinite(float(grads[name])) or float(grads[name]) < 0
                    or not math.isfinite(float(deltas[name])) or float(deltas[name]) < 0):
                failures.append(f"{name} family gradient/delta is invalid")
            elif float(grads[name]) <= 0 or float(deltas[name]) <= 0:
                failures.append(f"{name} family has no finite-positive first-two-update credit/change")
    if not math.isfinite(float(row.get("gradient_norm_before_clip", math.nan))):
        failures.append("global gradient norm is nonfinite")
    return failures


def _atomic_json_create(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def _artifact_paths(output_root, arm, seed):
    name = f"{arm}_seed{int(seed)}"
    return Path(output_root) / "work" / name, Path(output_root) / name


def train(arm, *, seed: int, physical_gpu: int, output_root=OUTPUT_ROOT):
    if arm not in ARMS or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("SW0179 requires its registered objective and assigned physical GPU")
    if int(seed) not in (0, 2):
        raise ValueError("SW0179 trains only fresh source seeds 0 and 2")
    inputs = prepare_training_inputs_cpu()
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 CUDA initialized before CPU dataset admission")
    cpu_admission = prepare_fresh_training_state_cpu(seed, inputs)
    cpu_proof = cpu_admission["proof"]
    del cpu_admission
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    source_utils = _module(SOURCE_UTIL_MODULE)
    found, optimizer, families, bindings, setup = _make_fresh(arm, seed, inputs, device)
    if bindings["initial_model_state_sha256"] != cpu_proof["initial_model_state_sha256"]:
        raise ValueError("SW0179 GPU factory initial model differs from the pre-CUDA CPU admission")
    source_cal, source_utils, base_encoder, frozen_before, _data = setup
    runtime_module = _module(CHECKPOINT_MODULE)
    runtime = runtime_module.runtime_signature(device)
    work, output = _artifact_paths(output_root, arm, seed)
    if output.exists():
        return validate_training_artifact(arm, output, seed=seed, expected_bindings=bindings)
    if work.exists() and (work / "boundary.ptx").is_file():
        loaded = runtime_module.load_latest(work, wrapped=found.wrapped, encoder=found.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=device, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("preserving unexpected partial SW0179 work directory")
        completed, history = 0, []
        _atomic_json_create(work / "run_bindings.json", bindings)
        runtime_module.save_boundary(work, completed_updates=0, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
    _validate_cursor(completed, history)
    rejection = work / "credit_rejection.json"
    if rejection.is_file():
        saved = json.loads(rejection.read_text(encoding="utf-8"))
        if (saved.get("bindings") != bindings or saved.get("completed_updates") > completed
                or saved.get("history_sha256") != _sha_bytes(json.dumps(history, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode())):
            raise ValueError("SW0179 rejection receipt is not bound to current committed history")
        return saved
    expected = _expected_batches()
    found.encoder.train(); found.encoder.base_encoder.eval()
    found.patcher.eval(); found.wrapped.train()
    dataset_cache = Path(inputs["dataset_prepared"]["cache_root"])
    source_checkpoint = Path(bindings["source_binding"]["source_core_path"])
    source_manifest = Path(bindings["source_binding"]["source_manifest_path"])
    cache_before = {"manifest": _sha_file(dataset_cache / "manifest.json"),
                    "arrays": _sha_file(dataset_cache / "arrays.npz"),
                    "source_checkpoint": _sha_file(source_checkpoint),
                    "source_manifest": _sha_file(source_manifest)}
    for update in range(completed, UPDATES):
        indices = np.asarray(expected[update], dtype=np.int64)
        update_seed = MODEL_SEED_BASE + int(seed) + update
        torch.manual_seed(update_seed)
        torch.cuda.manual_seed(update_seed)
        row = _train_update(arm, found, optimizer, indices, inputs, device, update + 1)
        row["source_seed"] = int(seed)
        row["pass"] = update // 32 + 1
        history.append(row)
        runtime_module.save_boundary(work, completed_updates=update + 1, wrapped=found.wrapped,
            encoder=found.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=device)
        if update < 2:
            failures = _credit_failures(row)
            if failures:
                rejection_record = {"experiment": EXPERIMENT, "status": "scientific_rejection",
                    "arm": arm, "completed_updates": update + 1, "history_sha256": _sha_bytes(
                    json.dumps(history, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
                    "bindings": bindings, "rejection_reasons": failures,
                    "validation_data_read": False}
                _atomic_json_create(rejection, rejection_record)
                return rejection_record
    after_cache = {"manifest": _sha_file(dataset_cache / "manifest.json"),
                   "arrays": _sha_file(dataset_cache / "arrays.npz"),
                   "source_checkpoint": _sha_file(source_checkpoint),
                   "source_manifest": _sha_file(source_manifest)}
    if cache_before != after_cache:
        raise RuntimeError("SW0179 shared TRAIN input cache changed during model updates")
    frozen_after = source_utils._frozen_source_components(found, base_encoder)
    if frozen_after != frozen_before:
        raise ValueError("SW0179 frozen seed-specific SourceT/base components changed during training")
    return _publish_training(arm, seed, output, work, found, optimizer, history, bindings,
        runtime, inputs, cache_before, frozen_before, frozen_after, source_cal._source_digest(found))


def _publish_training(arm, seed, output, work, found, optimizer, history, bindings, runtime,
                      inputs, cache_hashes, frozen_before, frozen_after, trained_source_sha):
    if output.exists(): raise FileExistsError("preserving existing SW0179 output artifact")
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = {"experiment": EXPERIMENT, "arm": arm, "seed": int(seed), "bindings": bindings,
            "wrapped": {k: v.detach().cpu().clone() for k, v in found.wrapped.state_dict().items()},
            "encoder": {k: v.detach().cpu().clone() for k, v in found.encoder.state_dict().items()}}
        checkpoint["model_state_sha256"] = _state_sha(found.wrapped, found.encoder)
        with (stage / "checkpoint.pt").open("xb") as stream:
            torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
        with (stage / "optimizer.pt").open("xb") as stream:
            torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
        shutil.copyfile(work / "boundary.ptx", stage / "boundary.ptx")
        _atomic_json_create(stage / "history.json", {"updates": history})
        util = _module(SOURCE_UTIL_MODULE)
        manifest = {"experiment": EXPERIMENT, "status": "training_complete", "arm": arm,
            "seed": int(seed), "completed_updates": UPDATES, "updates": UPDATES, "passes": PASSES,
            "training_images_per_pass": COUNT, "logical_batch": BATCH, "microbatch": MICRO,
            "exposures": UPDATES * BATCH, "train_image_indices": [i for b in _expected_batches() for i in b],
            "train_ids": [x for row in history for x in row["image_ids"]],
            "validation_data_read": False, "ground_truth_scope": "TRAIN modal labels only",
            "bindings": bindings, "runtime": runtime,
            "source_model_state_before_sha256": bindings["source_state_sha256"],
            "source_model_state_after_sha256": trained_source_sha,
            "initial_model_state_sha256": bindings["initial_model_state_sha256"],
            "frozen_base_encoder_sha256_before": bindings["base_encoder_sha256"],
            "frozen_base_encoder_sha256_after": util.preflight._module_sha(found.encoder.base_encoder),
            "frozen_source_components_before": frozen_before, "frozen_source_components_after": frozen_after,
            "final_model_state_sha256": checkpoint["model_state_sha256"],
            "optimizer_step_summary": util._optimizer_summary(optimizer),
            "dataset_file_sha256": cache_hashes,
            "history_sha256": _sha_file(stage / "history.json"),
            "checkpoint_sha256": _sha_file(stage / "checkpoint.pt"),
            "optimizer_sha256": _sha_file(stage / "optimizer.pt"),
            "boundary_sha256": _sha_file(stage / "boundary.ptx"),
            "training_fingerprint": bindings["training_fingerprint"]}
        _atomic_json_create(stage / "manifest.json", manifest)
        validate_training_artifact(arm, stage, seed=seed, expected_bindings=bindings)
        os.rename(stage, output)
    except BaseException:
        raise
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def validate_training_artifact(arm, root, *, seed=None, expected_bindings=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or manifest.get("status") != "training_complete"
            or manifest.get("arm") != arm or manifest.get("completed_updates") != UPDATES
            or manifest.get("validation_data_read") is not False):
        raise ValueError("SW0179 training artifact manifest is not a complete 64-update TRAIN-only result")
    bindings = manifest.get("bindings")
    artifact_seed = manifest.get("seed")
    if artifact_seed not in (0, 2) or (seed is not None and artifact_seed != int(seed)):
        raise ValueError("SW0179 artifact is not for the requested source seed")
    if expected_bindings is not None and bindings != expected_bindings:
        raise ValueError("SW0179 artifact bindings differ from current registered source")
    if (not isinstance(bindings, dict) or bindings.get("arm") != arm
            or bindings.get("seed") != artifact_seed
            or bindings.get("training_fingerprint") != implementation_fingerprint()
            or manifest.get("training_fingerprint") != bindings.get("training_fingerprint")
            or manifest.get("source_model_state_before_sha256") != bindings.get("source_state_sha256")
            or manifest.get("frozen_base_encoder_sha256_before") != bindings.get("base_encoder_sha256")
            or manifest.get("frozen_base_encoder_sha256_after") != bindings.get("base_encoder_sha256")
            or manifest.get("frozen_source_components_before") != manifest.get("frozen_source_components_after")):
        raise ValueError("SW0179 artifact lineage/frozen-source binding mismatch")
    for filename, key in (("history.json", "history_sha256"), ("checkpoint.pt", "checkpoint_sha256"),
                          ("optimizer.pt", "optimizer_sha256"), ("boundary.ptx", "boundary_sha256")):
        if _sha_file(root / filename) != manifest.get(key):
            raise ValueError(f"SW0179 artifact hash mismatch: {filename}")
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))["updates"]
    _validate_cursor(UPDATES, history)
    if len(history) != UPDATES or any(not math.isfinite(float(row["mean_registered_loss"])) for row in history):
        raise ValueError("SW0179 training history is incomplete or nonfinite")
    ordered_ids = bindings.get("ordered_train_ids")
    batches = _expected_batches()
    if (not isinstance(ordered_ids, list) or len(ordered_ids) != COUNT
            or [row.get("image_ids") for row in history]
               != [[int(ordered_ids[index]) for index in batch] for batch in batches]
            or manifest.get("train_image_indices") != [index for batch in batches for index in batch]
            or manifest.get("train_ids") != [int(ordered_ids[index]) for batch in batches for index in batch]):
        raise ValueError("SW0179 artifact TRAIN IDs/order differ from bound input mapping")
    for row in history[:2]:
        if _credit_failures(row):
            raise ValueError("SW0179 first-two-update credit evidence failed")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("bindings") != bindings or checkpoint.get("arm") != arm
            or checkpoint.get("seed") != artifact_seed
            or _state_sha_from_payload(checkpoint) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0179 checkpoint payload/state digest mismatch")
    optimizer_payload = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
    util = _module(SOURCE_UTIL_MODULE)
    util._validate_optimizer_state(optimizer_payload, manifest.get("optimizer_step_summary"))
    return manifest
