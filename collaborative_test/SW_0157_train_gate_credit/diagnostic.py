"""Read-only SW0157 event-margin and TRAIN relation-gradient diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0155_supervised_spike_relations import artifactvalidator, objective, predict_evaluate, train

EXPERIMENT = "SW0157_train_gate_credit_diagnostic"
MODEL_ORDER = ("source97", "source97_T", "SW0155_encoder_graph", "SW0155_joint")
POSITIONS = (0, 1, 2, 3, 2048, 2049, 2050, 2051)
BATCH_POSITIONS = (POSITIONS[:4], POSITIONS[4:])
MODEL_ARMS = {"SW0155_encoder_graph": "encoder_graph", "SW0155_joint": "joint"}
SEED = 1
ROLLOUT_SEED = 157001
STEPS, SETTLE, LIVE_TAIL = 1024, 512, 64
WATCHDOG_SECONDS = 300.0
OUTPUT = HERE / "results_archive" / "train_gate_credit_seed1.json"
PARAMETER_GROUPS = ("spatial_residual", "graph", "gamma_drive", "kuramoto",
                    "dendritic", "membrane", "integration_tensors", "other_core")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [HERE / "protocol.json", HERE / "diagnostic.py", HERE / "README.md"]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/artifactvalidator.py",
        "collaborative_test/SW_0155_supervised_spike_relations/preflight_migration.py",
        "collaborative_test/SW_0155_supervised_spike_relations/predict_evaluate.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
        "snn_kuramoto_bidirectional/dendric_layer.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
    ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0157 implementation dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _source_and_assets():
    source = train.source_contract()
    assets = sw130.validate_rgb_assets()
    if (len(source["image_ids"]) != 4096 or len(source["rows"]) != 4096
            or len(set(source["image_ids"])) != 4096
            or len(np.unique(source["rows"])) != 4096):
        raise ValueError("SW0157 requires the exact 4096-row source TRAIN contract")
    expected_hdf = Path(train.HDF5_PATH)
    if not expected_hdf.is_file():
        raise FileNotFoundError(expected_hdf)
    return source, assets, expected_hdf.stat()


def validate_upstream():
    """CPU-only lineage validation; never reads validation/test masks."""
    source, assets, hdf_stat = _source_and_assets()
    t_ref = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if t_ref.get("arm") != "T" or not np.array_equal(
            t_ref["ids"], np.asarray(sw143.contract.IMAGE_IDS, dtype=np.int64)):
        raise ValueError("SW0157 requires the validated SW0143-T fixed320 reference")
    source_sha = source["source_sha256"]
    source_manifest_sha = sha256_file(source["manifest_path"])
    source_ids_sha = train.ids_sha(source["image_ids"])
    arms = {}
    for arm in train.ARMS:
        output = train.TRAIN_ROOT / f"{arm}_seed1"
        if not (output / "manifest.json").is_file():
            raise FileNotFoundError(f"SW0157 waits for completed SW0155 {arm} training")
        task = {"arm": arm, "output": str(output)}
        if not artifactvalidator.validate_training_artifact(task, source=source, assets=assets):
            raise ValueError(f"completed SW0155 {arm} artifact failed its current validator")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arms[arm] = {
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
            "manifest": manifest,
        }
    return {
        "source_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_training_ids_sha256": source_ids_sha,
        "source_training_ids": source["image_ids"],
        "source_rows": np.asarray(source["rows"], dtype=np.int64).tolist(),
        "rgb_asset_validation": assets,
        "sw0143_T_prediction_sha256": t_ref["prediction_sha256"],
        "sw0143_T_protocol_sha256": t_ref["protocol_sha256"],
        "sw0155_arms": arms,
        "hdf5_path": str(Path(train.HDF5_PATH).resolve()),
        "hdf5_size_bytes": hdf_stat.st_size,
        "hdf5_mtime_ns": hdf_stat.st_mtime_ns,
    }


def event_diagnostics(component_membrane, component_spikes, component_gates, b, *, settle=SETTLE):
    """Summarize E=(M-threshold)>0 and verify native S=E*g without mutation."""
    if (not all(torch.is_tensor(value) for value in
                (component_membrane, component_spikes, component_gates, b))
            or component_membrane.ndim != 4 or component_membrane.shape[1] != 4
            or component_spikes.shape != component_membrane.shape
            or component_gates.shape != component_membrane.shape
            or tuple(b.shape) != (4,)
            or not isinstance(settle, int) or isinstance(settle, bool)
            or not 0 <= settle < component_membrane.shape[-1]):
        raise ValueError("event summary expects matching [B,4,N,T] traces and b[4]")
    if not all(torch.isfinite(value).all() for value in
               (component_membrane, component_spikes, component_gates, b)):
        raise FloatingPointError("event summary received nonfinite values")
    threshold = 0.06 * torch.exp(b.detach()).view(1, 4, 1, 1)
    margin = component_membrane.detach() - threshold
    events = margin > 0
    expected = events.to(component_gates.dtype) * component_gates.detach()
    shared_gate = torch.equal(component_gates.detach(), component_gates[:, :1].detach().expand_as(component_gates))
    spike_identity = torch.equal(component_spikes.detach(), expected)
    def summarize(start, stop):
        records = []
        for component in range(4):
            current = margin[:, component, :, start:stop].reshape(-1)
            event = events[:, component, :, start:stop]
            gate = component_gates[:, component, :, start:stop].detach()
            spikes = component_spikes[:, component, :, start:stop].detach()
            supported = gate > 0
            support_count = int(supported.sum().item())
            transition = (event[..., 1:] != event[..., :-1]).to(torch.float32)
            records.append({
                "margin_quantiles": {str(q): float(torch.quantile(current.float(), q).cpu())
                                     for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
                "margin_mean": float(current.mean().cpu()),
                "margin_std": float(current.float().std(unbiased=False).cpu()),
                "margin_rms": float(current.float().square().mean().sqrt().cpu()),
                "event_on_fraction": float(event.float().mean().cpu()),
                "event_transition_fraction": float(transition.mean().cpu()) if transition.numel() else 0.0,
                "gate_positive_support_count": support_count,
                "event_on_fraction_when_gate_positive": (
                    float(event[supported].float().mean().cpu()) if support_count else None),
                "weighted_spike_mean": float(spikes.mean().cpu()),
            })
        return records

    return {"full_window_per_component": summarize(0, component_membrane.shape[-1]),
            "settled_window_start": settle,
            "per_component": summarize(settle, component_membrane.shape[-1]),
            "shared_gate_exact": shared_gate,
            "emitted_spike_equals_event_times_gate_exact": spike_identity}


def _named_parameter_groups(foundation):
    core = foundation.wrapped.core
    groups = {name: [] for name in PARAMETER_GROUPS}
    encoder = foundation.encoder
    residual = getattr(encoder, "residual", None)
    base = getattr(encoder, "base_encoder", None)
    if residual is not None:
        groups["spatial_residual"] = list(residual.parameters())
    assigned = {id(p) for p in groups["spatial_residual"]}

    def add_module(group, module):
        if module is not None:
            parameters = ([module] if isinstance(module, torch.nn.Parameter)
                          else module.parameters())
            for parameter in parameters:
                if id(parameter) not in assigned:
                    groups[group].append(parameter)
                    assigned.add(id(parameter))

    add_module("graph", core.graph_generator)
    for module in (getattr(core, "gamma_to_drive", None),
                   getattr(core, "gamma_channel_proj", None),
                   getattr(core, "gamma_phase_gain", None)):
        add_module("gamma_drive", module)
    add_module("kuramoto", core.kuramoto)
    add_module("dendritic", core.dendric_layer)
    add_module("membrane", core.membrane_layer)
    integration = [foundation.wrapped.a_d, foundation.wrapped.a_m, foundation.wrapped.b]
    for parameter in integration:
        if id(parameter) not in assigned:
            groups["integration_tensors"].append(parameter)
            assigned.add(id(parameter))
    for parameter in core.parameters():
        if id(parameter) not in assigned:
            groups["other_core"].append(parameter)
            assigned.add(id(parameter))

    # Source RGB encoders are frozen; SW0152's exact-zero residual remains live.
    if base is not None:
        encoder.requires_grad_(True)
        base.requires_grad_(False)
        if residual is not None:
            residual.requires_grad_(True)
    else:
        encoder.requires_grad_(False)
    foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(True)
    foundation.encoder.eval()
    foundation.patcher.eval()
    foundation.wrapped.eval()
    frozen_base = list(base.parameters()) if base is not None else list(encoder.parameters())
    if any(parameter.requires_grad for parameter in frozen_base):
        raise AssertionError("registered source RGB base must remain frozen in this diagnostic")
    return groups, frozen_base


def _digest_foundation(foundation):
    state = {}
    for prefix, module in (("encoder", foundation.encoder), ("patcher", foundation.patcher),
                           ("wrapped", foundation.wrapped)):
        for name, value in module.state_dict().items():
            state[f"{prefix}.state.{name}"] = value
        for name, value in module.named_buffers():
            state[f"{prefix}.buffer.{name}"] = value
    return source_training.state_sha(state)


def _gradient_summary(loss, groups, *, extra=None, retain_graph=False):
    params = [parameter for values in groups.values() for parameter in values
              if parameter.requires_grad]
    extras = [extra] if extra is not None and extra.requires_grad else []
    requested = params + extras
    grads = torch.autograd.grad(loss, requested, allow_unused=True,
                                retain_graph=retain_graph) if requested else []
    report = {}
    cursor = 0
    for name, values in groups.items():
        selected = [parameter for parameter in values if parameter.requires_grad]
        local = grads[cursor:cursor + len(selected)]
        cursor += len(selected)
        squared, used = 0.0, 0
        finite = True
        for gradient in local:
            if gradient is None:
                continue
            used += 1
            finite &= bool(torch.isfinite(gradient).all().item())
            squared += float(gradient.detach().double().square().sum().cpu())
        report[name] = {"parameter_count": len(selected), "parameters_with_gradient": used,
                        "unused_parameter_count": len(selected) - used,
                        "finite": bool(finite), "l2_norm": math.sqrt(squared)}
    extra_report = None
    if extras:
        gradient = grads[-1]
        extra_report = {"finite": bool(torch.isfinite(gradient).all().item()),
                        "l2_norm": float(gradient.detach().double().norm().cpu())}
    return report, extra_report


def _pair_class_means(q, labels):
    rows, cols = torch.triu_indices(labels.shape[1], labels.shape[1], offset=1,
                                   device=labels.device)
    left, right = labels[:, rows], labels[:, cols]
    lf, rf = left > 0, right > 0
    masks = (lf & rf & (left == right), lf & rf & (left != right),
             lf ^ rf, (~lf) & (~rf))
    values = q[:, rows, cols]
    names = objective.PAIR_CLASSES
    return {name: {"count": int(mask.sum().item()),
                   "mean_q": (float(values[mask].mean().detach().cpu()) if mask.any() else None)}
            for name, mask in zip(names, masks)}


def _load_model(name, device, source, upstream):
    if name in ("source97", "source97_T"):
        foundation = train._load_foundation(device)
        if (foundation.provenance.get("source_core_sha256") != source["source_sha256"]
                or foundation.provenance.get("source_manifest_sha256") !=
                sha256_file(source["manifest_path"])
                or foundation.provenance.get("rgb_asset_validation") != upstream["rgb_asset_validation"]
                or list(map(int, foundation.image_ids)) != source["image_ids"]
                or not np.array_equal(foundation.pool_indices, source["rows"])):
            raise ValueError("fresh SW0097 foundation differs from current source contract")
        if name == "source97_T":
            geodesic_operator.install_operator(foundation.wrapped.core.graph_generator, "T")
        return foundation, {"model_role": name, "graph_operator":
                            "SW0143_T inference" if name == "source97_T" else "native SW0097 source"}
    arm = MODEL_ARMS[name]
    foundation, candidate_source, assets, manifest, manifest_sha = predict_evaluate._load_candidate(
        arm, device)
    if (candidate_source["source_sha256"] != source["source_sha256"]
            or assets != upstream["rgb_asset_validation"]):
        raise ValueError(f"SW0155 {arm} loaded against a different source/assets record")
    artifact = train.TRAIN_ROOT / f"{arm}_seed1"
    return foundation, {"model_role": name, "training_arm": arm,
        "graph_operator": "SW0155 completed checkpoint (native source graph recipe)",
        "training_manifest_sha256": manifest_sha,
        "training_checkpoint_sha256": sha256_file(artifact / "checkpoint.pt"),
        "training_implementation_fingerprint": manifest["implementation_fingerprint"]}


def _capture_rng(device):
    dev = torch.device(device)
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(dev).clone() if dev.type == "cuda" else None}


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], torch.device(device))


def _rng_digest(state):
    digest = hashlib.sha256(state["cpu"].cpu().numpy().tobytes())
    if state["cuda"] is not None:
        digest.update(state["cuda"].cpu().numpy().tobytes())
    return digest.hexdigest()


def _finite_number(value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    if minimum is not None and float(value) < minimum:
        return False
    if maximum is not None and float(value) > maximum:
        return False
    return True


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    """Strictly bind a completed diagnostic to current TRAIN/source artifacts."""
    try:
        path = Path(path)
        if expected_fingerprint is None:
            expected_fingerprint = implementation_fingerprint()
        if upstream is None:
            upstream = validate_upstream()
        record = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = [int(upstream["source_training_ids"][position]) for position in POSITIONS]
        if (record.get("experiment") != EXPERIMENT or record.get("status") != "complete"
                or record.get("seed") != SEED
                or record.get("implementation_fingerprint") != expected_fingerprint
                or record.get("source_order_sha256") != upstream["source_training_ids_sha256"]
                or record.get("source_training_ids") != expected_ids
                or record.get("source_positions") != list(POSITIONS)
                or record.get("batch_positions") != [list(batch) for batch in BATCH_POSITIONS]
                or record.get("steps") != STEPS or record.get("settle") != SETTLE
                or record.get("live_tail_steps") != LIVE_TAIL
                or record.get("optimizer_updates") != 0
                or record.get("ground_truth_used") is not True
                or record.get("ground_truth_scope") != "the eight registered source TRAIN masks only"
                or record.get("validation_or_test_ground_truth_used") is not False
                or record.get("rng", {}).get("caller_rng_restored") is not True
                or record.get("rng", {}).get("matched_seed_per_batch") !=
                    [ROLLOUT_SEED, ROLLOUT_SEED + 1]):
            return False
        report_upstream = record.get("upstream")
        expected_upstream = {key: value for key, value in upstream.items()
                             if key not in ("source_training_ids", "source_rows", "sw0155_arms")}
        if report_upstream != expected_upstream:
            return False
        lineages = record.get("sw0155_arm_lineage")
        if not isinstance(lineages, dict) or set(lineages) != set(train.ARMS):
            return False
        for arm in train.ARMS:
            expected = upstream["sw0155_arms"][arm]
            row = lineages[arm]
            if (row.get("manifest_sha256") != expected["manifest_sha256"]
                    or row.get("checkpoint_sha256") != expected["checkpoint_sha256"]
                    or row.get("implementation_fingerprint") !=
                        expected["manifest"].get("implementation_fingerprint")):
                return False
        models = record.get("models")
        if not isinstance(models, dict) or tuple(models) != MODEL_ORDER:
            return False
        for name in MODEL_ORDER:
            model = models[name]
            before, after = model.get("model_state_sha256_before"), model.get("model_state_sha256_after")
            if (not isinstance(before, str) or len(before) != 64 or before != after
                    or model.get("model_state_unchanged") is not True
                    or model.get("optimizer_updates") != 0
                    or not isinstance(model.get("frozen_rgb_base_sha256"), str)
                    or len(model["frozen_rgb_base_sha256"]) != 64):
                return False
            if len(model.get("batches", [])) != 2:
                return False
            for index, batch in enumerate(model["batches"]):
                ids = expected_ids[index * 4:(index + 1) * 4]
                if (batch.get("source_positions") != list(BATCH_POSITIONS[index])
                        or batch.get("image_ids") != ids
                        or batch.get("rng_seed") != ROLLOUT_SEED + index
                        or not _finite_number(batch.get("mean_actual_spike_relation_loss"))
                        or not _finite_number(batch.get("mean_membrane_proxy_relation_loss"))):
                    return False
                dynamics = batch.get("event_dynamics")
                if (not isinstance(dynamics, dict) or dynamics.get("shared_gate_exact") is not True
                        or dynamics.get("emitted_spike_equals_event_times_gate_exact") is not True
                        or len(dynamics.get("per_component", [])) != 4):
                    return False
                if dynamics.get("settled_window_start") != SETTLE:
                    return False
                for window_name in ("full_window_per_component", "per_component"):
                    window = dynamics.get(window_name)
                    if not isinstance(window, list) or len(window) != 4:
                        return False
                    for component in window:
                        quantiles = component.get("margin_quantiles")
                        if (not isinstance(quantiles, dict) or set(quantiles) !=
                                {"0.01", "0.1", "0.5", "0.9", "0.99"}
                                or any(not _finite_number(value) for value in quantiles.values())
                                or any(not _finite_number(component.get(key)) for key in
                                    ("margin_mean", "margin_std", "margin_rms", "event_on_fraction",
                                     "event_transition_fraction", "weighted_spike_mean"))
                                or not _finite_number(component.get("event_on_fraction"), minimum=0, maximum=1)
                                or not _finite_number(component.get("event_transition_fraction"), minimum=0, maximum=1)
                                or not isinstance(component.get("gate_positive_support_count"), int)):
                            return False
                        conditional = component.get("event_on_fraction_when_gate_positive")
                        if conditional is not None and not _finite_number(conditional, minimum=0, maximum=1):
                            return False
                for field in ("actual_spike_parameter_gradients", "membrane_proxy_parameter_gradients"):
                    gradients = batch.get(field)
                    if not isinstance(gradients, dict) or set(gradients) != set(PARAMETER_GROUPS):
                        return False
                    for group in gradients.values():
                        if (group.get("finite") is not True
                                or not _finite_number(group.get("l2_norm"), minimum=0)
                                or not isinstance(group.get("parameter_count"), int)
                                or not isinstance(group.get("parameters_with_gradient"), int)
                                or not isinstance(group.get("unused_parameter_count"), int)):
                            return False
                for field in ("actual_component_spikes_input_gradient",
                              "component_membrane_proxy_input_gradient"):
                    gradient = batch.get(field)
                    if (not isinstance(gradient, dict) or gradient.get("finite") is not True
                            or not _finite_number(gradient.get("l2_norm"), minimum=0)):
                        return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError,
            RuntimeError):
        return False


def run_diagnostic(*, device="cuda:0", output=OUTPUT, watchdog_seconds=WATCHDOG_SECONDS):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0157 diagnostic output: {output}")
    if not math.isfinite(float(watchdog_seconds)) or watchdog_seconds <= 0:
        raise ValueError("watchdog_seconds must be positive and finite")
    upstream = validate_upstream()
    source, assets, _ = _source_and_assets()
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0157 source diagnostic requires its assigned CUDA device")
    torch.cuda.set_device(dev)
    torch.cuda.init()
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered SW0097 TRAIN RGB cache schema changed")
    ordered_ids = source["image_ids"]
    ordered_rows = source["rows"]
    selected_ids = [int(ordered_ids[index]) for index in POSITIONS]
    if len(set(selected_ids)) != 8 or train.ids_sha(ordered_ids) != upstream["source_training_ids_sha256"]:
        raise ValueError("SW0157 selected IDs do not match immutable source order")
    rng_before = _capture_rng(dev)
    common_rng_seeds = [ROLLOUT_SEED, ROLLOUT_SEED + 1]
    model_results = {}
    started = time.monotonic()
    try:
        for model_index, name in enumerate(MODEL_ORDER):
            foundation, model_record = _load_model(name, dev, source, upstream)
            before_state = _digest_foundation(foundation)
            groups, frozen_base = _named_parameter_groups(foundation)
            model_record["model_state_sha256_before"] = before_state
            model_record["frozen_rgb_base_sha256"] = source_training.parameter_hash(frozen_base)
            model_record["batches"] = []
            for batch_index, positions in enumerate(BATCH_POSITIONS):
                if time.monotonic() - started > watchdog_seconds:
                    raise TimeoutError("SW0157 watchdog expired before next TRAIN rollout")
                rgb_np = np.asarray(cache[ordered_rows[list(positions)] ]).copy()
                rgb_cpu, masks_cpu = train.load_aligned_train_rows(
                    train.HDF5_PATH, ordered_ids, _RgbRowsView(cache, ordered_rows),
                    positions[0], 4)
                if not np.array_equal(rgb_cpu.numpy(), rgb_np):
                    raise ValueError("aligned TRAIN HDF5 images differ from registered RGB cache")
                rgb = rgb_cpu.to(dev)
                labels = objective.modal_train_targets32(masks_cpu).to(dev)
                seed_value = common_rng_seeds[batch_index]
                torch.random.default_generator.manual_seed(seed_value)
                torch.cuda.manual_seed(seed_value)
                torch.cuda.synchronize(dev)
                gamma = foundation.encode(rgb)
                if tuple(gamma.shape) != (4, 8, 1024) or not torch.isfinite(gamma).all():
                    raise ValueError("SW0157 encoder returned invalid native32 gamma")
                trace = foundation.rollout(gamma, total_steps=STEPS,
                                           live_tail_steps=LIVE_TAIL)
                torch.cuda.synchronize(dev)
                if (tuple(trace["component_membrane"].shape) != (4, 4, 1024, STEPS)
                        or tuple(trace["component_spikes"].shape) != (4, 4, 1024, STEPS)
                        or tuple(trace["component_gates"].shape) != (4, 4, 1024, STEPS)):
                    raise ValueError("SW0157 native rollout trace shapes changed")
                loss, details = objective.supervised_relation_loss(trace, labels, settle=SETTLE)
                if not torch.isfinite(loss):
                    raise FloatingPointError("SW0157 actual-spike TRAIN relation loss is nonfinite")
                actual_gradients, actual_input_gradient = _gradient_summary(
                    loss, groups, extra=trace["component_spikes"], retain_graph=True)
                if (not all(row["finite"] for row in actual_gradients.values())
                        or not actual_input_gradient["finite"]):
                    raise FloatingPointError("nonfinite actual-spike gradient in SW0157 diagnostic")
                membrane_trace = dict(trace, component_spikes=trace["component_membrane"])
                membrane_loss, membrane_details = objective.supervised_relation_loss(
                    membrane_trace, labels, settle=SETTLE)
                if not torch.isfinite(membrane_loss):
                    raise FloatingPointError("SW0157 membrane-proxy relation loss is nonfinite")
                membrane_gradients, membrane_input_gradient = _gradient_summary(
                    membrane_loss, groups, extra=trace["component_membrane"], retain_graph=False)
                if (not all(row["finite"] for row in membrane_gradients.values())
                        or not membrane_input_gradient["finite"]):
                    raise FloatingPointError("nonfinite membrane-proxy gradient in SW0157 diagnostic")
                dynamics = event_diagnostics(trace["component_membrane"],
                    trace["component_spikes"], trace["component_gates"], foundation.wrapped.b,
                    settle=SETTLE)
                if not dynamics["shared_gate_exact"] or not dynamics[
                        "emitted_spike_equals_event_times_gate_exact"]:
                    raise AssertionError("native component event/gate identity failed")
                model_record["batches"].append({
                    "source_positions": list(map(int, positions)),
                    "image_ids": selected_ids[batch_index * 4:(batch_index + 1) * 4],
                    "rng_seed": seed_value,
                    "mean_actual_spike_relation_loss": float(loss.detach().cpu()),
                    "mean_membrane_proxy_relation_loss": float(membrane_loss.detach().cpu()),
                    "actual_spike_relation_pair_summary": _pair_class_means(
                        details["q"].detach(), labels),
                    "membrane_proxy_relation_pair_summary": _pair_class_means(
                        membrane_details["q"].detach(), labels),
                    "actual_spike_parameter_gradients": actual_gradients,
                    "membrane_proxy_parameter_gradients": membrane_gradients,
                    "actual_component_spikes_input_gradient": actual_input_gradient,
                    "component_membrane_proxy_input_gradient": membrane_input_gradient,
                    "event_dynamics": dynamics,
                })
                del loss, details, membrane_loss, membrane_details, trace, membrane_trace
                del gamma, rgb, rgb_cpu, masks_cpu, labels, actual_gradients, membrane_gradients
                torch.cuda.empty_cache()
                if time.monotonic() - started > watchdog_seconds:
                    raise TimeoutError("SW0157 watchdog expired during bounded TRAIN diagnostic")
            after_state = _digest_foundation(foundation)
            model_record["model_state_sha256_after"] = after_state
            model_record["model_state_unchanged"] = before_state == after_state
            model_record["optimizer_updates"] = 0
            if before_state != after_state:
                raise AssertionError(f"SW0157 diagnostic mutated {name} model state")
            model_results[name] = model_record
            del foundation, groups, frozen_base
            torch.cuda.empty_cache()
    finally:
        _restore_rng(rng_before, dev)
    hdf_stat = Path(train.HDF5_PATH).stat()
    if (str(Path(train.HDF5_PATH).resolve()) != upstream["hdf5_path"]
            or hdf_stat.st_size != upstream["hdf5_size_bytes"]
            or hdf_stat.st_mtime_ns != upstream["hdf5_mtime_ns"]):
        raise ValueError("canonical HDF5 identity changed during SW0157 diagnostic")
    report = {
        "experiment": EXPERIMENT, "status": "complete", "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(),
        "upstream": {key: value for key, value in upstream.items()
                     if key not in ("source_training_ids", "source_rows", "sw0155_arms")},
        "sw0155_arm_lineage": {arm: {
            "manifest_sha256": value["manifest_sha256"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "implementation_fingerprint": value["manifest"]["implementation_fingerprint"]}
            for arm, value in upstream["sw0155_arms"].items()},
        "source_order_sha256": upstream["source_training_ids_sha256"],
        "source_training_ids": selected_ids,
        "source_positions": list(POSITIONS),
        "batch_positions": [list(batch) for batch in BATCH_POSITIONS],
        "models": model_results,
        "rng": {"matched_seed_per_batch": common_rng_seeds,
                "caller_rng_restored": _rng_digest(_capture_rng(dev)) == _rng_digest(rng_before)},
        "steps": STEPS, "settle": SETTLE, "live_tail_steps": LIVE_TAIL,
        "optimizer_updates": 0, "ground_truth_used": True,
        "ground_truth_scope": "the eight registered source TRAIN masks only",
        "validation_or_test_ground_truth_used": False,
        "watchdog_seconds": float(watchdog_seconds),
        "elapsed_seconds": time.monotonic() - started,
        "interpretation_limits": (
            "Eight TRAIN images are descriptive only. The membrane-relation pass is a proxy; "
            "gradient norms do not establish gate necessity, causality, or model performance."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return report


class _RgbRowsView:
    """Lazy RGB-cache row mapping so only the selected eight rows are copied."""
    def __init__(self, cache, source_rows):
        self.cache, self.source_rows = cache, np.asarray(source_rows, dtype=np.int64)

    def __getitem__(self, positions):
        return self.cache[self.source_rows[positions]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--watchdog-seconds", type=float, default=WATCHDOG_SECONDS)
    args = parser.parse_args(argv)
    report = run_diagnostic(device=args.device, output=args.output,
                            watchdog_seconds=args.watchdog_seconds)
    print(json.dumps({"status": report["status"], "models": list(report["models"]),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "ground_truth_scope": report["ground_truth_scope"]},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
