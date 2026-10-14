"""SW0160 actual-source, eight-TRAIN, two-update admission preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch
import h5py

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155
from collaborative_test.SW_0160_end_to_end_source_encoder import model

EXPERIMENT = "SW0160_end_to_end_source_encoder"
SEED = 1
POSITIONS = (0, 1, 2, 3, 2048, 2049, 2050, 2051)
STEPS, SETTLE, TAIL = 1024, 512, 64
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _state_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(np.asarray(tensor.shape, dtype="<i8").tobytes()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _params_sha(parameters):
    h = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        value = parameter.detach().cpu().contiguous()
        h.update(str(index).encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def _ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _feature_preprocess_sha(foundation):
    h = hashlib.sha256()
    for name, value in (("mean", foundation.feature_mean), ("std", foundation.feature_std)):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(tensor.numpy().tobytes())
    h.update(repr(float(foundation.feature_clip)).encode())
    return h.hexdigest()


def implementation_fingerprint():
    rels = [
        "collaborative_test/SW_0160_end_to_end_source_encoder/DESIGN_KO.md",
        "collaborative_test/SW_0160_end_to_end_source_encoder/model.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/preflight.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/deep_encoder.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
        "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json",
        "snn_kuramoto_bidirectional/graph_generator.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
    ]
    paths = [ROOT / item for item in rels]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("a frozen SW0160 dependency is missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _source_contract():
    ckpt, manifest_path, manifest, rows, ids, source_sha = sw130.source_contract(SEED)
    rows = np.asarray(rows, dtype=np.int64)
    ids = [int(v) for v in ids]
    if len(ids) != 4096 or len(set(ids)) != 4096 or rows.shape != (4096,) or len(np.unique(rows)) != 4096:
        raise ValueError("SW0160 requires exact registered source1 ordered TRAIN contract")
    return {"checkpoint": Path(ckpt), "manifest_path": Path(manifest_path), "manifest": manifest,
            "rows": rows, "ids": ids, "source_sha256": source_sha}


def validate_upstream():
    """Resolve current source/assets/HDF/parity and the exact eight RGB rows on CPU."""
    src = _source_contract()
    assets = sw130.validate_rgb_assets()
    manifest_sha = sha256_file(src["manifest_path"])
    saved_parity, parity_sha = sw150.load_and_validate_parity(
        sw150.PARITY_PATH, source_sha=src["source_sha256"],
        source_manifest_sha=manifest_sha, training_ids=src["ids"], asset_validation=assets)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    rgb8 = np.asarray(cache[src["rows"][list(POSITIONS)]], dtype=np.uint8).copy()
    if rgb8.shape != (8, 128, 128, 3):
        raise ValueError("registered SW0160 eight-image RGB selection has invalid shape")
    hdf_stat = Path(sw155.HDF5_PATH).stat()
    with h5py.File(sw155.HDF5_PATH, "r") as hdf:
        images_shape, masks_shape = list(hdf["image"].shape), list(hdf["mask"].shape)
    if images_shape != [100000, 128, 128, 3] or masks_shape not in (
            [100000, 128, 128], [100000, 128, 128, 1]):
        raise ValueError("canonical supervised HDF5 identity/schema changed")
    binding = {"seed": SEED, "checkpoint_path": str(src["checkpoint"]),
        "checkpoint_sha256": src["source_sha256"], "manifest_path": str(src["manifest_path"]),
        "manifest_sha256": manifest_sha, "ordered_train_ids": src["ids"],
        "ordered_train_ids_sha256": _ids_sha(src["ids"]), "source_positions": list(POSITIONS),
        "selected_image_ids": [src["ids"][i] for i in POSITIONS],
        "rgb_asset_validation": assets, "rgb_batch_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
        "saved_parity_sha256": parity_sha, "saved_parity_status": saved_parity.get("status"),
        "hdf5_identity": {"path": str(Path(sw155.HDF5_PATH).resolve()),
            "size_bytes": hdf_stat.st_size, "mtime_ns": hdf_stat.st_mtime_ns,
            "images_shape": images_shape, "masks_shape": masks_shape}}
    return binding


def _load_source(device):
    device = torch.device(device)
    index = device.index if device.type == "cuda" else None
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        torch.random.default_generator.manual_seed(149001)
        if index is not None:
            torch.cuda.manual_seed(149001)
        return sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)


def _trace(foundation, rgb, *, live_tail=TAIL):
    gamma = foundation.encode(rgb)
    trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=live_tail)
    for name in sw143.TRACE_FIELDS:
        if name not in trace or not torch.isfinite(trace[name]).all():
            raise FloatingPointError(f"nonfinite/missing SW0160 trace field: {name}")
    return gamma, trace


def _q_and_labels(trace):
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                  settle=SETTLE, affinity_mode="spike")
    return q, sw136._qcc_labels(trace)


def _family_norm(params):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu())
                         for p in params if p.grad is not None))


def _feature_clip_fraction(foundation, rgb_uint8):
    with torch.no_grad():
        rgb = rgb_uint8.to(next(foundation.encoder.parameters()).device).permute(0, 3, 1, 2).float() / 255.
        feature = foundation.encoder(rgb)
        normalized = (feature - foundation.feature_mean) / foundation.feature_std
        return float((normalized.abs() > foundation.feature_clip).float().mean().cpu())


def _event_diagnostics(trace, wrapped):
    from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp
    threshold = (0.06 * torch.exp(wrapped.b.detach())).view(1, 4, 1, 1)
    membrane = trace["component_membrane"][..., SETTLE:]
    gates = trace["component_gates"][..., SETTLE:]
    events = act_fun_adp(membrane - threshold)
    support = gates > 0
    transitions = (events[..., 1:] != events[..., :-1]) & support[..., 1:]
    spikes = trace["component_spikes"][..., SETTLE:]
    if not torch.equal(spikes, events * gates):
        raise AssertionError("SW0160 report trace violates native emitted-spike contract")
    return {"settled_event_on_fraction": float(events.float().mean().cpu()),
        "settled_gate_positive_count": int(support.sum().item()),
        "settled_event_on_fraction_when_gate_positive": (
            float(events[support].float().mean().cpu()) if bool(support.any()) else None),
        "settled_gate_positive_event_transitions": int(transitions.sum().item()),
        "per_component_settled_event_on_fraction": [
            float(events[:, c].float().mean().cpu()) for c in range(4)]}


def run_preflight(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve existing SW0160 preflight output")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0160 actual-source preflight requires assigned CUDA")
    torch.cuda.set_device(dev)
    src = _source_contract()
    assets = sw130.validate_rgb_assets()
    parity, parity_sha = sw150.load_and_validate_parity(
        sw150.PARITY_PATH, source_sha=src["source_sha256"],
        source_manifest_sha=sha256_file(src["manifest_path"]), training_ids=src["ids"],
        asset_validation=assets)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    rgb8 = np.asarray(cache[src["rows"][list(POSITIONS)]], dtype=np.uint8).copy()
    if rgb8.shape != (8, 128, 128, 3):
        raise ValueError("registered SW0160 eight-image RGB selection has invalid shape")
    aligned, masks = sw155.load_aligned_train_rows(sw155.HDF5_PATH,
        [src["ids"][i] for i in POSITIONS], rgb8, 0, len(POSITIONS))
    if aligned.dtype != torch.uint8 or not np.array_equal(aligned.cpu().numpy(), rgb8):
        raise ValueError("canonical TRAIN RGB/HDF5 identity check failed")
    if tuple(masks.shape) != (8, 128, 128):
        raise ValueError("canonical TRAIN masks have unexpected shape")

    source = _load_source(dev)
    if (source.provenance.get("source_core_sha256") != src["source_sha256"]
            or source.provenance.get("rgb_asset_validation") != assets
            or list(map(int, source.image_ids)) != src["ids"]
            or not np.array_equal(source.pool_indices, src["rows"])):
        raise ValueError("loaded SW0097 source differs from registered source/assets/order")
    source.encoder.eval(); source.patcher.eval(); source.wrapped.eval()
    geodesic_operator.install_operator(source.wrapped.core.graph_generator, "T")
    source.encoder.requires_grad_(False)
    first_rgb = torch.from_numpy(rgb8[:4]).to(dev)
    with torch.no_grad():
        source_gamma, source_trace = _trace(source, first_rgb, live_tail=0)
        source_q, source_qcc = _q_and_labels(source_trace)
    source_encoder_sha = _state_sha(source.encoder)
    source_preprocess_sha = _feature_preprocess_sha(source)
    source_patch_sha = _state_sha(source.patcher)
    hdf_stat = Path(sw155.HDF5_PATH).stat()
    with h5py.File(sw155.HDF5_PATH, "r") as hdf:
        hdf_image_shape = list(hdf["image"].shape)
        hdf_mask_shape = list(hdf["mask"].shape)
    if (hdf_image_shape != [100000, 128, 128, 3]
            or hdf_mask_shape not in ([100000, 128, 128], [100000, 128, 128, 1])):
        raise ValueError("canonical supervised HDF5 identity/schema changed")
    upstream = validate_upstream()
    source_initial_trace = {key: source_trace[key].detach().cpu() for key in sw143.TRACE_FIELDS}
    source_initial_gamma = source_gamma.detach().cpu()
    source_initial_q = source_q.detach().cpu()
    source_initial_qcc = np.asarray(source_qcc).copy()
    source_clip_fraction = _feature_clip_fraction(source, torch.from_numpy(rgb8))
    del source, source_trace, source_gamma, source_q

    arm_reports = {}
    for arm in model.ARMS:
        foundation = _load_source(dev)
        foundation.encoder.eval(); foundation.patcher.eval(); foundation.wrapped.eval()
        geodesic_operator.install_operator(foundation.wrapped.core.graph_generator, "T")
        if _state_sha(foundation.encoder) != source_encoder_sha:
            raise ValueError("fresh arm source encoder initialization differs from source")
        model.install_encoder(foundation, arm, residual_seed=160001)
        foundation.encoder.to(dev)
        foundation.patcher.eval(); foundation.wrapped.train()
        # Effective geodesic contrast is fixed by the T operator, not trained.
        graph = foundation.wrapped.core.graph_generator
        graph.geodesic_contrast.requires_grad_(False)
        base_before = _state_sha(foundation.encoder.base_encoder)
        patcher_before = _state_sha(foundation.patcher)
        with torch.no_grad():
            gamma0, trace0 = _trace(foundation, first_rgb, live_tail=0)
            q0, qcc0 = _q_and_labels(trace0)
        parity_fields = {key: bool(torch.equal(source_initial_trace[key], trace0[key].detach().cpu()))
                         for key in sw143.TRACE_FIELDS}
        parity_exact = (all(parity_fields.values())
            and torch.equal(source_initial_gamma, gamma0.detach().cpu())
            and torch.equal(source_initial_q, q0.detach().cpu())
            and np.array_equal(source_initial_qcc, qcc0))
        if not parity_exact:
            raise AssertionError(f"SW0160 {arm} zero-residual source-T parity failed")
        initial_event_stats = _event_diagnostics(trace0, foundation.wrapped)
        del gamma0, trace0, q0

        families = model.trainable_families(foundation, arm)
        family_initial_sha = {name: _params_sha(params) for name, params in families.items()}
        optimizer = model.make_optimizer(families)
        labels = objective.modal_train_targets32(masks).to(dev)
        update_records = []
        for update in range(2):
            start = update * 4
            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            for micro in range(4):
                rgb = torch.from_numpy(rgb8[start+micro:start+micro+1]).to(dev)
                _gamma, trace = _trace(foundation, rgb)
                q, rho = objective.production_q_and_signed_rho(trace, settle=SETTLE)
                loss, _details = objective.balanced_relation_objective(
                    q, rho, labels[start+micro:start+micro+1])
                if not torch.isfinite(loss):
                    raise FloatingPointError("SW0160 TRAIN relation loss is nonfinite")
                (loss / 4).backward()
                loss_values.append(float(loss.detach().cpu()))
                del _gamma, trace, loss, _details
            norms = {name: _family_norm(params) for name, params in families.items()}
            hidden_residual = list(foundation.encoder.residual[:-1].parameters())
            hidden_residual_norm = _family_norm(hidden_residual)
            if any(not math.isfinite(value) or value < 0 for value in norms.values()):
                raise FloatingPointError("SW0160 family gradient norm is invalid")
            active = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(active, 1.0)
            snapshots = {name: [p.detach().clone() for p in params]
                         for name, params in families.items()}
            optimizer.step()
            with torch.no_grad():
                foundation.wrapped.project_integrations_()
            changes = {}
            for name, params in families.items():
                changes[name] = math.sqrt(sum(float((p.detach() - old).double().square().sum().cpu())
                    for p, old in zip(params, snapshots[name]))) if params else 0.0
            if arm == "frozen_base_joint" and _state_sha(foundation.encoder.base_encoder) != base_before:
                raise AssertionError("SW0160 frozen-base control changed source encoder weights")
            hidden_change = math.sqrt(sum(float((p.detach() - old).double().square().sum().cpu())
                for p, old in zip(hidden_residual, snapshots["spatial_residual"][:len(hidden_residual)])))
            update_records.append({"update": update + 1, "mean_loss": float(np.mean(loss_values)),
                "family_gradient_norms": norms, "family_parameter_change_l2": changes,
                "residual_hidden_gradient_norm": hidden_residual_norm,
                "residual_hidden_parameter_change_l2": hidden_change})
        base_after = _state_sha(foundation.encoder.base_encoder)
        patcher_after = _state_sha(foundation.patcher)
        feature_clip = _feature_clip_fraction(foundation, torch.from_numpy(rgb8))
        with torch.no_grad():
            _gamma_final, trace_final = _trace(foundation, first_rgb, live_tail=0)
            final_event_stats = _event_diagnostics(trace_final, foundation.wrapped)
        del _gamma_final, trace_final
        if patcher_after != patcher_before or patcher_before != source_patch_sha:
            raise AssertionError("SW0160 fixed source patcher/statistics changed")
        arm_reports[arm] = {"initial_zero_residual_parity": {"exact": parity_exact,
                "trace_fields": parity_fields, "gamma_exact": True, "production_Q_exact": True,
                "production_QCC_exact": True},
            "base_encoder_sha256_before": base_before, "base_encoder_sha256_after": base_after,
            "base_encoder_changed": base_before != base_after,
            "initial_trainable_family_sha256": family_initial_sha,
            "feature_clip_fraction_after_updates": feature_clip,
            "source_feature_clip_fraction": source_clip_fraction,
            "patcher_sha256_before_after": [patcher_before, patcher_after],
            "feature_preprocess_sha256": _feature_preprocess_sha(foundation),
            "settled_event_diagnostics_before_updates": initial_event_stats,
            "settled_event_diagnostics_after_updates": final_event_stats,
            "two_updates": update_records,
            "base_encoder_parameters": sum(p.numel() for p in foundation.encoder.base_encoder.parameters()),
            "frozen_effective_geodesic_contrast": True}
        del foundation, optimizer

    candidate = arm_reports["full_encoder_joint"]
    control = arm_reports["frozen_base_joint"]
    for family in ("spatial_residual", "graph", "native_core", "integration_tensors"):
        if (candidate["initial_trainable_family_sha256"][family]
                != control["initial_trainable_family_sha256"][family]):
            raise AssertionError(f"paired SW0160 arms differ initially in {family}")
    reasons = []
    common = ("spatial_residual", "graph", "native_core", "integration_tensors")
    for arm, record in arm_reports.items():
        for row in record["two_updates"]:
            for family in common:
                if row["family_gradient_norms"][family] <= 0:
                    reasons.append(f"{arm}_update_{row['update']}_{family}_gradient_zero")
                if row["family_parameter_change_l2"][family] <= 0:
                    reasons.append(f"{arm}_update_{row['update']}_{family}_unchanged")
    for row in candidate["two_updates"]:
        if row["family_gradient_norms"]["source_base"] <= 0:
            reasons.append(f"candidate_update_{row['update']}_source_base_gradient_zero")
        if row["family_parameter_change_l2"]["source_base"] <= 0:
            reasons.append(f"candidate_update_{row['update']}_source_base_unchanged")
    if candidate["base_encoder_sha256_before"] == candidate["base_encoder_sha256_after"]:
        reasons.append("candidate_source_base_unchanged_after_two_updates")
    for arm, record in arm_reports.items():
        if (record["two_updates"][1]["residual_hidden_gradient_norm"] <= 0
                or record["two_updates"][1]["residual_hidden_parameter_change_l2"] <= 0):
            reasons.append(f"{arm}_residual_hidden_credit_missing_on_update_2")
    if control["base_encoder_sha256_before"] != control["base_encoder_sha256_after"]:
        reasons.append("control_source_base_changed")
    for arm, record in arm_reports.items():
        for row in record["two_updates"]:
            if any(not math.isfinite(v) or v < 0 for v in row["family_gradient_norms"].values()):
                raise FloatingPointError(f"nonfinite gradient in {arm}")
    report = {"experiment": EXPERIMENT, "status": "complete", "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(),
        "upstream": upstream,
        "source": {"checkpoint_sha256": src["source_sha256"],
            "manifest_sha256": sha256_file(src["manifest_path"]),
            "ordered_train_ids": src["ids"], "ordered_train_ids_sha256": _ids_sha(src["ids"]),
            "source_positions": list(POSITIONS), "selected_image_ids": [src["ids"][i] for i in POSITIONS],
            "rgb_asset_validation": assets, "rgb_batch_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
            "hdf5_identity": {"path": str(Path(sw155.HDF5_PATH).resolve()),
                "size_bytes": hdf_stat.st_size, "mtime_ns": hdf_stat.st_mtime_ns,
                "images_shape": hdf_image_shape, "masks_shape": hdf_mask_shape},
            "source_patch_sha256": source_patch_sha,
            "feature_preprocess_sha256": source_preprocess_sha,
            "source_base_encoder_sha256": source_encoder_sha,
            "saved_parity_sha256": parity_sha, "saved_parity_status": parity.get("status")},
        "registered": {"arms": list(model.ARMS), "optimizer_updates_per_arm": 2,
            "preflight_images": 8, "logical_batch": 4, "microbatch": 1,
            "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
            "learning_rates": model.LEARNING_RATES,
            "ground_truth_scope": "the eight registered source TRAIN masks only"},
        "arms": arm_reports, "training_admission": not reasons,
        "admission_reasons": reasons,
        "ground_truth_used": True, "ground_truth_scope": "the eight registered source TRAIN masks only",
        "validation_or_test_ground_truth_used": False,
        "interpretation_limits": ["eight TRAIN images and two disposable updates per arm",
            "zero-output residual establishes initialization parity only",
            "no validation performance or final model quality claim"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    if not validate_report(output, expected_fingerprint=report["implementation_fingerprint"],
                           upstream=upstream):
        raise ValueError("SW0160 written preflight failed its strict validator")
    return report


def validate_report_record(report, *, expected_fingerprint=None, expected_upstream=None):
    try:
        if (report.get("experiment") != EXPERIMENT or report.get("status") != "complete"
                or report.get("seed") != SEED or report.get("ground_truth_used") is not True
                or report.get("ground_truth_scope") != "the eight registered source TRAIN masks only"
                or report.get("validation_or_test_ground_truth_used") is not False
                or type(report.get("training_admission")) is not bool
                or not isinstance(report.get("admission_reasons"), list)):
            return False
        if expected_fingerprint is not None and report.get("implementation_fingerprint") != expected_fingerprint:
            return False
        if expected_upstream is not None and report.get("upstream") != expected_upstream:
            return False
        source = report["source"]
        registered = report["registered"]
        if (registered.get("arms") != list(model.ARMS)
                or registered.get("optimizer_updates_per_arm") != 2
                or registered.get("preflight_images") != 8
                or registered.get("logical_batch") != 4 or registered.get("microbatch") != 1
                or registered.get("steps") != STEPS or registered.get("settle") != SETTLE
                or registered.get("live_tail") != TAIL
                or registered.get("learning_rates") != model.LEARNING_RATES
                or registered.get("ground_truth_scope") != "the eight registered source TRAIN masks only"):
            return False
        if (len(source["ordered_train_ids"]) != 4096 or len(set(source["ordered_train_ids"])) != 4096
                or source["source_positions"] != list(POSITIONS)
                or source["selected_image_ids"] != [source["ordered_train_ids"][i] for i in POSITIONS]
                or any(len(str(source[key])) != 64 for key in ("checkpoint_sha256", "manifest_sha256",
                      "ordered_train_ids_sha256", "rgb_batch_sha256", "saved_parity_sha256",
                      "source_patch_sha256", "source_base_encoder_sha256"))):
            return False
        if len(source.get("feature_preprocess_sha256", "")) != 64:
            return False
        if _ids_sha(source["ordered_train_ids"]) != source["ordered_train_ids_sha256"]:
            return False
        if report.get("upstream", {}).get("ordered_train_ids") != source["ordered_train_ids"]:
            return False
        upstream_record = report.get("upstream", {})
        for upstream_key, source_key in (("checkpoint_sha256", "checkpoint_sha256"),
                ("manifest_sha256", "manifest_sha256"),
                ("ordered_train_ids_sha256", "ordered_train_ids_sha256"),
                ("selected_image_ids", "selected_image_ids"),
                ("rgb_batch_sha256", "rgb_batch_sha256"),
                ("saved_parity_sha256", "saved_parity_sha256"),
                ("rgb_asset_validation", "rgb_asset_validation"),
                ("hdf5_identity", "hdf5_identity")):
            if upstream_record.get(upstream_key) != source.get(source_key):
                return False
        hdf = source["hdf5_identity"]
        if (not hdf["path"] or hdf["size_bytes"] <= 0 or hdf["mtime_ns"] <= 0
                or hdf["images_shape"] != [100000, 128, 128, 3]
                or hdf["masks_shape"] not in ([100000, 128, 128], [100000, 128, 128, 1])
                or len(source["source_patch_sha256"]) != 64):
            return False
        arms = report["arms"]
        if set(arms) != set(model.ARMS):
            return False
        for arm in model.ARMS:
            rec = arms[arm]
            parity = rec["initial_zero_residual_parity"]
            if (parity.get("exact") is not True or parity.get("gamma_exact") is not True
                    or parity.get("production_Q_exact") is not True or parity.get("production_QCC_exact") is not True
                    or set(parity.get("trace_fields", {})) != set(sw143.TRACE_FIELDS)
                    or not all(parity["trace_fields"].values())):
                return False
            if (len(rec.get("two_updates", [])) != 2
                    or set(rec.get("initial_trainable_family_sha256", {})) != set(model.LEARNING_RATES)
                    or any(len(str(rec.get(key, ""))) != 64 for key in
                           ("base_encoder_sha256_before", "base_encoder_sha256_after",
                            "feature_preprocess_sha256"))
                    or rec.get("patcher_sha256_before_after", [None, None])[0]
                       != rec.get("patcher_sha256_before_after", [None, None])[1]
                    or rec.get("patcher_sha256_before_after", [None, None])[0]
                       != source["source_patch_sha256"]
                    or rec["feature_preprocess_sha256"] != source["feature_preprocess_sha256"]
                    or rec.get("frozen_effective_geodesic_contrast") is not True):
                return False
            if arm == "frozen_base_joint" and rec["base_encoder_sha256_before"] != rec["base_encoder_sha256_after"]:
                return False
            if rec.get("base_encoder_changed") is not (
                    rec["base_encoder_sha256_before"] != rec["base_encoder_sha256_after"]):
                return False
            for ix, row in enumerate(rec["two_updates"], 1):
                if row.get("update") != ix or not math.isfinite(float(row["mean_loss"])):
                    return False
                grads, changes = row["family_gradient_norms"], row["family_parameter_change_l2"]
                if set(grads) != set(model.LEARNING_RATES) or set(changes) != set(model.LEARNING_RATES):
                    return False
                if any(not math.isfinite(float(v)) or float(v) < 0 for v in (*grads.values(), *changes.values())):
                    return False
                for key in ("residual_hidden_gradient_norm", "residual_hidden_parameter_change_l2"):
                    if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                        return False
            for key in ("feature_clip_fraction_after_updates", "source_feature_clip_fraction"):
                if not math.isfinite(float(rec[key])) or not 0 <= rec[key] <= 1:
                    return False
            for phase in ("settled_event_diagnostics_before_updates", "settled_event_diagnostics_after_updates"):
                event = rec[phase]
                fraction = float(event["settled_event_on_fraction"])
                if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                    return False
                support_fraction = event.get("settled_event_on_fraction_when_gate_positive")
                if support_fraction is not None and (not math.isfinite(float(support_fraction)) or not 0 <= support_fraction <= 1):
                    return False
                if not isinstance(event.get("settled_gate_positive_count"), int) or event["settled_gate_positive_count"] < 0:
                    return False
                if not isinstance(event.get("settled_gate_positive_event_transitions"), int) or event["settled_gate_positive_event_transitions"] < 0:
                    return False
        cand = arms["full_encoder_joint"]
        ctrl = arms["frozen_base_joint"]
        if cand["base_encoder_sha256_before"] != source["source_base_encoder_sha256"]:
            return False
        if ctrl["base_encoder_sha256_before"] != source["source_base_encoder_sha256"]:
            return False
        for family in ("spatial_residual", "graph", "native_core", "integration_tensors"):
            if (cand["initial_trainable_family_sha256"][family]
                    != arms["frozen_base_joint"]["initial_trainable_family_sha256"][family]):
                return False
        reasons = []
        common = ("spatial_residual", "graph", "native_core", "integration_tensors")
        for arm, rec in arms.items():
            for row in rec["two_updates"]:
                for family in common:
                    if row["family_gradient_norms"][family] <= 0:
                        reasons.append(f"{arm}_update_{row['update']}_{family}_gradient_zero")
                    if row["family_parameter_change_l2"][family] <= 0:
                        reasons.append(f"{arm}_update_{row['update']}_{family}_unchanged")
        for row in cand["two_updates"]:
            if row["family_gradient_norms"]["source_base"] <= 0:
                reasons.append(f"candidate_update_{row['update']}_source_base_gradient_zero")
            if row["family_parameter_change_l2"]["source_base"] <= 0:
                reasons.append(f"candidate_update_{row['update']}_source_base_unchanged")
        if cand["base_encoder_sha256_before"] == cand["base_encoder_sha256_after"]:
            reasons.append("candidate_source_base_unchanged_after_two_updates")
        for arm, rec in arms.items():
            row = rec["two_updates"][1]
            if row["residual_hidden_gradient_norm"] <= 0 or row["residual_hidden_parameter_change_l2"] <= 0:
                reasons.append(f"{arm}_residual_hidden_credit_missing_on_update_2")
        if ctrl["base_encoder_sha256_before"] != ctrl["base_encoder_sha256_after"]:
            reasons.append("control_source_base_changed")
        return (report["admission_reasons"] == reasons
                and report["training_admission"] is (not reasons))
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return False


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        current = validate_upstream() if upstream is None else upstream
        if data.get("upstream") != current:
            return False
        return validate_report_record(data,
            expected_fingerprint=expected_fingerprint or implementation_fingerprint(), expected_upstream=current)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": report["status"], "training_admission": report["training_admission"],
                      "arms": list(report["arms"]), "images": len(report["source"]["selected_image_ids"])}), flush=True)
    return 0 if report["training_admission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
