"""Conditional seed-1 paired training for SW0126 (local implementation)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0126_history_event_binding"
SEED = 1
ARM_NAMES = ("history_event", "gate_only")
BATCH = 16
UPDATES = 256
CORE_LR = 3e-5
HEAD_LR = 1e-3
CLIP_NORM = 1.0
STEPS, SETTLE, TAIL = 1024, 512, 64
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0126_history_event_binding import screen, screen_queue
from collaborative_test.SW_0126_history_event_binding.binder import TemporalSlotRGBBinder
from collaborative_test.SW_0126_history_event_binding.history_event import (
    attach_history_event_membrane, head_trace_for_arm,
)
from collaborative_test.SW_0126_history_event_binding.pilot_rollout import rollout


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def implementation_fingerprint():
    paths = [HERE / "pilot_run.py", HERE / "pilot_rollout.py", HERE / "binder.py",
             HERE / "history_event.py", HERE / "training_protocol.json",
             HERE / "screen.py", HERE / "screen_queue.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
             ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             sw117.RUNNER]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def _screen_evidence():
    reports = {}
    for task in screen_queue.task_plan():
        if not screen_queue.valid_result(task):
            raise AssertionError(f"all three current SW0126 screens must pass before training: {task['task_id']}")
        path = Path(task["output"])
        reports[str(task["seed"])] = {"path": str(path), "sha256": sha(path)}
    return reports


def load_assets(seed, device):
    if seed != SEED:
        raise ValueError("SW0126 paired pilot is preregistered for seed 1 only")
    result = screen.source_and_cache_contract(seed)
    source, source_manifest, source_meta, ids, rows = result[:5]
    gamma_cache, gamma_meta, rgb_cache, rgb_meta, rgb_cache_sha = result[5:]
    if (len(ids) != 4096 or len(rows) != 4096 or len(np.unique(ids)) != 4096
            or not np.array_equal(np.asarray(source_meta["training_ids"]), ids)):
        raise AssertionError("pilot requires the exact registered SW0097 seed-1 4096-image order")
    if (sha(source) != base.EXPECTED_SOURCE_SHAS[seed]
            or sha(sw117.ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(sw117.STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered source/encoder/preprocessing hash mismatch")
    encoder = sw117.load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
        kernel_size=3, channels=3, device=device)
    encoder.eval().requires_grad_(False)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device).eval()
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    core = base.make_core(device, steps=64)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    if (core.graph_generator is None or core.graph_generator.uses_feedback
            or core.osc_dim != 4 or core.kuramoto.spike_pulse_gain is not None):
        raise AssertionError("source core is outside the registered static D4 no-pulse contract")
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    adapter = attach_history_event_membrane(core, capture_gate_trace=True)
    for name, parameter in core.named_parameters():
        parameter.requires_grad_(not name.startswith("graph_generator."))
    core.eval()
    binder = TemporalSlotRGBBinder(init_seed=1261).to(device)
    return {"source": source, "source_manifest": source_manifest,
            "source_meta": source_meta, "ids": np.asarray(ids), "rows": np.asarray(rows),
            "gamma_cache": gamma_cache, "gamma_meta": gamma_meta,
            "rgb_cache": rgb_cache, "rgb_meta": rgb_meta,
            "rgb_cache_sha": rgb_cache_sha, "encoder": encoder, "patcher": patcher,
            "mean": mean, "std": std, "clip": clip, "core": core,
            "adapter": adapter, "binder": binder}


def _read_batch(assets, start, device):
    rows = assets["rows"][start:start+BATCH]
    ids = assets["ids"][start:start+BATCH]
    image = sw117.read_rgb(assets["rgb_cache"], rows, device)
    with torch.no_grad():
        gamma = sw117.encode_rgb(assets["encoder"], assets["patcher"], assets["mean"],
                                 assets["std"], assets["clip"], image)
    cached = assets["gamma_cache"][torch.as_tensor(rows, dtype=torch.long)].to(device)
    gamma_delta = float((gamma - cached).abs().max().detach().cpu())
    if not math.isfinite(gamma_delta) or gamma_delta > 2e-5:
        raise AssertionError(f"live RGB gamma differs from the registered cache by {gamma_delta}")
    target = image.permute(0, 2, 3, 1).contiguous() / 255.0
    return image, target, gamma, gamma_delta, [int(v) for v in ids]


def _loss_for_batch(assets, gamma, target, arm):
    trace = rollout(assets["core"], gamma, total_steps=STEPS,
                    settle=SETTLE, live_tail_steps=TAIL)
    events = assets["adapter"].component_event_trace(gamma.shape[0])[..., -SETTLE:]
    gates = assets["adapter"].component_gate_trace(gamma.shape[0])[..., -SETTLE:]
    spikes = trace["component_spikes"][..., -SETTLE:]
    if not torch.equal(spikes, gates * events):
        raise AssertionError("actual event spikes must equal the captured gate×event traces")
    binder_input = binder_trace_for_arm(spikes, gates, arm)
    result = assets["binder"](binder_input)
    reconstruction = result["reconstruction"]
    if not torch.isfinite(reconstruction).all() or reconstruction.shape != target.shape:
        raise AssertionError("pilot RGB reconstruction is nonfinite or has wrong shape")
    loss = native_rgb_objective(reconstruction, target)
    return loss, trace, result, events, gates


def binder_trace_for_arm(emitted_spikes, gates, arm):
    """Route actual emitted S=g*e to history_event, continuous g to control."""
    if emitted_spikes.shape != gates.shape:
        raise ValueError("emitted spike and gate traces must have matching shapes")
    return head_trace_for_arm(emitted_spikes, gates, arm)


def native_rgb_objective(reconstruction, target):
    """Registered pixelwise native RGB MSE; deliberately no auxiliary terms."""
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("native RGB objective requires matching [B,H,W,C] tensors")
    return F.mse_loss(reconstruction, target)


def _trainable(core):
    core_params = [p for n, p in core.named_parameters()
                   if p.requires_grad and not n.startswith("graph_generator.")]
    if not core_params:
        raise AssertionError("pilot core optimizer has no trainable nongraph parameters")
    return core_params


def _gradient_summary(params):
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    if not grads:
        return {"norm": 0.0, "finite": True, "nonzero_tensors": 0}
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    norm = float(torch.stack([g.norm() for g in grads]).norm().detach().cpu())
    return {"norm": norm, "finite": finite,
            "nonzero_tensors": sum(int(bool(torch.count_nonzero(g))) for g in grads)}


def _core_parameter_families(core):
    named = [(name, parameter) for name, parameter in core.named_parameters()
             if parameter.requires_grad and not name.startswith("graph_generator.")]
    drive = [(n, p) for n, p in named if "gamma_channel_proj" in n or "gamma_phase_gain" in n]
    beta = [(n, p) for n, p in named if n.endswith("beta_logits")]
    membrane = [(n, p) for n, p in named if "membrane_layer.tau_m" in n]
    dendrite = [(n, p) for n, p in named if n.startswith("dendric_layer.")]
    if not all((drive, beta, membrane, dendrite)):
        raise AssertionError("source core lacks a registered drive/beta/membrane/dendrite parameter family")
    return {"oscillator_drive": drive, "beta": beta, "membrane": membrane,
            "dendrite": dendrite, "all": named}


def _named_family_gradients(groups):
    result = {}
    for family, named in groups.items():
        params = [parameter for _, parameter in named]
        grads = [parameter.grad for parameter in params]
        present = [g for g in grads if g is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in present)
        norm = _gradient_summary(params)["norm"]
        result[family] = {"norm": norm, "finite": finite,
                          "gradient_parameter_names": [name for (name, parameter)
                              in named if parameter.grad is not None],
                          "nonzero_parameter_names": [name for (name, parameter)
                              in named if parameter.grad is not None
                              and bool(torch.count_nonzero(parameter.grad))]}
    return result


def _apply_two_optimizer_step(loss, core_params, head_params, core_optimizer,
                              head_optimizer, *, step=True):
    core_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    core_norm = torch.nn.utils.clip_grad_norm_(core_params, CLIP_NORM)
    head_norm = torch.nn.utils.clip_grad_norm_(head_params, CLIP_NORM)
    summary = {"core": _gradient_summary(core_params),
               "head": _gradient_summary(head_params),
               "core_preclip_norm": float(core_norm.detach().cpu()),
               "head_preclip_norm": float(head_norm.detach().cpu())}
    if any(not row["finite"] or not math.isfinite(row["norm"])
           for row in (summary["core"], summary["head"])):
        raise FloatingPointError("nonfinite pilot gradient")
    if summary["core"]["norm"] <= 0 or summary["head"]["norm"] <= 0:
        raise AssertionError("both separate optimizers require finite nonzero gradient credit")
    if step:
        core_optimizer.step()
        head_optimizer.step()
    return summary


def _preflight_path(arm):
    return ARCHIVE / f"pilot_seed1_{arm}_preflight.json"


def validate_screen_prerequisites():
    return _screen_evidence()


def _source_phase_parity(device, gamma, candidate_trace):
    source, *_ = screen.source_and_cache_contract(SEED)
    reference = base.make_core(device, steps=64)
    reference.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    reference.eval()
    with torch.no_grad():
        _, _, _, native_theta = reference(gamma, num_time_steps=STEPS,
                                           return_theta=True, return_core_out=True)
    from collaborative_test.SW_0125_late_rollout_credit.late_rollout import sinusoidal_gating
    history = [native_theta[:, i] for i in range(STEPS)]
    native_carrier, native_gate = [], []
    for step in range(STEPS):
        c, g = sinusoidal_gating(history[:step+1], step, reference.phase_delay_steps,
                                 gate_mode=reference.gate_mode)
        native_carrier.append(c); native_gate.append(g)
    native_carrier = torch.stack(native_carrier, dim=-1)
    native_gate = torch.stack(native_gate, dim=-1)
    checks = {"theta": theta_trace_exact(candidate_trace["theta"], native_theta),
              "carrier": torch.equal(candidate_trace["carrier"], native_carrier),
              "gate": torch.equal(candidate_trace["gate"], native_gate)}
    if not all(checks.values()):
        raise AssertionError(f"source phase/carrier/gate parity failed: {checks}")
    return checks


def theta_trace_exact(candidate_bndt, native_btnd):
    """Compare custom [B,N,D,T] history to the core's [B,T,N,D] return."""
    return (candidate_bndt.ndim == 4 and native_btnd.ndim == 4
            and torch.equal(candidate_bndt, native_btnd.permute(0, 2, 3, 1)))


def preflight(arm, output, device="cuda:0"):
    if arm not in ARM_NAMES:
        raise ValueError("unknown SW0126 pilot arm")
    output = Path(output)
    if output.resolve() != _preflight_path(arm).resolve():
        raise ValueError(f"preflight output must use the registered path {_preflight_path(arm)}")
    if output.exists():
        raise FileExistsError(f"preserving existing preflight output: {output}")
    screens = validate_screen_prerequisites()
    assets = load_assets(SEED, device)
    core_params = _trainable(assets["core"])
    core_families = _core_parameter_families(assets["core"])
    head_params = list(assets["binder"].parameters())
    core_optimizer = torch.optim.Adam(core_params, lr=CORE_LR)
    head_optimizer = torch.optim.Adam(head_params, lr=HEAD_LR)
    image, target, gamma, gamma_delta, ids = _read_batch(assets, 0, device)
    loss, trace, output_values, events, gates = _loss_for_batch(assets, gamma, target, arm)
    parity = _source_phase_parity(device, gamma, trace)
    before_core = {n: p.detach().clone() for n, p in assets["core"].named_parameters()
                   if p.requires_grad}
    before_head = {n: p.detach().clone() for n, p in assets["binder"].named_parameters()}
    grad = _apply_two_optimizer_step(loss, core_params, head_params,
                                     core_optimizer, head_optimizer, step=True)
    family_gradients = _named_family_gradients(core_families)
    core_before_by_name = before_core
    core_changed = any(not torch.equal(before_core[n], p) for n, p in assets["core"].named_parameters()
                       if n in before_core)
    family_changes = {family: any(not torch.equal(core_before_by_name[name], parameter)
                          for name, parameter in named)
                      for family, named in core_families.items() if family != "all"}
    head_changed = any(not torch.equal(before_head[n], p) for n, p in assets["binder"].named_parameters())
    if not core_changed or not head_changed:
        raise AssertionError("disposable preflight Adam step did not change both parameter groups")
    if arm == "history_event":
        for family in ("beta", "membrane", "dendrite"):
            report = family_gradients[family]
            if (not report["finite"] or report["norm"] <= 0 or not report["nonzero_parameter_names"]
                    or not family_changes[family]):
                raise AssertionError(f"actual-event disposable step did not update {family}: {report}")
    if (not family_gradients["oscillator_drive"]["finite"]
            or family_gradients["oscillator_drive"]["norm"] <= 0):
        raise AssertionError("disposable reconstruction has no finite oscillator-drive gradient")
    record = {"status": "passed", "experiment": "SW0126", "stage": "preflight",
        "seed": SEED, "arm": arm, "optimizer_updates_training": 0,
        "disposable_optimizer_steps": 1, "ground_truth_used": False,
        "implementation_fingerprint": implementation_fingerprint(),
        "screen_prerequisite_reports": screens,
        "source_sha256": sha(assets["source"]), "source_manifest_sha256": sha(assets["source_manifest"]),
        "training_ids_sha256": hashlib.sha256(np.asarray(assets["ids"], dtype="<i8").tobytes()).hexdigest(),
        "matched_global_ids": ids, "gamma_cache_max_abs_diff": gamma_delta,
        "source_theta_carrier_gate_parity": parity,
        "rgb_mse": float(loss.detach().cpu()), "gradient_summary": grad,
        "core_family_gradient_summary": family_gradients,
        "core_family_parameter_changed": family_changes,
        "parameter_groups_changed": {"core": core_changed, "head": head_changed},
        "contract": {"batch": BATCH, "updates": UPDATES, "steps": STEPS,
                     "settle": SETTLE, "tail": TAIL,
                     "core_lr": CORE_LR, "head_lr": HEAD_LR, "clip_norm_each": CLIP_NORM}}
    write_once(output, record)
    return record


def _validate_preflight(arm):
    path = _preflight_path(arm)
    record = json.loads(path.read_text(encoding="utf-8"))
    current_screens = validate_screen_prerequisites()
    expected_order_sha = hashlib.sha256(
        np.asarray(base.train_indices(SEED)[0], dtype="<i8").tobytes()).hexdigest()
    family_gradients = record.get("core_family_gradient_summary", {})
    family_changes = record.get("core_family_parameter_changed", {})
    groups_ok = (all(family_gradients.get(name, {}).get("finite") is True
                     and float(family_gradients[name].get("norm", 0.0)) > 0
                     for name in ("oscillator_drive",))
                 and all(family_gradients.get(name, {}).get("finite") is True
                         and float(family_gradients[name].get("norm", 0.0)) > 0
                         and bool(family_gradients[name].get("nonzero_parameter_names"))
                         and family_changes.get(name) is True
                         for name in (("beta", "membrane", "dendrite")
                                      if arm == "history_event" else ())))
    core_grad = record.get("gradient_summary", {}).get("core", {})
    head_grad = record.get("gradient_summary", {}).get("head", {})
    if (record.get("status") != "passed" or record.get("experiment") != "SW0126"
            or record.get("seed") != SEED or record.get("arm") != arm
            or record.get("implementation_fingerprint") != implementation_fingerprint()
            or record.get("source_sha256") != base.EXPECTED_SOURCE_SHAS[SEED]
            or record.get("source_manifest_sha256") != sha(base.source_paths(SEED)[1])
            or record.get("training_ids_sha256") != expected_order_sha
            or record.get("screen_prerequisite_reports") != current_screens
            or record.get("ground_truth_used") is not False
            or not math.isfinite(float(record.get("gamma_cache_max_abs_diff", float("nan"))))
            or float(record.get("gamma_cache_max_abs_diff", float("inf"))) > 2e-5
            or record.get("source_theta_carrier_gate_parity") != {
                "theta": True, "carrier": True, "gate": True}
            or len(record.get("matched_global_ids", [])) != BATCH
            or record.get("matched_global_ids") != base.train_indices(SEED)[0][:BATCH].tolist()
            or not core_grad.get("finite") or float(core_grad.get("norm", 0.0)) <= 0
            or not head_grad.get("finite") or float(head_grad.get("norm", 0.0)) <= 0
            or not groups_ok
            or record.get("contract") != {"batch": BATCH, "updates": UPDATES,
                "steps": STEPS, "settle": SETTLE, "tail": TAIL,
                "core_lr": CORE_LR, "head_lr": HEAD_LR, "clip_norm_each": CLIP_NORM}
            or not record.get("parameter_groups_changed", {}).get("core")
            or not record.get("parameter_groups_changed", {}).get("head")):
        raise AssertionError(f"pilot {arm} preflight is absent, stale, or failed")
    return {"path": str(path), "sha256": sha(path), "record": record}


def train(arm, output, device="cuda:0"):
    if arm not in ARM_NAMES:
        raise ValueError("unknown SW0126 pilot arm")
    screens = validate_screen_prerequisites()
    preflight_evidence = _validate_preflight(arm)
    output = Path(output)
    if output.resolve() != training_dir(arm).resolve():
        raise ValueError(f"training output must use the registered path {training_dir(arm)}")
    if output.exists():
        raise FileExistsError(f"preserving existing training path: {output}")
    assets = load_assets(SEED, device)
    core, binder = assets["core"], assets["binder"]
    core_params, head_params = _trainable(core), list(binder.parameters())
    core_optimizer = torch.optim.Adam(core_params, lr=CORE_LR)
    head_optimizer = torch.optim.Adam(head_params, lr=HEAD_LR)
    order_sha = hashlib.sha256(np.asarray(assets["ids"], dtype="<i8").tobytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    history, max_gamma_diff = [], 0.0
    started = time.time()
    graph_before = {n: p.detach().clone() for n, p in core.named_parameters()
                    if n.startswith("graph_generator.")}
    for update in range(UPDATES):
        start = update * BATCH
        image, target, gamma, gamma_delta, ids = _read_batch(assets, start, device)
        max_gamma_diff = max(max_gamma_diff, gamma_delta)
        loss, trace, values, events, gates = _loss_for_batch(assets, gamma, target, arm)
        core_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        core_norm = torch.nn.utils.clip_grad_norm_(core_params, CLIP_NORM)
        head_norm = torch.nn.utils.clip_grad_norm_(head_params, CLIP_NORM)
        if not torch.isfinite(loss) or not torch.isfinite(core_norm) or not torch.isfinite(head_norm):
            raise FloatingPointError(f"nonfinite SW0126 training update {update+1}")
        core_optimizer.step(); head_optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in core.parameters()) or any(
                not bool(torch.isfinite(p).all()) for p in binder.parameters()):
            raise FloatingPointError(f"nonfinite SW0126 parameter after update {update+1}")
        history.append({"update": update + 1, "ids_first": ids[0], "ids_last": ids[-1],
            "rgb_mse": float(loss.detach().cpu()),
            "core_grad_norm_preclip": float(core_norm.detach().cpu()),
            "head_grad_norm_preclip": float(head_norm.detach().cpu()),
            "gamma_cache_max_abs_diff": gamma_delta})
    graph_changed = [n for n, p in core.named_parameters()
                     if n in graph_before and not torch.equal(graph_before[n], p)]
    if graph_changed:
        raise AssertionError(f"frozen source graph changed during pilot: {graph_changed}")
    _save_torch(output / "core.pt", core.state_dict())
    _save_torch(output / "binder.pt", binder.state_dict())
    _save_torch(output / "core_optimizer.pt", core_optimizer.state_dict())
    _save_torch(output / "head_optimizer.pt", head_optimizer.state_dict())
    write_once(output / "history.json", history)
    manifest = {"status": "training_complete", "experiment": "SW0126",
        "seed": SEED, "arm": arm, "updates": UPDATES, "batch_size": BATCH,
        "total_steps": STEPS, "settle_steps": SETTLE, "live_tail_steps": TAIL,
        "core_lr": CORE_LR, "head_lr": HEAD_LR,
        "core_clip_norm": CLIP_NORM, "head_clip_norm": CLIP_NORM,
        "loss": "native RGB reconstruction MSE only",
        "ground_truth_used_for_training": False, "encoder_frozen": True,
        "graph_frozen": True, "training_ids_sha256": order_sha,
        "training_ids_count": len(assets["ids"]), "screen_prerequisite_reports": screens,
        "preflight_sha256": preflight_evidence["sha256"],
        "source_core_sha256": sha(assets["source"]),
        "source_manifest_sha256": sha(assets["source_manifest"]),
        "encoder_source_sha256": sha(sw117.ENCODER_PATH),
        "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_cache_sha256": sha(base.GAMMA_TRAIN),
        "gamma_cache_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": assets["rgb_cache_sha"],
        "implementation_fingerprint": implementation_fingerprint(),
        "max_gamma_cache_abs_diff": max_gamma_diff,
        "core_sha256": sha(output / "core.pt"), "binder_sha256": sha(output / "binder.pt"),
        "core_optimizer_sha256": sha(output / "core_optimizer.pt"),
        "head_optimizer_sha256": sha(output / "head_optimizer.pt"),
        "history_sha256": sha(output / "history.json"),
        "elapsed_seconds": time.time() - started}
    write_once(output / "manifest.json", manifest)
    write_once(output / "TRAINING_COMPLETED", {"manifest_sha256": sha(output / "manifest.json")})
    return manifest


def _save_torch(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"preserving existing temporary artifact: {tmp}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def valid_training(path, arm):
    path = Path(path)
    try:
        manifest_path = path / "manifest.json"
        marker_path = path / "TRAINING_COMPLETED"
        if not manifest_path.is_file() or not marker_path.is_file():
            return False
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        source, source_manifest, *_ = base.source_paths(SEED)
        expected_ids = base.train_indices(SEED)[0]
        expected_ids_sha = hashlib.sha256(np.asarray(expected_ids, dtype="<i8").tobytes()).hexdigest()
        expected_screens = validate_screen_prerequisites()
        return (m.get("status") == "training_complete" and m.get("experiment") == "SW0126"
            and m.get("seed") == SEED and m.get("arm") == arm and m.get("updates") == UPDATES
            and m.get("batch_size") == BATCH and m.get("total_steps") == STEPS
            and m.get("settle_steps") == SETTLE and m.get("live_tail_steps") == TAIL
            and m.get("training_ids_count") == 4096 and m.get("ground_truth_used_for_training") is False
            and m.get("source_core_sha256") == sha(source)
            and m.get("source_manifest_sha256") == sha(source_manifest)
            and m.get("encoder_source_sha256") == sha(sw117.ENCODER_PATH)
            and m.get("preprocessing_sha256") == sha(sw117.STATS_PATH)
            and m.get("gamma_cache_sha256") == sha(base.GAMMA_TRAIN)
            and m.get("gamma_cache_manifest_sha256") == sha(base.GAMMA_TRAIN_MANIFEST)
            and m.get("training_ids_sha256") == expected_ids_sha
            and m.get("screen_prerequisite_reports") == expected_screens
            and m.get("preflight_sha256") == sha(_preflight_path(arm))
            and m.get("implementation_fingerprint") == implementation_fingerprint()
            and marker.get("manifest_sha256") == sha(manifest_path)
            and m.get("core_sha256") == sha(path / "core.pt")
            and m.get("binder_sha256") == sha(path / "binder.pt")
            and m.get("core_optimizer_sha256") == sha(path / "core_optimizer.pt")
            and m.get("head_optimizer_sha256") == sha(path / "head_optimizer.pt")
            and m.get("history_sha256") == sha(path / "history.json")
            and len(json.loads((path / "history.json").read_text())) == UPDATES
            and [row.get("update") for row in json.loads((path / "history.json").read_text())]
                == list(range(1, UPDATES + 1)))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def training_dir(arm):
    return OUT / f"seed1_{arm}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "train"):
        command = subs.add_parser(stage)
        command.add_argument("--arm", choices=ARM_NAMES, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (preflight(args.arm, args.output, args.device) if args.stage == "preflight"
              else train(args.arm, args.output, args.device))
    print(json.dumps({"status": result["status"], "stage": args.stage,
                      "seed": SEED, "arm": args.arm, "output": str(args.output)},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
