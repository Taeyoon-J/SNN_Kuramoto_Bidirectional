"""Read-only real-data parity/credit check for SW0125 late-rollout foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "collaborative_test") not in sys.path:
    sys.path.insert(0, str(ROOT / "collaborative_test"))

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss, production_partition,
)
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0125_late_rollout_credit.late_rollout import late_rollout
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import phase_locking_value

FULL_STEPS, LOSS_SETTLE, TAIL_STEPS, BATCH = 1024, 512, 64, 16


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_nonzero_norm(loss, parameters):
    parameters = list(parameters)
    if not parameters:
        raise AssertionError("registered parameter family is empty")
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = 0.0
    for grad in grads:
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite read-only Q/RGB gradient")
            squares += float(grad.detach().double().square().sum())
    norm = math.sqrt(squares)
    if not math.isfinite(norm) or norm <= 0:
        raise AssertionError("registered Q/RGB credit is zero or nonfinite")
    return norm


def verify(seed, output, device="cuda:0"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing foundation report: {output}")
    source, source_manifest, source_meta = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    if len(ids) != 4096 or len(rows) != 4096:
        raise AssertionError("SW0097 source must bind its registered 4096 TRAIN ordering")
    if sha(source) != base.EXPECTED_SOURCE_SHAS[seed]:
        raise AssertionError("registered immutable SW0097 source checkpoint SHA mismatch")
    if sha(sw117.ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256:
        raise AssertionError("registered encoder SHA mismatch")
    if sha(sw117.STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256:
        raise AssertionError("registered preprocessing SHA mismatch")

    gamma_cache, gamma_manifest = base.validate_gamma_cache(
        base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = sw117.load_rgb_training_cache()
    core = base.make_core(device, steps=64)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core.requires_grad_(True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    if (core.kuramoto.spike_pulse_gain is not None or core.graph_generator is None
            or core.graph_generator.uses_feedback or core.osc_dim != 4):
        raise AssertionError("SW0097 source is outside the registered static four-component contract")
    encoder = sw117.load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
                                       kernel_size=3, channels=3, device=device)
    encoder.requires_grad_(True)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device)
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    batch_rows = rows[:BATCH]
    row_tensor = torch.as_tensor(batch_rows, dtype=torch.long)
    images = sw117.read_rgb(rgb_cache, batch_rows, device)
    rgb_patches = rgb_base.rgb_patch_means(images / 255.0)
    # Keep this regenerated gamma live: the reference rollout below is no-grad,
    # while the late-tail probe must measure credit into the encoder.
    gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
    cached = gamma_cache[row_tensor].to(device)
    max_gamma_diff = float((gamma - cached).abs().max().detach().cpu().item())
    if not math.isfinite(max_gamma_diff) or max_gamma_diff > 2e-5:
        raise AssertionError(f"live encoder gamma differs from registered cache by {max_gamma_diff}")

    core.eval(); encoder.eval()
    with torch.no_grad():
        _, source_spikes, source_membrane, source_theta = core(
            gamma, return_core_out=True, return_theta=True, num_time_steps=FULL_STEPS)
        source_components = core.last_component_spikes.detach().clone()
        source_component_membrane = core.last_component_out.detach().clone()
    tail = late_rollout(core, gamma, total_steps=FULL_STEPS, live_tail_steps=TAIL_STEPS)
    actual = tail["component_spikes"]
    if tuple(actual.shape) != (BATCH, 4, 256, FULL_STEPS):
        raise AssertionError(f"actual rollout shape mismatch: {tuple(actual.shape)}")
    exact = {
        "theta": torch.equal(tail["theta"], source_theta),
        "spikes": torch.equal(tail["spikes"], source_spikes),
        "membrane": torch.equal(tail["membrane"], source_membrane),
        "component_spikes": torch.equal(actual, source_components),
        "component_membrane": torch.equal(tail["component_membrane"], source_component_membrane),
    }
    if not all(exact.values()):
        raise AssertionError(f"late-tail rollout changed no-grad forward traces: {exact}")

    lossfn = sw117.criterion()
    start = FULL_STEPS - LOSS_SETTLE
    source_q = spike_synchrony_affinity(
        source_spikes, components=source_components, settle=LOSS_SETTLE,
        affinity_mode="spike")
    live_q = spike_synchrony_affinity(
        tail["spikes"], components=actual, settle=LOSS_SETTLE,
        affinity_mode="spike")
    if not torch.equal(live_q, source_q):
        raise AssertionError("full-512 actual-Q value changed under late-tail gradient truncation")
    source_labels, source_hard = production_partition(
        source_spikes, source_components, settle=LOSS_SETTLE)
    labels, hard = production_partition(tail["spikes"], actual, settle=LOSS_SETTLE)
    if not torch.equal(labels, source_labels) or any(
            not torch.equal(a, b) for a, b in zip(hard, source_hard)):
        raise AssertionError("full-512 production labels/H changed under late-tail rollout")

    source_plv = phase_locking_value(source_theta, settle=LOSS_SETTLE, combine="mean")
    live_plv = phase_locking_value(tail["theta"], settle=LOSS_SETTLE, combine="mean")
    if not torch.equal(live_plv, source_plv):
        raise AssertionError("late-tail phase objective value differs from no-grad full rollout")
    source_primary, _ = lossfn(plv=source_plv,
                               theta=source_theta[:, LOSS_SETTLE:])
    live_primary, _ = lossfn(plv=live_plv,
                             theta=tail["theta"][:, LOSS_SETTLE:])
    source_q_loss, _ = lossfn(plv=source_q)
    live_q_loss, _ = lossfn(plv=live_q)
    source_rgb, _, _, _ = batch_reconstruction_loss(source_q, source_hard, rgb_patches)
    live_rgb, _, _, _ = batch_reconstruction_loss(live_q, hard, rgb_patches)
    exact_losses = {
        "primary_phase": torch.equal(source_primary, live_primary),
        "positive_actual_q": torch.equal(source_q_loss, live_q_loss),
        "analytic_actual_h_rgb": torch.equal(source_rgb, live_rgb),
    }
    if not all(exact_losses.values()):
        raise AssertionError(f"late-tail objective forward values changed: {exact_losses}")

    named_core = dict(core.named_parameters())
    families = {
        "encoder": list(encoder.parameters()),
        "graph": [p for name, p in named_core.items() if name.startswith("graph_generator.")],
        "oscillator_drive": [p for name, p in named_core.items()
                             if name.startswith(("gamma_channel_proj.", "gamma_phase_gain"))],
        "kuramoto": [p for name, p in named_core.items() if name.startswith("kuramoto.")],
        "dendritic": [p for name, p in named_core.items() if name.startswith("dendric_layer.")],
        "membrane": [p for name, p in named_core.items() if name.startswith("membrane_layer.")],
    }
    gradient_norms = {name: _finite_nonzero_norm(live_rgb, params)
                      for name, params in families.items()}
    report = {
        "status": "complete", "experiment": "SW0125", "seed": int(seed),
        "device": str(device), "batch_size": BATCH, "global_image_ids": ids[:BATCH].tolist(),
        "gamma_cache_rows": np.asarray(batch_rows, dtype=np.int64).tolist(),
        "total_time_steps": FULL_STEPS, "loss_settle_start": start,
        "loss_settle_steps": LOSS_SETTLE, "gradient_tail_steps": TAIL_STEPS,
        "detached_prefix_steps": FULL_STEPS - TAIL_STEPS,
        "ground_truth_used": False, "optimizer_updates": 0,
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "encoder_sha256": sha(sw117.ENCODER_PATH),
        "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_train_cache_sha256": rgb_cache_sha,
        "rgb_train_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "implementation_sha256": {
            "foundation": sha(HERE / "late_rollout.py"),
            "foundation_verifier": sha(HERE / "verify_foundation.py"),
            "source_core": sha(ROOT / "snn_kuramoto_bidirectional/s2net_cls.py"),
            "synchrony": sha(ROOT / "snn_kuramoto_bidirectional/spike_classifier.py"),
            "kuramoto": sha(ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py"),
            "dendrite": sha(ROOT / "snn_kuramoto_bidirectional/dendric_layer.py"),
            "membrane": sha(ROOT / "snn_kuramoto_bidirectional/membrane_layer.py"),
            "gating": sha(ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py"),
            "phase_loss": sha(ROOT / "snn_kuramoto_bidirectional/loss_function.py"),
            "rgb_loss": sha(ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py"),
        },
        "gamma_cache_max_abs_diff": max_gamma_diff,
        "trace_parity_exact": exact, "loss_parity_exact": exact_losses,
        "q_shape": list(live_q.shape), "q_finite": bool(torch.isfinite(live_q).all()),
        "production_hard_labels_and_H_exact": True,
        "phase_primary": float(live_primary.detach()),
        "positive_actual_q": float(live_q_loss.detach()),
        "analytic_actual_h_rgb": float(live_rgb.detach()),
        "late_tail_rgb_gradient_norm_by_family": gradient_norms,
        "interpretation": "Forward-equivalent late-tail truncated BPTT credit probe; not full 1024-step BPTT or a training result.",
        "gamma_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "gamma_manifest_status": gamma_manifest.get("status"),
        "rgb_cache_source_size_bytes": rgb_manifest.get("source_size_bytes"),
        "source_model_seed": source_meta.get("source_model_seed"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"preserve existing foundation report: {output}")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Exclusive creation prevents a concurrent attempt from replacing evidence.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = verify(args.seed, args.output, args.device)
    print(json.dumps({"status": record["status"], "experiment": record["experiment"],
                      "seed": record["seed"], "output": str(args.output),
                      "optimizer_updates": 0, "ground_truth_used": False}), flush=True)


if __name__ == "__main__":
    main()
