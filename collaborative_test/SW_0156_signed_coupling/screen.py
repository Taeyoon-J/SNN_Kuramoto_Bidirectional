"""GT-free single-image SW0156 native signed-coupling feasibility screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0156_signed_coupling.signed_coupling import (
    install_instance_signed_interaction,
    native_logits_and_signed_u,
)

SEED = 1
STEPS, SETTLE = 1024, 512
BETA = 0.5
OUTPUT = HERE / "results_archive" / "source_screen_seed1.json"
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    files = {
        "collaborative_test/SW_0156_signed_coupling/protocol.json": HERE / "protocol.json",
        "collaborative_test/SW_0156_signed_coupling/signed_coupling.py": HERE / "signed_coupling.py",
        "collaborative_test/SW_0156_signed_coupling/screen.py": HERE / "screen.py",
    }
    merged = dict(sw143.implementation_fingerprint())
    # SW0145 exposes the actual-spike Q helper in run.py, but it does not
    # define an implementation_fingerprint API. Bind the module that supplies
    # the helper directly, alongside SW0143's fingerprint (which already
    # binds the classifier implementation used by that helper).
    sw145_run = Path(sw145.__file__).resolve()
    if not sw145_run.is_file():
        raise FileNotFoundError(sw145_run)
    merged[sw145_run.relative_to(ROOT).as_posix()] = sha256_file(sw145_run)
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        merged[name] = sha256_file(path)
    return merged


def _rng_state(device):
    state = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _set_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _trace_digest(trace):
    result = {}
    for key in TRACE_FIELDS:
        value = trace.get(key)
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"trace field {key} missing or nonfinite")
        result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def _component_dynamics(trace, q):
    """Compact descriptive statistics from an already-computed settled trace."""
    spikes = trace["component_spikes"][:, :, :, SETTLE:].detach().to(torch.float64)
    gates = trace["component_gates"][:, :, :, SETTLE:].detach().to(torch.float64)
    if spikes.ndim != 4 or spikes.shape[1] != 4 or gates.shape != spikes.shape:
        raise ValueError("component spike/gate trace must be [B,4,T,N]")
    if not torch.isfinite(spikes).all() or not torch.isfinite(gates).all():
        raise ValueError("component spike/gate trace is nonfinite")
    component_stats = []
    for component in range(4):
        s = spikes[:, component]
        g = gates[:, component]
        gate_support = g > 0
        support_count = int(gate_support.sum().cpu())
        event_fraction = (float((s[gate_support] != 0).to(torch.float64).mean().cpu())
                          if support_count else None)
        component_stats.append({
            # component_spikes is E*g, so its mean is a weighted amplitude,
            # not an estimate of binary event occupancy when gates can be soft.
            "weighted_spike_mean": float(s.mean().cpu()),
            "spike_variance": float(s.var(unbiased=False).cpu()),
            "gate_mean": float(g.mean().cpu()),
            "gate_active_fraction": float((g > 0).to(torch.float64).mean().cpu()),
            "event_on_fraction_when_gate_positive": event_fraction,
            "gate_positive_support_count": support_count,
        })
    if q.ndim != 3 or q.shape[-1] != q.shape[-2] or q.shape[-1] != spikes.shape[-2]:
        raise ValueError("actual Q must be [B,N,N] matching the trace")
    q64 = q.detach().to(torch.float64)
    offdiag = ~torch.eye(q.shape[-1], dtype=torch.bool, device=q.device)
    q_values = q64[:, offdiag]
    if not torch.isfinite(q_values).all():
        raise ValueError("actual Q off-diagonal is nonfinite")
    return {
        "settled_steps": int(spikes.shape[3]),
        "per_component": component_stats,
        "actual_spike_q_offdiagonal": {
            "mean": float(q_values.mean().cpu()),
            "min": float(q_values.min().cpu()),
            "max": float(q_values.max().cpu()),
        },
    }


def _run_rollout(found, gamma, device):
    start = time.perf_counter()
    trace = late_rollout(found.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    q = sw145._actual_q(trace)
    labels = sw136._qcc_labels(trace)
    return trace, q, labels, elapsed


def _digest_state(*modules):
    h = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            h.update(f"{module_index}:{name}".encode("utf-8"))
            h.update(str(value.dtype).encode("ascii"))
            h.update(np.asarray(value.detach().cpu().contiguous()).tobytes())
    return h.hexdigest()


def _run_arm(found, gamma, device, *, beta, base_scale):
    graph = found.wrapped.core.graph_generator
    layer = found.wrapped.core.kuramoto
    restore = install_instance_signed_interaction(graph, layer, beta=beta,
                                                   base_scale=base_scale)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        trace, q, labels, seconds = _run_rollout(found, gamma, device)
        peak = int(torch.cuda.max_memory_allocated(device))
        return trace, q, labels, seconds, peak
    finally:
        restore()


@torch.no_grad()
def run_screen(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0156 screen output: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0156 actual native32 source screen requires CUDA")
    torch.cuda.set_device(dev); torch.cuda.init()
    fp = implementation_fingerprint()

    # Require the already completed same-seed SW0143-T reference and its
    # registered source/parity checks before loading this local intervention.
    preflight_path = sw143._preflight_path(SEED)
    preflight_bytes = preflight_path.read_bytes()
    preflight = json.loads(preflight_bytes.decode("utf-8"))
    if (preflight.get("status") != "preflight_complete"
            or preflight.get("seed") != SEED
            or preflight.get("implementation") != sw143.implementation_fingerprint()
            or preflight.get("ground_truth_used") is not False
            or preflight.get("optimizer_updates") != 0):
        raise ValueError("registered SW0143 seed1 source preflight is missing or stale")
    t_reference = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if t_reference["protocol"].get("arm") != "T":
        raise ValueError("same-seed SW0143-T reference is not validated")

    found = foundation.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
    found.encoder.eval(); found.wrapped.eval()
    source = sw143._native_source_record(found)
    if source != preflight.get("source"):
        raise ValueError("current frozen source lineage differs from SW0143 preflight")
    graph = found.wrapped.core.graph_generator
    layer = found.wrapped.core.kuramoto
    if getattr(layer, "kuramoto_backend", None) != "factorized" or graph.uses_feedback:
        raise ValueError("loaded source model is outside registered factorized non-feedback path")
    original_geodesic = graph._geodesic_distance
    geodesic_operator.install_operator(graph, "T")

    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source RGB cache schema mismatch")
    row, image_id = int(found.pool_indices[0]), int(found.image_ids[0])
    if row < 0 or row >= len(cache):
        raise ValueError("first registered source TRAIN row is invalid")
    rgb_np = np.asarray(cache[row:row + 1]).copy()
    if rgb_np.dtype != np.uint8 or rgb_np.shape != (1, 128, 128, 3):
        raise ValueError("source TRAIN sample schema mismatch")
    rgb_sha = hashlib.sha256(rgb_np.tobytes()).hexdigest()
    rgb = torch.from_numpy(rgb_np).to(device=dev)
    gamma = found.encode(rgb)
    before = _digest_state(found.encoder, found.patcher, found.wrapped)

    try:
        # The registered SW0143 T source path is the baseline. Rewind RNG before
        # every arm so only coupling intervention differs.
        reference_rng = _rng_state(dev)
        source_trace, source_q, source_labels, source_seconds = _run_rollout(found, gamma, dev)
        source_trace_summary = _trace_digest(source_trace)
        source_dynamics = _component_dynamics(source_trace, source_q)
        source_trace_cpu = {key: value.detach().cpu().clone() for key, value in source_trace.items()
                            if key in TRACE_FIELDS}
        source_q_cpu = source_q.detach().cpu().clone()
        source_labels_cpu = np.asarray(source_labels).copy()
        del source_trace, source_q, source_labels

        _set_rng(reference_rng, dev)
        zero_trace, zero_q, zero_labels, zero_seconds, zero_peak = _run_arm(
            found, gamma, dev, beta=0.0, base_scale=1.0)
        parity = {}
        for key in TRACE_FIELDS:
            candidate = zero_trace[key].detach().cpu()
            if (candidate.shape != source_trace_cpu[key].shape
                    or not torch.isfinite(candidate).all()):
                raise ValueError(f"beta0 source trace invalid at {key}")
            parity[key] = bool(torch.equal(source_trace_cpu[key], candidate))
            if not parity[key]:
                raise AssertionError(f"beta0 trace does not exactly reproduce source T at {key}")
        del zero_trace
        if not torch.equal(source_q_cpu, zero_q.detach().cpu()):
            raise AssertionError("beta0 actual-spike Q differs from SW0143 T source")
        if not np.array_equal(source_labels_cpu, zero_labels):
            raise AssertionError("beta0 QCC differs from SW0143 T source")
        del source_trace_cpu

        # Build and validate U from the exact source T logits/A0 before either
        # signed/control rollout. Diagonal support is kept if native top-k chose it.
        native_adjacency = graph(gamma)
        logits, symmetric_u = native_logits_and_signed_u(graph, gamma, native_adjacency)
        native_a0 = layer.prepare_coupling(native_adjacency)[0]
        candidate_a = native_a0 - BETA * symmetric_u
        control_a = 0.5 * native_a0 + 0.5e-6
        offdiag = ~torch.eye(candidate_a.shape[-1], dtype=torch.bool, device=dev)
        neg_mass = candidate_a[:, offdiag].clamp_max(0).abs().sum()
        total_mass = candidate_a[:, offdiag].abs().sum()
        if not torch.isfinite(candidate_a).all() or total_mass <= 0:
            raise FloatingPointError("signed candidate interaction matrix is invalid")
        negative_fraction = float((neg_mass / total_mass).cpu())
        if negative_fraction <= 0:
            raise ValueError("registered signed screen produced no negative off-diagonal interaction mass")
        del native_adjacency, logits, symmetric_u, native_a0, candidate_a, control_a

        _set_rng(reference_rng, dev)
        cand_trace, cand_q, cand_labels, cand_seconds, cand_peak = _run_arm(
            found, gamma, dev, beta=BETA, base_scale=1.0)
        cand_summary = _trace_digest(cand_trace)
        cand_dynamics = _component_dynamics(cand_trace, cand_q)
        cand_finite_q = bool(torch.isfinite(cand_q).all())
        if not cand_finite_q or np.asarray(cand_labels).shape != (1, 32, 32) \
                or not np.issubdtype(np.asarray(cand_labels).dtype, np.integer) \
                or np.any(np.asarray(cand_labels) < 0):
            raise ValueError("signed candidate actual Q/QCC output is invalid")
        cand_counts = np.unique(cand_labels, return_counts=True)
        del cand_trace, cand_q

        _set_rng(reference_rng, dev)
        ctrl_trace, ctrl_q, ctrl_labels, ctrl_seconds, ctrl_peak = _run_arm(
            found, gamma, dev, beta=0.0, base_scale=0.5)
        ctrl_summary = _trace_digest(ctrl_trace)
        ctrl_dynamics = _component_dynamics(ctrl_trace, ctrl_q)
        ctrl_finite_q = bool(torch.isfinite(ctrl_q).all())
        if not ctrl_finite_q or np.asarray(ctrl_labels).shape != (1, 32, 32) \
                or not np.issubdtype(np.asarray(ctrl_labels).dtype, np.integer) \
                or np.any(np.asarray(ctrl_labels) < 0):
            raise ValueError("matched control actual Q/QCC output is invalid")
        ctrl_counts = np.unique(ctrl_labels, return_counts=True)
        del ctrl_trace, ctrl_q
        torch.cuda.synchronize(dev)
        after = _digest_state(found.encoder, found.patcher, found.wrapped)
        if before != after:
            raise AssertionError("source model state changed during inference-only screen")

        record = {
            "experiment": "SW0156_native32_signed_coupling",
            "status": "source_screen_complete",
            "seed": SEED,
            "training_image_id": image_id,
            "training_cache_row": row,
            "training_rgb_sha256": rgb_sha,
            "source_lineage": source,
            "sw0143_preflight_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
            "sw0143_T_prediction_sha256": t_reference["prediction_sha256"],
            "sw0143_T_protocol_sha256": t_reference["protocol_sha256"],
            "implementation": fp,
            "source_model_state_before_sha256": before,
            "source_model_state_after_sha256": after,
            "source_model_unchanged": before == after,
            "registered_parameters": {"steps": STEPS, "settle": SETTLE,
                                       "beta": BETA, "source_top_k": int(graph.top_k),
                                       "gain": float(graph.log_coupling_gain.exp().cpu()),
                                       "coupling_backend": layer.kuramoto_backend},
            "zero_intervention_exact_parity": {
                "trace_fields": parity, "q_equal": True, "qcc_equal": True},
            "arms": {
                "source_T": {"seconds": source_seconds, "trace_schema": source_trace_summary,
                             "dynamics": source_dynamics},
                "beta0_exact_control": {"seconds": zero_seconds,
                    "peak_cuda_allocated_bytes": zero_peak,
                    "trace_schema": {key: {"shape": source_trace_summary[key]["shape"],
                                            "dtype": source_trace_summary[key]["dtype"]}
                                     for key in TRACE_FIELDS},
                    "dynamics": source_dynamics},
                "signed_candidate": {"beta": BETA, "base_scale": 1.0,
                    "seconds": cand_seconds, "peak_cuda_allocated_bytes": cand_peak,
                    "trace_schema": cand_summary, "dynamics": cand_dynamics,
                    "finite_q": cand_finite_q,
                    "qcc_label_counts": {str(int(k)): int(v) for k, v in zip(*cand_counts)}},
                "matched_control": {"beta": 0.0, "base_scale": 0.5,
                    "seconds": ctrl_seconds, "peak_cuda_allocated_bytes": ctrl_peak,
                    "trace_schema": ctrl_summary, "dynamics": ctrl_dynamics,
                    "finite_q": ctrl_finite_q,
                    "qcc_label_counts": {str(int(k)): int(v) for k, v in zip(*ctrl_counts)}},
            },
            "signed_interaction_screen": {
                "candidate_negative_offdiagonal_abs_mass_fraction": negative_fraction,
                "ground_truth_used": False,
                "optimizer_updates": 0,
                "full320_authorized_by_this_record": False,
            },
            "ground_truth_used": False,
            "optimizer_updates": 0,
        }
    finally:
        graph._geodesic_distance = original_geodesic
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run_screen(device=args.device, output=args.output), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
