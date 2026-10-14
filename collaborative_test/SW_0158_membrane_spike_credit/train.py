"""Checkpointable, fixed-budget SW0158 training continuation.

Only the emitted-spike amplitude path is new. The source encoder, native patcher,
and SW0143-T graph stay frozen; the supervised objective and TRAIN data loader
are the existing SW0155 versions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155
from collaborative_test.SW_0158_membrane_spike_credit import preflight, spike_credit

EXPERIMENT = "SW0158_membrane_spike_credit"
ARMS = ("live_membrane_credit", "detached_membrane_control")
SEED = 1
BATCH, MICRO, UPDATES = 16, 4, 128
STEPS, SETTLE, TAIL = 1024, 512, 64
HDF5_PATH = sw155.HDF5_PATH
TRAIN_ROOT = HERE / "training"
WORK_ROOT = HERE / "work"
PREFLIGHT_PATH = HERE / "results_archive" / "preflight_seed1.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [HERE / name for name in (
        "protocol.json", "README.md", "spike_credit.py", "preflight.py", "train.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
        "snn_kuramoto_bidirectional/dendric_layer.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
        "snn_kuramoto_bidirectional/sinusoidal_gating.py"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0158 training dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_contract():
    return sw155.source_contract()


def _load_foundation(device):
    foundation = sw155._load_foundation(device)
    geodesic_operator.install_operator(foundation.wrapped.core.graph_generator, "T")
    margin = spike_credit.registered_margin_rms()
    spike_credit.install_spike_credit(foundation.wrapped, margin)
    return foundation


def _validate_preflight(path, source, assets):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("SW0158 training requires a completed actual-source preflight")
    upstream = preflight.validate_upstream()
    if upstream["source_sha256"] != source["source_sha256"] or upstream["rgb_asset_validation"] != assets:
        raise ValueError("SW0158 preflight source/assets differ from current source contract")
    if not preflight.validate_report(path, expected_fingerprint=preflight.implementation_fingerprint(),
                                    upstream=upstream):
        raise ValueError("SW0158 preflight failed current strict validator or admission gates")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("training_admission") is not True:
        raise ValueError("SW0158 training admission is not true")
    return record


def _configure(foundation):
    wrapped, core = foundation.wrapped, foundation.wrapped.core
    foundation.encoder.requires_grad_(False)
    foundation.encoder.eval()
    foundation.patcher.requires_grad_(False)
    foundation.patcher.eval()
    if core.graph_generator is None:
        raise ValueError("SW0158 requires the installed source-T graph")
    core.graph_generator.requires_grad_(False)
    core.graph_generator.eval()
    native = [p for name, p in core.named_parameters()
              if not name.startswith("graph_generator.")]
    integration = [wrapped.a_d, wrapped.a_m, wrapped.b]
    for parameter in native + integration:
        parameter.requires_grad_(True)
    wrapped.sw0158_eta.requires_grad_(True)
    wrapped.eval()
    families = {"native_core": native, "integration_tensors": integration,
                "eta": [wrapped.sw0158_eta]}
    params = [p for group in families.values() for p in group]
    if (len({id(p) for p in params}) != len(params) or not params
            or any(not p.requires_grad for p in params)
            or any(p.requires_grad for p in foundation.encoder.parameters())
            or any(p.requires_grad for p in foundation.patcher.parameters())
            or any(p.requires_grad for p in core.graph_generator.parameters())):
        raise AssertionError("SW0158 trainable/frozen parameter partition is invalid")
    optimizer = torch.optim.Adam([
        {"name": "native_core", "params": native, "lr": 3e-5},
        {"name": "integration_tensors", "params": integration, "lr": 1e-3},
        {"name": "eta", "params": [wrapped.sw0158_eta], "lr": 1e-3}])
    return families, optimizer


def _source_t_parity(foundation, source, assets, ids, rows, cache):
    rgb = torch.from_numpy(np.asarray(cache[rows[:4]]).copy())
    if rgb.dtype != torch.uint8 or tuple(rgb.shape) != (4, 128, 128, 3):
        raise ValueError("SW0158 initial parity RGB must be the first four registered TRAIN images")
    with torch.no_grad():
        gamma = foundation.encode(rgb.to(next(foundation.wrapped.parameters()).device))
        from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
        from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
        from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
        expected = late_rollout(foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
        actual = spike_credit.late_rollout_credit(
            foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
    parity = {key: torch.equal(expected[key], actual[key])
              for key in ("theta", "membrane", "spikes", "component_membrane",
                          "component_spikes", "component_gates")}
    if not all(parity.values()):
        raise AssertionError(f"eta-zero SW0158 source-T parity failed: {parity}")
    expected_q = spike_synchrony_affinity(expected["spikes"], expected["component_spikes"],
        settle=SETTLE, affinity_mode="spike")
    actual_q = spike_synchrony_affinity(actual["spikes"], actual["component_spikes"],
        settle=SETTLE, affinity_mode="spike")
    expected_qcc, actual_qcc = sw136._qcc_labels(expected), sw136._qcc_labels(actual)
    if not torch.equal(expected_q, actual_q) or not np.array_equal(expected_qcc, actual_qcc):
        raise AssertionError("eta-zero SW0158 source-T production Q/QCC parity failed")
    parity.update(production_Q=True, production_QCC=True)
    return parity, hashlib.sha256(rgb.numpy().tobytes()).hexdigest()


def _parameter_changes(families, before):
    result = {}
    for name, params in families.items():
        squared = sum(float((parameter.detach() - old).double().square().sum().cpu())
                      for parameter, old in zip(params, before[name]))
        result[name] = math.sqrt(squared)
    return result


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT_PATH,
              hdf5_path=HDF5_PATH, train_root=TRAIN_ROOT, work_root=WORK_ROOT):
    if arm not in ARMS:
        raise ValueError(f"unregistered SW0158 arm: {arm}")
    src = source_contract()
    assets = sw155.sw130.validate_rgb_assets()
    preflight_record = _validate_preflight(preflight_path, src, assets)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0158 training requires its exclusively assigned CUDA device")
    torch.cuda.set_device(dev)
    cache = np.load(sw155.sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    ids, rows = src["image_ids"][:2048], src["rows"][:2048]
    objective.validate_training_prefix(ids, src["image_ids"])
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    hdf5_stat = hdf5_path.stat()
    foundation = _load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != assets
            or list(map(int, foundation.image_ids)) != src["image_ids"]
            or not np.array_equal(foundation.pool_indices, src["rows"])):
        raise ValueError("fresh SW0158 source foundation differs from registered order/assets")
    parity, parity_rgb_sha = _source_t_parity(foundation, src, assets, ids, rows, cache)
    families, optimizer = _configure(foundation)
    initial_hashes = {name: sw155.source_training.parameter_hash(params)
                      for name, params in families.items()}
    frozen_hashes = {name: sha256_file(path) for name, path in (
        ("encoder", Path(sw155.sw130.ENCODER_SOURCE)),
        ("feature_stats", Path(sw155.sw130.FEATURE_STATS)))}
    frozen_module_before = {name: sw155.source_training.state_sha(module) for name, module in (
        ("rgb_encoder", foundation.encoder), ("native_patcher", foundation.patcher),
        ("source_T_graph", foundation.wrapped.core.graph_generator))}
    bindings = {"experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha256_file(src["manifest_path"]),
        "source_training_ids_sha256": sw155.ids_sha(src["image_ids"]),
        "ordered_train_ids_sha256": sw155.ids_sha(ids), "rgb_asset_validation": assets,
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_implementation_fingerprint": preflight_record["implementation_fingerprint"],
        "source_t_initial_parity": parity, "source_t_parity_rgb_sha256": parity_rgb_sha,
        "sw0157_calibration_sha256": spike_credit.sha256_file(spike_credit.SW157_REPORT),
        "component_margin_rms": list(spike_credit.registered_margin_rms()),
        "implementation_fingerprint": implementation_fingerprint(),
        "hdf5_path": str(hdf5_path.resolve()), "hdf5_size": hdf5_stat.st_size,
        "hdf5_mtime_ns": hdf5_stat.st_mtime_ns, "batch": BATCH, "microbatch": MICRO,
        "updates": UPDATES, "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "initial_parameter_sha256": initial_hashes,
        "frozen_encoder_asset_hashes": frozen_hashes,
        "frozen_module_state_sha256": frozen_module_before,
        "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only"}
    runtime = durable.runtime_signature(dev)
    work_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    output_dir = Path(train_root) / f"{arm}_seed1"
    if output_dir.exists():
        raise FileExistsError("preserving existing SW0158 completed training output")
    if work_dir.exists() and (work_dir / "boundary.ptx").is_file():
        if durable.inspect_committed_metadata(work_dir).get("bindings") != bindings:
            raise ValueError("SW0158 checkpoint lineage mismatch")
        loaded = durable.load_latest(work_dir, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, expected_bindings=bindings,
            expected_runtime=runtime, device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise ValueError("preserving unknown SW0158 checkpoint work")
        work_dir.mkdir(parents=True, exist_ok=True)
        completed, history = 0, []
        durable.save_boundary(work_dir, completed_updates=0, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=dev)
    if (isinstance(completed, bool) or not isinstance(completed, int)
            or not 0 <= completed <= UPDATES or len(history) != completed):
        raise ValueError("SW0158 checkpoint cursor/history is invalid")

    train_cache = np.asarray(cache[rows]).copy()
    with h5py.File(hdf5_path, "r") as hdf5_handle:
        if hdf5_handle["image"].shape != (100000, 128, 128, 3):
            raise ValueError("canonical supervised HDF5 image schema changed")
        if hdf5_handle["mask"].shape not in ((100000, 128, 128), (100000, 128, 128, 1)):
            raise ValueError("canonical supervised HDF5 mask schema changed")
        for update in range(completed, UPDATES):
            batch_ids = ids[update * BATCH:(update + 1) * BATCH]
            def compute_micro(micro):
                start = update * BATCH + micro * MICRO
                rgb_cpu, masks_cpu = sw155.load_aligned_train_rows(
                    hdf5_path, ids, train_cache, start, MICRO, hdf5_handle=hdf5_handle)
                labels = objective.modal_train_targets32(masks_cpu).to(dev)
                with torch.no_grad():
                    gamma = foundation.encode(rgb_cpu.to(dev))
                trace = spike_credit.late_rollout_credit(foundation.wrapped, gamma,
                    total_steps=STEPS, live_tail_steps=TAIL,
                    detached_control=(arm == "detached_membrane_control"))
                loss, _ = objective.supervised_relation_loss(trace, labels, settle=SETTLE)
                return loss
            row = sw155.logical_update(optimizer=optimizer, family_map=families,
                compute_micro=compute_micro, wrapped=foundation.wrapped, micro_count=4)
            row.update(update=update + 1, image_ids=[int(x) for x in batch_ids])
            history.append(row)
            if len(history) != update + 1:
                raise AssertionError("SW0158 update/history cursor diverged")
            durable.save_boundary(work_dir, completed_updates=update + 1,
                wrapped=foundation.wrapped, encoder=foundation.encoder, optimizer=optimizer,
                history=history, bindings=bindings, runtime=runtime, device=dev)
            if (update + 1) % 16 == 0:
                print(json.dumps({"arm": arm, "completed_updates": update + 1,
                                  "total_updates": UPDATES}, allow_nan=False), flush=True)
    if len(history) != UPDATES:
        raise AssertionError("SW0158 training did not reach its registered update count")
    if {name: sw155.source_training.parameter_hash(params) for name, params in families.items()} == initial_hashes:
        raise AssertionError("SW0158 trainable parameters did not change")
    frozen_module_after = {name: sw155.source_training.state_sha(module) for name, module in (
        ("rgb_encoder", foundation.encoder), ("native_patcher", foundation.patcher),
        ("source_T_graph", foundation.wrapped.core.graph_generator))}
    if frozen_module_after != frozen_module_before:
        raise AssertionError("SW0158 source encoder/patcher/T graph weights or buffers changed")
    if {name: sha256_file(path) for name, path in (
            ("encoder", Path(sw155.sw130.ENCODER_SOURCE)),
            ("feature_stats", Path(sw155.sw130.FEATURE_STATS)))} != frozen_hashes:
        raise AssertionError("SW0158 frozen source encoder assets changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    checkpoint = staging / "checkpoint.pt"
    optimizer_path, history_path = staging / "optimizer.pt", staging / "history.json"
    with checkpoint.open("xb") as stream:
        torch.save({"wrapped": foundation.wrapped.state_dict(),
                    "encoder": foundation.encoder.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with optimizer_path.open("xb") as stream:
        torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
    with history_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    final_hashes = {name: sw155.source_training.parameter_hash(params)
                    for name, params in families.items()}
    manifest = {"status": "training_complete", "experiment": EXPERIMENT,
        "arm": arm, "seed": SEED, "detached_amplitude_gradient": arm == "detached_membrane_control",
        "completed_updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO, "exposures": BATCH * UPDATES,
        "train_ids": ids, "bindings": bindings, "initial_parameter_sha256": initial_hashes,
        "final_parameter_sha256": final_hashes,
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_sha256": sha256_file(optimizer_path), "history_sha256": sha256_file(history_path),
        "implementation_fingerprint": bindings["implementation_fingerprint"],
        "optimizer_step_summary": {group["name"]: max(
            (int(optimizer.state[p]["step"].item()) for p in group["params"] if p in optimizer.state),
            default=0) for group in optimizer.param_groups},
        "frozen_encoder_asset_hashes": frozen_hashes,
        "frozen_module_state_sha256_before_after": frozen_module_before,
        "frozen_module_state_sha256_final": frozen_module_after,
        "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only"}
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.rename(staging, output_dir)
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    args = parser.parse_args(argv)
    manifest = train_arm(args.arm, device=args.device,
                         preflight_path=args.preflight, hdf5_path=args.hdf5)
    print(json.dumps({"status": manifest["status"], "completed_updates": UPDATES,
                      "arm": args.arm, "ground_truth_scope": "TRAIN masks only"}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
