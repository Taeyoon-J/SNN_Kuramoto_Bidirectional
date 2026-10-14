"""Fresh paired SW0159 source-T continuation, guarded by its bounded preflight."""
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
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0159_one_shot_event_regime import calibration, preflight

EXPERIMENT = "SW0159_one_shot_event_regime"
ARMS = ("control", "candidate")
SEED = 1
CORE_INIT_SEED = 149001
TRAIN_COUNT = 2048
UPDATES = 128
BATCH = 16
MICRO = 4
STEPS, SETTLE, TAIL = 1024, 512, 64
PREFLIGHT_PATH = HERE / "results_archive" / "preflight_seed1.json"
WORK_ROOT = HERE / "results_archive" / "training_work_20261014"
TRAIN_ROOT = HERE / "results_archive" / "training_20261014"
HDF5_PATH = sw155.HDF5_PATH


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    rels = [
        "collaborative_test/SW_0159_one_shot_event_regime/protocol.json",
        "collaborative_test/SW_0159_one_shot_event_regime/README.md",
        "collaborative_test/SW_0159_one_shot_event_regime/calibration.py",
        "collaborative_test/SW_0159_one_shot_event_regime/preflight.py",
        "collaborative_test/SW_0159_one_shot_event_regime/train.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
    ]
    paths = [ROOT / rel for rel in rels]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0159 frozen training dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    return hashlib.sha256(json.dumps(implementation_fingerprint(), sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_preflight_admission(path=PREFLIGHT_PATH):
    """Require the actual source/GPU preflight; fixtures cannot admit training."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    current_fp = preflight.fingerprint()
    current_upstream = preflight.validate_upstream()
    if not preflight.validate_report_record(
            record, expected_fingerprint=current_fp, upstream=current_upstream):
        raise ValueError("SW0159 preflight failed strict current-source admission")
    if record.get("training_admission") is not True:
        raise ValueError("SW0159 preflight recorded a scientific rejection; training is closed")
    if record.get("upstream", {}).get("sw0157_calibration_sha256") != calibration.REPORT_SHA256:
        raise ValueError("SW0159 preflight calibration lineage differs")
    return record


def _load_foundation(device):
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED)
        torch.cuda.manual_seed(CORE_INIT_SEED)
        foundation = sw155._load_foundation(dev)
    graph = foundation.wrapped.core.graph_generator
    if graph is None:
        raise ValueError("SW0159 requires the learned source graph")
    geodesic_operator.install_operator(graph, "T")
    foundation.encoder.eval()
    foundation.patcher.eval()
    foundation.wrapped.eval()
    return foundation


def configure_trainability(foundation):
    """Freeze source encoder and T graph; train core plus existing integrations."""
    foundation.encoder.requires_grad_(False)
    foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(True)
    core = foundation.wrapped.core
    graph = core.graph_generator
    graph.requires_grad_(False)
    native = [p for name, p in core.named_parameters()
              if not name.startswith("graph_generator.")]
    integrations = [foundation.wrapped.a_d, foundation.wrapped.a_m, foundation.wrapped.b]
    for p in native + integrations:
        p.requires_grad_(True)
    if any(p.requires_grad for p in foundation.encoder.parameters()):
        raise AssertionError("SW0159 source encoder must remain frozen")
    if any(p.requires_grad for p in graph.parameters()):
        raise AssertionError("SW0159 source T graph must remain frozen")
    if set(id(p) for p in native) & set(id(p) for p in graph.parameters()):
        raise AssertionError("SW0159 native/graph optimizer groups overlap")
    return {"native_core": native, "integration_tensors": integrations}


def initialize_arm(foundation, arm, preflight_record):
    if arm not in ARMS:
        raise ValueError("unknown SW0159 arm")
    if arm == "control":
        with torch.no_grad():
            foundation.wrapped.b.zero_()
    else:
        calibration.install_bias(foundation.wrapped,
            preflight_record["initialization"]["bias"])
    actual_threshold = 0.06 * torch.exp(foundation.wrapped.b.detach())
    expected = (torch.full((4,), 0.06, dtype=actual_threshold.dtype) if arm == "control"
                else torch.tensor(preflight_record["initialization"]["thresholds"], dtype=actual_threshold.dtype))
    if not torch.allclose(actual_threshold.cpu(), expected, rtol=2e-6, atol=1e-6):
        raise AssertionError("arm threshold does not match its registered one-shot initialization")
    return actual_threshold.detach().cpu().tolist()


def make_optimizer(families):
    return torch.optim.Adam([
        {"name": "native_core", "params": families["native_core"], "lr": 3e-5},
        {"name": "integration_tensors", "params": families["integration_tensors"], "lr": 1e-3},
    ])


def _parameter_hash(parameters):
    digest = hashlib.sha256()
    for index, value in enumerate(parameters):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{index}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_training_artifact(arm, *, directory=None, preflight_path=PREFLIGHT_PATH,
                               hdf5_path=HDF5_PATH, source=None, assets=None):
    """Validate one completed SW0159 training result and every serialized payload."""
    try:
        if arm not in ARMS:
            return False
        output = Path(directory) if directory is not None else TRAIN_ROOT / f"{arm}_seed1"
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = source or sw155.source_contract()
        assets = assets or sw130.validate_rgb_assets()
        ids = source["image_ids"]
        rows = ids[:TRAIN_COUNT]
        objective.validate_training_prefix(rows, ids)
        preflight_path, hdf5_path = Path(preflight_path), Path(hdf5_path)
        if not preflight_path.is_file() or not hdf5_path.is_file():
            return False
        admission = validate_preflight_admission(preflight_path)
        hdf_stat = hdf5_path.stat()
        expected_fp = implementation_fingerprint()
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict):
            return False
        if (manifest.get("status") != "training_complete"
                or manifest.get("experiment") != EXPERIMENT or manifest.get("arm") != arm
                or manifest.get("seed") != SEED or manifest.get("completed_updates") != UPDATES
                or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
                or manifest.get("exposures") != BATCH * UPDATES
                or manifest.get("train_ids") != rows
                or bindings.get("experiment") != EXPERIMENT or bindings.get("arm") != arm
                or bindings.get("seed") != SEED
                or bindings.get("source_checkpoint_sha256") != source["source_sha256"]
                or bindings.get("source_manifest_sha256") != sha256_file(source["manifest_path"])
                or bindings.get("source_training_ids_sha256") != sw155.ids_sha(ids)
                or bindings.get("ordered_train_ids_sha256") != sw155.ids_sha(rows)
                or bindings.get("rgb_asset_validation") != assets
                or bindings.get("preflight_sha256") != sha256_file(preflight_path)
                or bindings.get("preflight_implementation_fingerprint") != admission.get("implementation_fingerprint")
                or bindings.get("hdf5_path") != str(hdf5_path.resolve())
                or bindings.get("hdf5_size_bytes") != hdf_stat.st_size
                or bindings.get("hdf5_mtime_ns") != hdf_stat.st_mtime_ns
                or bindings.get("updates") != UPDATES or bindings.get("logical_batch") != BATCH
                or bindings.get("microbatch") != MICRO or bindings.get("steps") != STEPS
                or bindings.get("settle") != SETTLE or bindings.get("live_tail") != TAIL
                or bindings.get("initial_thresholds") != manifest.get("initial_thresholds")
                or bindings.get("ground_truth_used") is not True
                or bindings.get("ground_truth_scope") != "registered TRAIN masks only"
                or manifest.get("ground_truth_used") is not True
                or manifest.get("ground_truth_scope") != "registered TRAIN masks only"
                or manifest.get("implementation_fingerprint") != expected_fp
                or bindings.get("implementation_fingerprint") != expected_fp):
            return False
        expected_initial = ([0.06] * 4 if arm == "control"
                            else admission["initialization"]["thresholds"])
        if (not isinstance(manifest.get("initial_thresholds"), list)
                or len(manifest["initial_thresholds"]) != 4
                or any(not np.isclose(float(a), float(b), rtol=2e-6, atol=1e-6)
                       for a, b in zip(manifest["initial_thresholds"], expected_initial))):
            return False
        initial = manifest.get("initial_trainable_parameter_sha256")
        final = manifest.get("final_trainable_parameter_sha256")
        groups = {"native_core", "integration_tensors"}
        if (not isinstance(initial, dict) or not isinstance(final, dict)
                or set(initial) != groups or set(final) != groups
                or any(not isinstance(v, str) or len(v) != 64 for v in (*initial.values(), *final.values()))
                or any(initial[name] == final[name] for name in groups)):
            return False
        frozen = (manifest.get("frozen_encoder_sha256"), manifest.get("frozen_graph_sha256"))
        if any(not isinstance(v, str) or len(v) != 64 for v in frozen):
            return False
        hashes = {}
        for name, suffix in (("checkpoint", "pt"), ("optimizer", "pt"), ("history", "json")):
            path = output / f"{name}.{suffix}"
            expected_hash = manifest.get(f"{name}_sha256")
            if (not path.is_file() or not isinstance(expected_hash, str) or len(expected_hash) != 64
                    or sha256_file(path) != expected_hash):
                return False
            hashes[name] = path
        history = json.loads(hashes["history"].read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != UPDATES:
            return False
        for index, row in enumerate(history, start=1):
            if (not isinstance(row, dict) or row.get("update") != index
                    or row.get("image_ids") != rows[(index - 1) * BATCH:index * BATCH]
                    or not isinstance(row.get("mean_loss"), (int, float))
                    or not math.isfinite(row["mean_loss"])):
                return False
            norms = row.get("family_gradient_norms")
            if (not isinstance(norms, dict) or set(norms) != groups
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                           for v in norms.values())):
                return False
        checkpoint = torch.load(hashes["checkpoint"], map_location="cpu", weights_only=False)
        if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("wrapped"), dict)
                or not isinstance(checkpoint.get("encoder"), dict)):
            return False
        optimizer_state = torch.load(hashes["optimizer"], map_location="cpu", weights_only=False)
        if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("param_groups"), list):
            return False
        param_groups = optimizer_state["param_groups"]
        if ([group.get("name") for group in param_groups] != ["native_core", "integration_tensors"]
                or [float(group.get("lr", -1)) for group in param_groups] != [3e-5, 1e-3]):
            return False
        states = optimizer_state.get("state")
        if not isinstance(states, dict):
            return False
        steps = []
        for state in states.values():
            if not isinstance(state, dict) or "step" not in state:
                return False
            step = state["step"]
            step = int(step.item()) if torch.is_tensor(step) else int(step)
            if not 1 <= step <= UPDATES:
                return False
            steps.append(step)
        actual_summary = {group["name"]: max(
            (int(states[param_id]["step"].item()) if torch.is_tensor(states[param_id]["step"])
             else int(states[param_id]["step"]) for param_id in group["params"] if param_id in states),
            default=0) for group in param_groups}
        return (bool(steps) and manifest.get("optimizer_step_summary") == actual_summary
                and set(actual_summary) == groups
                and all(1 <= value <= UPDATES for value in actual_summary.values()))
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError,
            json.JSONDecodeError, OverflowError):
        return False


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT_PATH,
              hdf5_path=HDF5_PATH, work_root=WORK_ROOT, train_root=TRAIN_ROOT):
    if arm not in ARMS:
        raise ValueError("unknown SW0159 arm")
    record = validate_preflight_admission(preflight_path)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0159 paired training requires assigned CUDA")
    torch.cuda.set_device(dev)
    src = sw155.source_contract()
    assets = sw130.validate_rgb_assets()
    if record["upstream"]["source_sha256"] != src["source_sha256"] or record["upstream"]["rgb_asset_validation"] != assets:
        raise ValueError("SW0159 source checkpoint/assets differ from admitted preflight")
    ids, rows = src["image_ids"][:TRAIN_COUNT], src["rows"][:TRAIN_COUNT]
    objective.validate_training_prefix(ids, src["image_ids"])
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered SW0159 TRAIN RGB cache schema changed")
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
        raise ValueError("fresh source97 native32 foundation differs from its contract")
    threshold_init = initialize_arm(foundation, arm, record)
    families = configure_trainability(foundation)
    optimizer = make_optimizer(families)
    fp = implementation_fingerprint()
    initial_parameter_sha = {name: _parameter_hash(params) for name, params in families.items()}
    bindings = {
        "experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha256_file(src["manifest_path"]),
        "source_training_ids_sha256": sw155.ids_sha(src["image_ids"]),
        "ordered_train_ids_sha256": sw155.ids_sha(ids),
        "rgb_asset_validation": assets, "preflight_sha256": sha256_file(preflight_path),
        "preflight_implementation_fingerprint": record["implementation_fingerprint"],
        "implementation_fingerprint": fp,
        "calibration_sha256": calibration.REPORT_SHA256,
        "margin_rms": record["upstream"]["margin_rms"],
        "initial_thresholds": threshold_init,
        "hdf5_path": str(hdf5_path.resolve()), "hdf5_size_bytes": hdf5_stat.st_size,
        "hdf5_mtime_ns": hdf5_stat.st_mtime_ns,
        "updates": UPDATES, "logical_batch": BATCH, "microbatch": MICRO,
        "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "initial_trainable_parameter_sha256": initial_parameter_sha,
        "ground_truth_used": True, "ground_truth_scope": "registered TRAIN masks only",
    }
    runtime = durable.runtime_signature(dev)
    work_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    output_dir = Path(train_root) / f"{arm}_seed1"
    if output_dir.exists():
        raise FileExistsError(f"preserving existing SW0159 training result: {output_dir}")
    if work_dir.exists() and (work_dir / "boundary.ptx").is_file():
        loaded = durable.load_latest(work_dir, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, expected_bindings=bindings,
            expected_runtime=runtime, device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise FileExistsError("preserving uncommitted SW0159 checkpoint data")
        work_dir.mkdir(parents=True, exist_ok=True)
        completed, history = 0, []
        durable.save_boundary(work_dir, completed_updates=0, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=dev)
    if completed > UPDATES or len(history) != completed:
        raise ValueError("SW0159 checkpoint cursor/history is invalid")

    cache_rows = np.asarray(cache[rows]).copy()
    if cache_rows.shape != (TRAIN_COUNT, 128, 128, 3) or cache_rows.dtype != np.uint8:
        raise ValueError("SW0159 ordered RGB training rows are invalid")
    frozen_encoder = source_training.state_sha(foundation.encoder)
    frozen_graph = source_training.state_sha(foundation.wrapped.core.graph_generator)
    with h5py.File(hdf5_path, "r") as h5:
        if h5["image"].shape != (100000, 128, 128, 3) or h5["mask"].shape not in (
                (100000, 128, 128), (100000, 128, 128, 1)):
            raise ValueError("canonical supervised HDF5 schema changed")
        for update in range(completed, UPDATES):
            update_ids = ids[update * BATCH:(update + 1) * BATCH]
            def compute_micro(micro):
                offset = update * BATCH + micro * MICRO
                rgb_cpu, masks_cpu = sw155.load_aligned_train_rows(
                    hdf5_path, ids, cache_rows, offset, MICRO, hdf5_handle=h5)
                labels = objective.modal_train_targets32(masks_cpu).to(dev)
                rgb = rgb_cpu.to(dev)
                with torch.no_grad():
                    gamma = foundation.encode(rgb)
                trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
                loss, _ = objective.supervised_relation_loss(trace, labels, settle=SETTLE)
                del gamma, trace, labels, rgb
                return loss
            row = sw155.logical_update(optimizer=optimizer, family_map=families,
                compute_micro=compute_micro, wrapped=foundation.wrapped, micro_count=4)
            row.update(update=update + 1, image_ids=update_ids)
            history.append(row)
            durable.save_boundary(work_dir, completed_updates=update + 1,
                wrapped=foundation.wrapped, encoder=foundation.encoder,
                optimizer=optimizer, history=history, bindings=bindings,
                runtime=runtime, device=dev)
            if (update + 1) % 16 == 0:
                print(json.dumps({"arm": arm, "completed_updates": update + 1,
                    "total_updates": UPDATES}, allow_nan=False), flush=True)
    if (len(history) != UPDATES
            or source_training.state_sha(foundation.encoder) != frozen_encoder
            or source_training.state_sha(foundation.wrapped.core.graph_generator) != frozen_graph):
        raise AssertionError("SW0159 completion or frozen source state validation failed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    checkpoint_path, optimizer_path, history_path = (
        staging / "checkpoint.pt", staging / "optimizer.pt", staging / "history.json")
    with checkpoint_path.open("xb") as stream:
        torch.save({"wrapped": foundation.wrapped.state_dict(),
                    "encoder": foundation.encoder.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with optimizer_path.open("xb") as stream:
        torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
    with history_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    final_parameter_sha = {name: _parameter_hash(params) for name, params in families.items()}
    manifest = {"status": "training_complete", "experiment": EXPERIMENT,
        "arm": arm, "seed": SEED, "completed_updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO, "exposures": BATCH * UPDATES,
        "train_ids": ids, "bindings": bindings,
        "initial_thresholds": threshold_init,
        "initial_trainable_parameter_sha256": initial_parameter_sha,
        "final_trainable_parameter_sha256": final_parameter_sha,
        "frozen_encoder_sha256": frozen_encoder, "frozen_graph_sha256": frozen_graph,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "optimizer_sha256": sha256_file(optimizer_path),
        "history_sha256": sha256_file(history_path),
        "implementation_fingerprint": fp,
        "optimizer_step_summary": {group["name"]: max(
            (int(optimizer.state[p]["step"].item()) for p in group["params"] if p in optimizer.state),
            default=0) for group in optimizer.param_groups},
        "ground_truth_used": True, "ground_truth_scope": "registered TRAIN masks only"}
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
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
    manifest = train_arm(args.arm, device=args.device, preflight_path=args.preflight,
                         hdf5_path=args.hdf5)
    print(json.dumps({"status": manifest["status"], "arm": args.arm,
        "completed_updates": UPDATES, "training_admission": True}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
