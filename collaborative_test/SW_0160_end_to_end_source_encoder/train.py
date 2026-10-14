"""Checkpoint-resumable paired SW0160 source-encoder continuation."""
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
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as score_contract
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155
from collaborative_test.SW_0160_end_to_end_source_encoder import model, preflight

EXPERIMENT = preflight.EXPERIMENT
SEED, CORE_INIT_SEED, RESIDUAL_SEED = 1, 149001, 160001
TRAIN_COUNT, UPDATES, BATCH, MICRO = 2048, 128, 16, 4
STEPS, SETTLE, TAIL = preflight.STEPS, preflight.SETTLE, preflight.TAIL
HDF5_PATH = Path(sw155.HDF5_PATH)
PREFLIGHT_PATH = preflight.OUTPUT
TRAIN_ROOT = HERE / "results_archive" / "training_20261014"
WORK_ROOT = HERE / "results_archive" / "training_work_20261014"
_SCORE_MODULE = "collaborative_test.SW_0159_one_shot_event_regime"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    rels = [
        "collaborative_test/SW_0160_end_to_end_source_encoder/DESIGN_KO.md",
        "collaborative_test/SW_0160_end_to_end_source_encoder/model.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/preflight.py",
        "collaborative_test/SW_0160_end_to_end_source_encoder/train.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/deep_encoder.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0159_one_shot_event_regime/protocol.json",
        "collaborative_test/SW_0159_one_shot_event_regime/predict_evaluate.py",
        "collaborative_test/SW_0159_one_shot_event_regime/score_queue.py",
        "collaborative_test/SW_0159_one_shot_event_regime/train.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    ]
    paths = [ROOT / rel for rel in rels]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0160 training dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _score_dependency():
    """Bind to SW0159's completed CPU score without recomputing its bootstrap."""
    from collaborative_test.SW_0159_one_shot_event_regime import (
        predict_evaluate as evaluation, score_queue)
    state_path = score_queue.STATE
    score_path = evaluation.SCORE_PATH
    if not state_path.is_file() or not score_path.is_file():
        raise RuntimeError("SW0160 waits for the registered SW0159 terminal score")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    report = json.loads(score_path.read_text(encoding="utf-8"))
    queue_fp, score_fp = score_queue.implementation_fingerprint(), evaluation.implementation_fingerprint()
    score_sha = sha256_file(score_path)
    if (state.get("status") != "complete" or state.get("experiment") != "SW0159_one_shot_event_regime"
            or state.get("score_sha256") != score_sha
            or state.get("queue_fingerprint") != queue_fp
            or state.get("score_fingerprint") != score_fp
            or report.get("status") != "complete"
            or report.get("experiment") != "SW0159_one_shot_event_regime"
            or report.get("seed_scope") != [1]
            or report.get("ground_truth_opened_after_all_predictions_validated") is not True
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or report.get("implementation_fingerprint") != score_fp
            or set(report.get("scores", {})) != {"control", "candidate", "source97", "SW0143_T"}
            or set(report.get("means", {})) != {"control", "candidate", "source97", "SW0143_T"}
            or any(set(report["means"][name]) != set(score_contract.METRICS)
                   for name in ("control", "candidate", "source97", "SW0143_T"))
            or any(not math.isfinite(float(value)) for group in report["means"].values()
                   for value in group.values())):
        raise ValueError("SW0159 terminal score or its registered provenance is invalid")
    return {"score_path": str(score_path.resolve()), "score_sha256": score_sha,
        "score_queue_fingerprint": queue_fp, "score_implementation_fingerprint": score_fp,
        "status": "complete", "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True}


def validate_preflight(path=PREFLIGHT_PATH, *, score_dependency=None):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    upstream = preflight.validate_upstream()
    if not preflight.validate_report_record(report,
            expected_fingerprint=preflight.implementation_fingerprint(), expected_upstream=upstream):
        raise ValueError("SW0160 preflight is stale or failed strict source validation")
    if report.get("training_admission") is not True:
        raise ValueError("SW0160 disposable preflight scientifically rejected training")
    current_score = _score_dependency()
    if score_dependency is not None and score_dependency != current_score:
        raise ValueError("SW0159 completed score changed since SW0160 admission")
    dependency = current_score
    if dependency.get("status") != "complete":
        raise ValueError("SW0160 requires the registered SW0159 completed score")
    return report, dependency


def validate_preflight_admission(path=PREFLIGHT_PATH):
    """Public queue admission API; requires both frozen preflight and SW0159 score."""
    return validate_preflight(path)[0]


def _load_foundation(device):
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[idx]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED)
        torch.cuda.manual_seed(CORE_INIT_SEED)
        foundation = sw155._load_foundation(dev)
    graph = foundation.wrapped.core.graph_generator
    if graph is None:
        raise ValueError("SW0160 requires source graph parameters")
    geodesic_operator.install_operator(graph, "T")
    foundation.patcher.eval()
    foundation.wrapped.eval()
    return foundation


def _state_sha(parameters):
    return source_training.parameter_hash(list(parameters))


def _module_sha(module):
    return source_training.state_sha(module)


def _parameter_change(before, params):
    total = 0.0
    for prior, current in zip(before, params):
        delta = current.detach().float() - prior
        total += float(delta.double().square().sum().cpu())
    return math.sqrt(total)


def _logical_update(*, optimizer, families, compute_micro, wrapped):
    """SW0155 four-micro update contract, allowing honest zero family gradients."""
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro in range(BATCH // MICRO):
        loss = compute_micro(micro)
        if not torch.is_tensor(loss) or loss.ndim or not torch.isfinite(loss):
            raise FloatingPointError("SW0160 objective must be a finite scalar")
        (loss / (BATCH // MICRO)).backward()
        losses.append(float(loss.detach().cpu()))
        del loss
    norms = {}
    for name, params in families.items():
        sq = 0.0
        for p in params:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"nonfinite SW0160 gradient in {name}")
                sq += float(p.grad.detach().double().square().sum().cpu())
        norms[name] = math.sqrt(sq)
    active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not active:
        raise FloatingPointError("SW0160 logical update produced no gradients")
    preclip = torch.nn.utils.clip_grad_norm_(active, 1.0)
    if not torch.isfinite(preclip):
        raise FloatingPointError("nonfinite SW0160 global gradient norm")
    before = {name: [p.detach().float().clone() for p in params]
              for name, params in families.items()}
    optimizer.step()
    with torch.no_grad():
        wrapped.project_integrations_()
    changes = {name: _parameter_change(before[name], params) for name, params in families.items()}
    return {"mean_loss": sum(losses) / len(losses), "family_gradient_norms": norms,
        "family_parameter_change_l2": changes,
        "global_preclip_gradient_norm": float(preclip.detach().cpu()),
        "clip_scale": min(1.0, 1.0 / max(float(preclip.detach().cpu()), 1e-30))}


def _preflight_bindings(record, path):
    return {"preflight_path": str(Path(path).resolve()), "preflight_sha256": sha256_file(path),
        "preflight_implementation_fingerprint": record["implementation_fingerprint"],
        "preflight_upstream": record["upstream"]}


def validate_optimizer_payload(optimizer_payload, step_summary, arm):
    """Check serialized Adam group identity/rates and completed logical steps."""
    if arm not in model.ARMS or not isinstance(optimizer_payload, dict):
        return False
    expected = [name for name in model.LEARNING_RATES
                if arm == model.ARMS[1] or name != "source_base"]
    groups = optimizer_payload.get("param_groups")
    states = optimizer_payload.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        return False
    if [group.get("name") for group in groups] != expected:
        return False
    if set(step_summary) != set(model.LEARNING_RATES):
        return False
    actual = {name: 0 for name in model.LEARNING_RATES}
    for group in groups:
        name = group["name"]
        if float(group.get("lr", -1)) != model.LEARNING_RATES[name]:
            return False
        steps = []
        for parameter_id in group.get("params", []):
            state = states.get(parameter_id)
            if state is None:
                continue
            step = state.get("step")
            step_value = int(step.item()) if torch.is_tensor(step) else int(step)
            if not 0 <= step_value <= UPDATES:
                return False
            steps.append(step_value)
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if not torch.is_tensor(value) or not torch.isfinite(value).all():
                    return False
        actual[name] = max(steps, default=0)
        if actual[name] != UPDATES:
            return False
    return (actual == step_summary and
            (arm != model.ARMS[0] or actual["source_base"] == 0))


def validate_training_artifact(arm, *, directory=None, preflight_path=PREFLIGHT_PATH,
                               train_root=TRAIN_ROOT, expected_score=None):
    """Strictly validate a completed arm plus checkpoint, optimizer and history bytes."""
    try:
        if arm not in model.ARMS:
            return False
        output = Path(directory) if directory is not None else Path(train_root) / f"{arm}_seed1"
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pre, score = validate_preflight(preflight_path, score_dependency=expected_score)
        src = preflight._source_contract()
        ids = src["ids"]
        rows = [int(x) for x in ids[:TRAIN_COUNT]]
        hdf_stat = Path(sw155.HDF5_PATH).stat()
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict):
            return False
        fp = implementation_fingerprint()
        if (manifest.get("status") != "training_complete" or manifest.get("experiment") != EXPERIMENT
                or manifest.get("arm") != arm or manifest.get("seed") != SEED
                or manifest.get("completed_updates") != UPDATES or manifest.get("logical_batch") != BATCH
                or manifest.get("microbatch") != MICRO or manifest.get("exposures") != BATCH * UPDATES
                or manifest.get("train_ids") != rows or manifest.get("implementation_fingerprint") != fp
                or bindings.get("implementation_fingerprint") != fp
                or bindings.get("preflight_sha256") != sha256_file(preflight_path)
                or bindings.get("score_dependency") != score
                or bindings.get("source_checkpoint_sha256") != src["source_sha256"]
                or bindings.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
                or bindings.get("ordered_train_ids_sha256") != sw155.ids_sha(rows)
                or bindings.get("full_source_train_ids_sha256") != sw155.ids_sha(ids)
                or bindings.get("hdf5_path") != str(Path(sw155.HDF5_PATH).resolve())
                or bindings.get("hdf5_size_bytes") != hdf_stat.st_size
                or bindings.get("hdf5_mtime_ns") != hdf_stat.st_mtime_ns
                or bindings.get("rgb_asset_validation") != sw130.validate_rgb_assets()
                or bindings.get("initial_parameter_sha256") != manifest.get("initial_parameter_sha256")
                or bindings.get("ground_truth_used") is not True
                or bindings.get("ground_truth_scope") != "registered source TRAIN masks only"):
            return False
        if (set(manifest.get("initial_parameter_sha256", {})) != set(model.LEARNING_RATES)
                or set(manifest.get("final_parameter_sha256", {})) != set(model.LEARNING_RATES)
                or set(manifest.get("optimizer_step_summary", {})) != set(model.LEARNING_RATES)
                or any(not isinstance(v, str) or len(v) != 64 for v in
                       (*manifest["initial_parameter_sha256"].values(),
                        *manifest["final_parameter_sha256"].values()))):
            return False
        for family in model.LEARNING_RATES:
            before, after = (manifest["initial_parameter_sha256"][family],
                             manifest["final_parameter_sha256"][family])
            if arm == model.ARMS[0] and family == "source_base":
                if before != after or manifest["optimizer_step_summary"][family] != 0:
                    return False
            elif manifest["optimizer_step_summary"][family] > UPDATES:
                return False
            elif before == after:
                return False
        if arm == model.ARMS[0] and manifest["source_base_before_sha256"] != manifest["source_base_after_sha256"]:
            return False
        if arm == model.ARMS[1] and manifest["source_base_before_sha256"] == manifest["source_base_after_sha256"]:
            return False
        for key, suffix in (("checkpoint", "pt"), ("optimizer", "pt"), ("history", "json")):
            file = output / f"{key}.{suffix}"
            if not file.is_file() or manifest.get(f"{key}_sha256") != sha256_file(file):
                return False
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != UPDATES:
            return False
        for ix, row in enumerate(history):
            if (row.get("update") != ix + 1
                    or row.get("image_ids") != rows[ix * BATCH:(ix + 1) * BATCH]
                    or not math.isfinite(float(row.get("mean_loss", float("nan"))))
                    or set(row.get("family_gradient_norms", {})) != set(model.LEARNING_RATES)
                    or set(row.get("family_parameter_change_l2", {})) != set(model.LEARNING_RATES)
                    or any(not math.isfinite(float(v)) or float(v) < 0
                           for v in row["family_gradient_norms"].values())):
                return False
            if any(not math.isfinite(float(v)) or float(v) < 0
                   for v in row["family_parameter_change_l2"].values()):
                return False
            clip = row.get("feature_clip_diagnostics", {})
            if (not math.isfinite(float(clip.get("overall", float("nan"))))
                    or not 0 <= float(clip["overall"]) <= 1
                    or len(clip.get("per_channel", [])) != 8
                    or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1
                           for v in clip["per_channel"])):
                return False
        family_names = set(model.LEARNING_RATES)
        if (set(manifest.get("initial_parameter_sha256", {})) != family_names
                or set(manifest.get("final_parameter_sha256", {})) != family_names
                or set(manifest.get("optimizer_step_summary", {})) != family_names
                or any(not isinstance(v, str) or len(v) != 64 for v in
                       (*manifest["initial_parameter_sha256"].values(),
                        *manifest["final_parameter_sha256"].values()))):
            return False
        if arm == model.ARMS[0] and (manifest["source_base_before_sha256"]
                                     != manifest["source_base_after_sha256"]):
            return False
        if arm == model.ARMS[1] and (manifest["source_base_before_sha256"]
                                     == manifest["source_base_after_sha256"]):
            return False
        if any(int(v) < 0 or int(v) > UPDATES for v in manifest["optimizer_step_summary"].values()):
            return False
        payload = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        if set(payload) != {"wrapped", "encoder"}:
            return False
        if not all(isinstance(payload[k], dict) for k in payload):
            return False
        opt = torch.load(output / "optimizer.pt", map_location="cpu", weights_only=False)
        if not validate_optimizer_payload(opt, manifest["optimizer_step_summary"], arm):
            return False
        flat = {**{"wrapped." + key: value for key, value in payload["wrapped"].items()},
                **{"encoder." + key: value for key, value in payload["encoder"].items()}}
        base = {name.removeprefix("base_encoder."): value
                for name, value in payload["encoder"].items() if name.startswith("base_encoder.")}
        if (source_training.state_sha(flat) != manifest.get("checkpoint_model_state_sha256")
                or source_training.state_sha(base) != manifest.get("source_base_after_sha256")):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT_PATH,
              hdf5_path=HDF5_PATH, work_root=WORK_ROOT, train_root=TRAIN_ROOT,
              score_dependency=None):
    if arm not in model.ARMS:
        raise ValueError("unregistered SW0160 arm")
    pre, score = validate_preflight(preflight_path, score_dependency=score_dependency)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0160 training requires assigned CUDA")
    torch.cuda.set_device(dev)
    src = preflight._source_contract()
    assets = sw130.validate_rgb_assets()
    ids, rows = src["ids"][:TRAIN_COUNT], src["rows"][:TRAIN_COUNT]
    objective.validate_training_prefix(ids, src["ids"])
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000,128,128,3):
        raise ValueError("SW0160 registered TRAIN RGB cache schema changed")
    hdf5_path = Path(hdf5_path)
    hdf_stat = hdf5_path.stat()
    expected_hdf = pre["source"]["hdf5_identity"]
    if (str(hdf5_path.resolve()) != expected_hdf["path"]
            or hdf_stat.st_size != expected_hdf["size_bytes"]
            or hdf_stat.st_mtime_ns != expected_hdf["mtime_ns"]):
        raise ValueError("SW0160 canonical HDF5 changed after admitted preflight")
    foundation = _load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != assets
            or list(map(int, foundation.image_ids)) != src["ids"]
            or not np.array_equal(foundation.pool_indices, src["rows"])):
        raise ValueError("fresh SW0160 source foundation differs from admitted source/assets/order")
    model.install_encoder(foundation, arm, residual_seed=RESIDUAL_SEED)
    families = model.trainable_families(foundation, arm)
    optimizer = model.make_optimizer(families)
    initial = {name: _state_sha(params) for name, params in families.items()}
    initial_base = _module_sha(foundation.encoder.base_encoder)
    bindings = {"experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha256_file(src["manifest_path"]),
        "full_source_train_ids_sha256": sw155.ids_sha(src["ids"]),
        "ordered_train_ids_sha256": sw155.ids_sha(ids), "rgb_asset_validation": assets,
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_implementation_fingerprint": pre["implementation_fingerprint"],
        "preflight_upstream": pre["upstream"], "score_dependency": score,
        "implementation_fingerprint": implementation_fingerprint(),
        "hdf5_path": str(hdf5_path.resolve()), "hdf5_size_bytes": hdf_stat.st_size,
        "hdf5_mtime_ns": hdf_stat.st_mtime_ns, "updates": UPDATES,
        "logical_batch": BATCH, "microbatch": MICRO, "steps": STEPS,
        "settle": SETTLE, "live_tail": TAIL, "initial_parameter_sha256": initial,
        "source_base_before_sha256": initial_base, "ground_truth_used": True,
        "ground_truth_scope": "registered source TRAIN masks only"}
    runtime = durable.runtime_signature(dev)
    work = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    output = Path(train_root) / f"{arm}_seed1"
    if output.exists():
        raise FileExistsError(f"preserving existing SW0160 training output: {output}")
    work.mkdir(parents=True, exist_ok=True)
    if (work / "boundary.ptx").is_file():
        loaded = durable.load_latest(work, wrapped=foundation.wrapped, encoder=foundation.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if not durable.is_only_uncommitted_temp(work):
            raise FileExistsError("preserving uncommitted SW0160 checkpoint data")
        completed, history = 0, []
        durable.save_boundary(work, completed_updates=0, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, history=history,
            bindings=bindings, runtime=runtime, device=dev)
    if not 0 <= completed <= UPDATES or len(history) != completed:
        raise ValueError("SW0160 checkpoint cursor/history mismatch")
    rgb_rows = np.asarray(cache[rows]).copy()
    if rgb_rows.dtype != np.uint8 or rgb_rows.shape != (TRAIN_COUNT,128,128,3):
        raise ValueError("SW0160 ordered source RGB rows invalid")
    frozen_patcher = _module_sha(foundation.patcher)
    frozen_stats = preflight._feature_preprocess_sha(foundation)
    frozen_contrast = _state_sha([foundation.wrapped.core.graph_generator.geodesic_contrast])
    with h5py.File(hdf5_path, "r") as h5:
        if h5["image"].shape != (100000,128,128,3) or h5["mask"].shape not in (
                (100000,128,128), (100000,128,128,1)):
            raise ValueError("canonical SW0160 supervised HDF5 schema changed")
        for update in range(completed, UPDATES):
            batch_ids = [int(x) for x in ids[update*BATCH:(update+1)*BATCH]]
            def compute_micro(micro):
                offset = update * BATCH + micro * MICRO
                rgb_cpu, masks_cpu = sw155.load_aligned_train_rows(
                    hdf5_path, ids, rgb_rows, offset, MICRO, hdf5_handle=h5)
                labels = objective.modal_train_targets32(masks_cpu).to(dev)
                rgb = rgb_cpu.to(dev)
                gamma = foundation.encode(rgb)
                trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
                loss, _ = objective.supervised_relation_loss(trace, labels, settle=SETTLE)
                del gamma, trace, labels, rgb
                return loss
            row = _logical_update(optimizer=optimizer, families=families,
                compute_micro=compute_micro, wrapped=foundation.wrapped)
            row.update(update=update+1, image_ids=batch_ids)
            row["feature_clip_diagnostics"] = _feature_clip_diagnostics(foundation,
                torch.from_numpy(rgb_rows[update*BATCH:(update+1)*BATCH]))
            history.append(row)
            durable.save_boundary(work, completed_updates=update+1, wrapped=foundation.wrapped,
                encoder=foundation.encoder, optimizer=optimizer, history=history,
                bindings=bindings, runtime=runtime, device=dev)
            if (update + 1) % 16 == 0:
                print(json.dumps({"arm": arm, "completed_updates": update+1,
                    "total_updates": UPDATES}, allow_nan=False), flush=True)
    final_base = _module_sha(foundation.encoder.base_encoder)
    if (len(history) != UPDATES or (arm == model.ARMS[0] and final_base != initial_base)
            or _module_sha(foundation.patcher) != frozen_patcher
            or preflight._feature_preprocess_sha(foundation) != frozen_stats
            or _state_sha([foundation.wrapped.core.graph_generator.geodesic_contrast]) != frozen_contrast):
        raise AssertionError("SW0160 endpoint/frozen source state validation failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    checkpoint, optfile, histfile = staging / "checkpoint.pt", staging / "optimizer.pt", staging / "history.json"
    with checkpoint.open("xb") as stream:
        torch.save({"wrapped": foundation.wrapped.state_dict(), "encoder": foundation.encoder.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with optfile.open("xb") as stream:
        torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
    with histfile.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    final = {name: _state_sha(params) for name, params in families.items()}
    manifest = {"status": "training_complete", "experiment": EXPERIMENT, "arm": arm,
        "seed": SEED, "completed_updates": UPDATES, "logical_batch": BATCH,
        "microbatch": MICRO, "exposures": BATCH*UPDATES, "train_ids": [int(x) for x in ids],
        "bindings": bindings, "initial_parameter_sha256": initial,
        "final_parameter_sha256": final, "source_base_before_sha256": initial_base,
        "source_base_after_sha256": final_base, "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_model_state_sha256": source_training.state_sha({
            **{"wrapped." + key: value for key, value in foundation.wrapped.state_dict().items()},
            **{"encoder." + key: value for key, value in foundation.encoder.state_dict().items()}}),
        "optimizer_sha256": sha256_file(optfile), "history_sha256": sha256_file(histfile),
        "implementation_fingerprint": implementation_fingerprint(),
        "optimizer_step_summary": {name: max((int(optimizer.state[p]["step"].item())
            for p in params if p in optimizer.state), default=0) for name, params in families.items()},
        "ground_truth_used": True, "ground_truth_scope": "registered source TRAIN masks only"}
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.rename(staging, output)
    if not validate_training_artifact(arm, directory=output, preflight_path=preflight_path,
                                     expected_score=score):
        raise ValueError("SW0160 saved training artifact failed strict validation")
    return manifest


def _feature_clip_diagnostics(foundation, rgb_uint8):
    with torch.no_grad():
        device = next(foundation.encoder.parameters()).device
        rgb = rgb_uint8.to(device).permute(0,3,1,2).float() / 255.
        feat = foundation.encoder(rgb)
        normalized = (feat - foundation.feature_mean) / foundation.feature_std
        clipped = normalized.abs() > foundation.feature_clip
        return {"overall": float(clipped.float().mean().cpu()),
                "per_channel": clipped.float().mean(dim=(0, 2, 3)).cpu().tolist()}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=model.ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--score-dependency", type=Path)
    args = parser.parse_args(argv)
    score_dep = None
    if args.score_dependency is not None:
        score_dep = json.loads(args.score_dependency.read_text(encoding="utf-8"))
    record = train_arm(args.arm, device=args.device, preflight_path=args.preflight,
                       score_dependency=score_dep)
    print(json.dumps({"status": record["status"], "arm": args.arm,
                      "completed_updates": record["completed_updates"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
