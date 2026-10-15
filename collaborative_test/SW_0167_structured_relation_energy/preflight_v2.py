"""Two-update TRAIN-only admission for SW0167's split feature provenance."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from typing import Any
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0167_structured_relation_energy import (
    feature_modes, qcc_readout, structured_energy, training_objective,
)
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed


ARMS = ("structured_hinge", "local_margin")
HEAD_SEED = 167001
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0167_structured_relation_energy/DESIGN_KO.md",
        "collaborative_test/SW_0167_structured_relation_energy/feature_modes.py",
        "collaborative_test/SW_0167_structured_relation_energy/structured_energy.py",
        "collaborative_test/SW_0167_structured_relation_energy/training_objective.py",
        "collaborative_test/SW_0167_structured_relation_energy/preflight_v2.py",
        "collaborative_test/SW_0167_structured_relation_energy/audit_train_cache.py",
        "collaborative_test/SW_0167_structured_relation_energy/qcc_readout.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0162_connected_signed_partition/connected_signed.py",
        "snn_kuramoto_bidirectional/evaluation.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    paths = [ROOT / rel for rel in rels]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0167 preflight dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _sha_tensor(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _sha_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _import_preserving_cuda_visibility(module_name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    module = importlib.import_module(module_name)
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return module


def prepare_train_targets(prepared: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Read masks for the exact first eight TRAIN IDs and project onto source F."""
    sw130 = _import_preserving_cuda_visibility(
        "collaborative_test.SW_0130_phase_state_integration.run")
    sw155_objective = _import_preserving_cuda_visibility(
        "collaborative_test.SW_0155_supervised_spike_relations.objective")
    sw155_train = _import_preserving_cuda_visibility(
        "collaborative_test.SW_0155_supervised_spike_relations.train")
    audit_train_cache = _import_preserving_cuda_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")

    ids = [int(value) for value in prepared["selected_ids"]]
    if len(ids) != 8:
        raise ValueError("SW0167 preflight is fixed to the first eight registered TRAIN rows")
    checkpoint, manifest_file, manifest, pool_rows, ordered_ids, source_sha = sw130.source_contract(1)
    del checkpoint, manifest_file, manifest, source_sha
    if ids != [int(value) for value in ordered_ids[:8]]:
        raise ValueError("preflight IDs differ from the seed-1 source TRAIN prefix")
    rgb = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or tuple(rgb.shape) != (70000, 128, 128, 3):
        raise ValueError("registered TRAIN RGB cache schema changed")
    rgb_rows = np.asarray(rgb[np.asarray(pool_rows[:8], dtype=np.int64)]).copy()
    _, masks = sw155_train.load_aligned_train_rows(
        sw155_train.HDF5_PATH, ids, rgb_rows, 0, 8)
    del rgb
    gt32 = sw155_objective.modal_train_targets32(torch.as_tensor(masks)).cpu().numpy()
    # The registered support is read from production Q recomputed on the
    # original contiguous B4 TRAIN cache groups. Do not substitute cached Q
    # labels or VAL-style B1 arithmetic for this target projection.
    supports, targets = [], []
    for start in (0, 4):
        b4 = np.stack(prepared["component_spikes"][start:start + 4])
        q_b4 = audit_train_cache._recompute_cpu_q(b4)
        for offset in range(4):
            index = start + offset
            source_labels = qcc_readout.qcc_labels_from_affinity(q_b4[offset:offset + 1])[0].numpy()
            if source_labels.shape != (32, 32):
                raise ValueError("CPU B4 production-Q readout returned unexpected source labels")
            target = structured_energy.feasible_target(source_labels, gt32[index].reshape(32, 32))
            supports.append(source_labels.astype(np.int32, copy=False))
            targets.append(target)
    return np.stack(supports), targets


def run_two_update_preflight(prepared: dict[str, Any], supports, targets, *, device) -> dict[str, Any]:
    """Check cached-source parity and disposable updates for both registered losses.

    ``prepared`` must be the real eight-row output of ``audit_train_cache``;
    this function does not load validation images or validation masks.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SW0167 two-update admission must execute on the leased GPU")
    if len(prepared.get("selected_ids", [])) != 8 or len(prepared.get("component_spikes", [])) != 8:
        raise ValueError("preflight requires exactly eight validated source-cache rows")
    if len(supports) != 8 or len(targets) != 8:
        raise ValueError("preflight requires eight TRAIN-derived feasible targets")

    features_by_row, e0_by_row, wa_by_row, source_rows = [], [], [], []
    solver_rows = []
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    for index in range(8):
        source_labels = np.asarray(supports[index])
        target_grid = np.asarray(targets[index])
        if (source_labels.dtype.kind not in "iu" or source_labels.shape != (32, 32)
                or target_grid.dtype.kind not in "iu" or target_grid.shape != (32, 32)
                or np.any(source_labels < 0) or np.any(target_grid < 0)):
            raise ValueError(f"TRAIN row {index} source/target labels are invalid")
        q_np, a_np = prepared["contexts"][index]
        q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
        spikes = torch.as_tensor(prepared["component_spikes"][index],
                                 dtype=torch.float32, device=device)
        spikes_b1 = spikes.unsqueeze(0)
        rho = feature_modes.component_rho(spikes)
        q_b1 = spike_synchrony_affinity(
            spikes_b1.mean(dim=1), spikes_b1,
            settle=0, affinity_mode="spike")[0]
        strict_features = structured_energy.pair_features(spikes, q_b1, a, prepared["gain"])
        same_mode_features = feature_modes.features_from_rho_and_context_q(
            rho, q_b1, a, prepared["gain"])
        if not torch.equal(strict_features, same_mode_features):
            raise ValueError(f"TRAIN row {index} split builder differs from strict same-mode features")
        features = feature_modes.features_from_rho_and_context_q(
            rho, q, a, prepared["gain"])
        e0, wa = structured_energy.energy_components(q, a, prepared["gain"])
        labels = np.asarray(supports[index], dtype=np.int32)
        oracle = foreground_signed.foreground_induced_merge(
            q_np, a_np, prepared["gain"], labels, grid_shape=(32, 32))
        zero_head_result = structured_energy.solve_energy(
            e0.detach().cpu().numpy(), labels, grid_shape=(32, 32))
        reference_e0 = structured_energy.base_energy(q_np, a_np, prepared["gain"])
        if (not np.array_equal(oracle["pair_energy"], reference_e0)
                or not np.array_equal(e0.detach().cpu().numpy(), reference_e0)
                or not np.array_equal(oracle["labels"], zero_head_result["labels"])
                or oracle["merge_log"] != zero_head_result["merge_log"]):
            raise ValueError(f"TRAIN row {index} zero-head source energy changed")
        features_by_row.append(features.detach())
        e0_by_row.append(e0.detach())
        wa_by_row.append(wa.detach())
        source_rows.append({
            "cache_index": index,
            "image_id": int(prepared["selected_ids"][index]),
            "q_sha256": prepared["rows"][index]["q_sha256"],
            "source_energy_sha256": _sha_tensor(e0),
            "feature_sha256": _sha_tensor(features),
            "cpu_b4_source_F_sha256": _sha_array(source_labels),
            "foreground_patch_count": int(np.count_nonzero(labels)),
            "source_foreground_exact": True,
            "source_zero_head_labels_exact": True,
            "source_zero_head_merge_log_exact": True,
        })
        solver_rows.append({
            "cache_index": index,
            "zero_residual_energy_exact": True,
            "source_qcc_foreground_count": int(np.count_nonzero(labels)),
            "target_foreground_support_count": int(np.count_nonzero(supports[index])),
        })
        del q, a, spikes, spikes_b1, rho, q_b1, strict_features, same_mode_features, features, e0, wa

    arm_reports = {}
    for arm_index, arm in enumerate(ARMS):
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(HEAD_SEED)
            head = structured_energy.PairRelationMLP().to(device)
        finally:
            torch.random.set_rng_state(rng_state)
        initial_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
        step_reports = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in range(8):
                energy, residual = head(features_by_row[row], e0_by_row[row], wa_by_row[row])
                if step == 1 and (not torch.equal(energy, e0_by_row[row])
                                  or torch.count_nonzero(residual).item() != 0):
                    raise AssertionError("zero-initialized SW0167 head changed source E0")
                if arm == "structured_hinge":
                    loss = training_objective.structured_hinge_gold_ties(
                        energy, e0_by_row[row],
                        torch.as_tensor(targets[row], device=device),
                        torch.as_tensor(supports[row], device=device))
                else:
                    loss = structured_energy.local_margin_loss(
                        energy, e0_by_row[row],
                        torch.as_tensor(targets[row], device=device),
                        torch.as_tensor(supports[row], device=device))
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(f"{arm} preflight loss is invalid")
                (loss / 8.0).backward()
                losses.append(float(loss.detach().cpu()))
            grad_norms = {}
            for name, parameter in head.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"{arm} {name} gradient is nonfinite")
                    grad_norms[name] = float(parameter.grad.detach().double().norm().cpu())
            output_grad = grad_norms.get("output.weight", 0.0) + grad_norms.get("output.bias", 0.0)
            hidden_grad = grad_norms.get("hidden.weight", 0.0) + grad_norms.get("hidden.bias", 0.0)
            if output_grad <= 0:
                raise ValueError(f"{arm} has no output-layer credit at disposable step {step}")
            if step == 2 and hidden_grad <= 0:
                raise ValueError(f"{arm} has no hidden-layer credit at disposable step 2")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            changed = {name: float((value.detach() - initial_state[name]).double().norm().cpu())
                       for name, value in head.state_dict().items()}
            if step == 1 and changed["output.weight"] + changed["output.bias"] <= 0:
                raise ValueError(f"{arm} output layer did not change at disposable step 1")
            if step == 2 and changed["hidden.weight"] + changed["hidden.bias"] <= 0:
                raise ValueError(f"{arm} hidden layer did not change by disposable step 2")
            step_reports.append({
                "step": step,
                "mean_loss": float(sum(losses) / len(losses)),
                "gradient_norms": grad_norms,
                "parameter_delta_from_initial": changed,
            })
        arm_reports[arm] = {
            "initial_state_sha256": hashlib.sha256(b"".join(
                value.detach().cpu().contiguous().numpy().tobytes()
                for _, value in sorted(initial_state.items()))).hexdigest(),
            "steps": step_reports,
            "both_steps_finite": True,
            "output_changed": any(step_reports[-1]["parameter_delta_from_initial"][name] > 0
                                  for name in ("output.weight", "output.bias")),
            "hidden_changed_after_second_step": any(
                step_reports[-1]["parameter_delta_from_initial"][name] > 0
                for name in ("hidden.weight", "hidden.bias")),
        }
        del head, optimizer

    if arm_reports[ARMS[0]]["initial_state_sha256"] != arm_reports[ARMS[1]]["initial_state_sha256"]:
        raise AssertionError("matched arms did not start from identical head initialization")
    return {
        "experiment": "SW0167_split_feature_two_update_preflight",
        "status": "complete_admitted",
        "selected_image_ids": [int(value) for value in prepared["selected_ids"]],
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "source_state_sha256": prepared["source_state_sha256"],
        "context_state_sha256": prepared["context_state_sha256"],
        "gain": float(prepared["gain"]),
        "feature_mode": "rho_from_cached_S_B1; context_Q_and_A_from_registered_original_B4",
        "source_F_mode": "production CPU float32 Q on original TRAIN B4 groups, then production QCC readout",
        "same_feature_bytes_for_both_arms": True,
        "source_foreground_and_zero_residual_parity": solver_rows,
        "feature_sha256_by_row": [row["feature_sha256"] for row in source_rows],
        "rows": source_rows,
        "arms": arm_reports,
        "training_ground_truth_scope": "TRAIN masks for exactly the first eight source IDs; no VAL masks",
        "validation_data_read": False,
        "optimizer_updates": 2,
        "interpretation": "Disposable head-credit admission only; no model-performance claim.",
    }


def run_registered_preflight(*, physical_gpu: int) -> dict[str, Any]:
    """Load only the registered eight TRAIN cache rows/masks and run admission.

    Intended for an owner-queue child that inherits a single leased GPU in
    ``CUDA_VISIBLE_DEVICES``. The source/cache and mask admission happens before
    CUDA initialization; no validation rows or validation masks are opened.
    """
    if type(physical_gpu) is not int or physical_gpu not in (0, 1, 2, 3):
        raise ValueError("physical GPU must be one of the registered four devices")
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned != str(physical_gpu):
        raise RuntimeError("preflight child does not see its leased physical GPU")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before source and TRAIN admission")
    audit = _import_preserving_cuda_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    prepared = audit._validate_upstream_and_rows()
    supports, targets = prepare_train_targets(prepared)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assigned or torch.cuda.is_initialized():
        raise RuntimeError("CPU source/mask admission changed GPU visibility or initialized CUDA")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = run_two_update_preflight(prepared, supports, targets, device=device)
    report["physical_gpu"] = physical_gpu
    report["cuda_visible_devices"] = assigned
    report["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["implementation_fingerprint"] = implementation_fingerprint()
    return report


def _atomic_json(path, value):
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.output is None:
        raise SystemExit("preflight must be launched by its owner queue with --worker and --output")
    result = run_registered_preflight(physical_gpu=args.physical_gpu)
    _atomic_json(args.output, result)
