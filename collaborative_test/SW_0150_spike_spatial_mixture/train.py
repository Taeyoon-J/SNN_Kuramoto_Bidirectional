"""Durable SW0150 source-free native32 mixture-likelihood training arms."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0150_spike_spatial_mixture import preflight
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import (
    gaussian_mixture_nll, make_decoder,
)

NAME = preflight.EXPERIMENT
SEED = 1
ARMS = ("encoder_graph", "joint")
BATCH, MICRO, UPDATES = 16, 4, 256
STEPS, SETTLE, TAIL = 1024, 512, 64
WARM_UPDATES, WARM_EXPOSURES = 32, 512
PREFLIGHT = HERE / "results_archive" / "preflight_seed1.json"
WARM_ARTIFACT = Path(os.environ.get(
    "SW0150_PRIVATE_RUNTIME",
    "/Data0/kevinswk/collaborative_private_runtime_20261008/SW0150_preflight_20261012"
)) / "decoder_warm_seed1.pt"
TRAIN = HERE / "training"
WORK = HERE / "work"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def implementation_fingerprint():
    # Reuse the frozen foundation's complete transitive source fingerprint,
    # then add this experiment's assignment/objective, admission, queue and
    # durable-update dependencies.
    result = dict(sw135.implementation_fingerprint())
    paths = (
        HERE / "train.py", HERE / "protocol.json", HERE / "preflight.py",
        HERE / "preflight_queue.py", HERE / "integration.py",
        HERE / "spatial_diffusion.py", HERE / "seed_selection.py", HERE / "likelihood.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        ROOT / "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0150 training dependency missing: {missing[0]}")
    result.update({path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    return result


def implementation_sha():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_record():
    checkpoint, manifest_path, manifest, rows, ids, core_sha = sw130.source_contract(SEED)
    rows = np.asarray(rows, dtype=np.int64)
    ids = [int(value) for value in ids]
    if len(ids) != BATCH * UPDATES or len(set(ids)) != len(ids):
        raise ValueError("SW0150 requires the fixed 4096 ordered source TRAIN IDs")
    if len(rows) != len(ids) or len(np.unique(rows)) != len(rows):
        raise ValueError("SW0150 source TRAIN pool rows must be unique and aligned")
    return {"checkpoint": Path(checkpoint), "manifest_path": Path(manifest_path),
            "manifest": manifest, "rows": rows, "ids": ids, "core_sha256": core_sha}


def validate_training_inputs(preflight_path=PREFLIGHT, warm_path=WARM_ARTIFACT):
    """Bind the successful no-GT preflight, exact source, parity, assets and warm bytes."""
    preflight_path, warm_path = Path(preflight_path), Path(warm_path)
    if warm_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("shared SW0150 warm checkpoint must remain outside the repository")
    raw_report = preflight_path.read_bytes()
    report = json.loads(raw_report.decode("utf-8-sig"))
    src = _source_record()
    assets = sw130.validate_rgb_assets()
    manifest_sha = sha(src["manifest_path"])
    if (report.get("experiment") != NAME
            or report.get("stage") != "source1_shared_warm_and_disposable_preflight"
            or report.get("status") != "source_preflight_complete"
            or report.get("seed") != SEED
            or report.get("training_admission") is not False
            or report.get("ground_truth_used") is not False
            or report.get("masks_read") is not False
            or report.get("implementation_fingerprint") != preflight.implementation_fingerprint()):
        raise ValueError("SW0150 source preflight status or implementation changed")
    source = report.get("source", {})
    native_provenance = source.get("native32_provenance", {})
    if (source.get("checkpoint_sha256") != src["core_sha256"]
            or source.get("manifest_sha256") != manifest_sha
            or source.get("training_ids") != src["ids"]
            or source.get("training_ids_sha256") != ids_sha(src["ids"])
            or source.get("pool_rows_sha256") != hashlib.sha256(
                np.asarray(src["rows"], dtype="<i8").tobytes()).hexdigest()
            or source.get("rgb_asset_validation") != assets
            or native_provenance.get("experiment") != "SW0135_native32_spike_binding"
            or native_provenance.get("seed") != SEED
            or native_provenance.get("source_core_sha256") != src["core_sha256"]
            or native_provenance.get("source_manifest_sha256") != manifest_sha
            or native_provenance.get("source_training_ids_sha256") != ids_sha(src["ids"])
            or native_provenance.get("implementation_fingerprint")
            != sw135.implementation_fingerprint()):
        raise ValueError("SW0150 source checkpoint, ordered IDs, rows, or RGB assets changed")
    parity_record, parity_sha = preflight.load_and_validate_parity(
        preflight.PARITY_PATH, source_sha=src["core_sha256"],
        source_manifest_sha=manifest_sha, training_ids=src["ids"], asset_validation=assets)
    saved_parity = report.get("saved_source_parity", {})
    if (saved_parity.get("path") != str(preflight.PARITY_PATH)
            or saved_parity.get("sha256") != parity_sha
            or saved_parity.get("status") != "native32_zero_adapter_parity_passed"
            or saved_parity.get("source1_actual_source_exact_T1024_settle512") is not True):
        raise ValueError("SW0150 preflight does not bind the frozen exact native32 parity")

    if not warm_path.is_file():
        raise FileNotFoundError("shared private SW0150 decoder warm artifact is missing")
    warm_sha = sha(warm_path)
    warm_meta = report.get("warm_checkpoint", {})
    if (warm_meta.get("path") != str(warm_path)
            or warm_meta.get("sha256") != warm_sha
            or warm_meta.get("shared_exactly_across_disposable_arms") is not True
            or warm_meta.get("optimizer_state_steps", {}).get("maximum") != WARM_UPDATES):
        raise ValueError("preflight report does not bind the current shared warm checkpoint")
    try:
        payload = torch.load(warm_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError("shared warm checkpoint cannot be loaded") from exc
    expected_warm = {
        "schema": 1, "experiment": NAME, "status": "shared_decoder_warmup_complete",
        "seed": SEED, "ground_truth_used": False, "training_admission": False,
        "optimizer_updates": WARM_UPDATES, "warmup_exposures": WARM_EXPOSURES,
        "warmup_image_ids": src["ids"][:WARM_EXPOSURES],
        "warmup_image_ids_sha256": ids_sha(src["ids"][:WARM_EXPOSURES]),
        "disposable_image_ids": src["ids"][:32],
        "source_core_sha256": src["core_sha256"],
        "source_manifest_sha256": manifest_sha,
        "source_training_ids_sha256": ids_sha(src["ids"]),
        "rgb_asset_validation": assets,
        "saved_sw0135_parity_sha256": parity_sha,
        "preflight_implementation_fingerprint": preflight.implementation_fingerprint(),
        "sigma2": report.get("likelihood", {}).get("sigma2"),
    }
    if any(payload.get(key) != value for key, value in expected_warm.items()):
        raise ValueError("warm checkpoint source, order, assets, code, or fixed sigma2 differs")
    if (not isinstance(payload.get("decoder_state"), dict)
            or not isinstance(payload.get("optimizer_state"), dict)
            or not isinstance(payload.get("rng_state"), dict)
            or not isinstance(payload["rng_state"].get("cpu"), torch.Tensor)
            or not isinstance(payload["rng_state"].get("cuda"), torch.Tensor)):
        raise ValueError("warm checkpoint lacks decoder, Adam, or RNG state")
    if payload.get("decoder_initial_sha256") != warm_meta.get("decoder_initial_sha256"):
        raise ValueError("warm checkpoint initial decoder digest differs from its report")
    decoder = make_decoder(8, seed=preflight.DECODER_INIT_SEED)
    try:
        decoder.load_state_dict(payload["decoder_state"], strict=True)
        warm_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        warm_optimizer.load_state_dict(payload["optimizer_state"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise ValueError("warm decoder weights or Adam parameter schema changed") from exc
    steps = preflight._optimizer_steps(warm_optimizer, list(decoder.parameters()))
    if (steps.get("count") != len(list(decoder.parameters()))
            or steps.get("minimum") != WARM_UPDATES
            or steps.get("maximum") != WARM_UPDATES):
        raise ValueError("shared decoder Adam state is not exactly at warm update 32")
    if (payload.get("decoder_optimizer_state_steps") != steps
            or len(payload.get("warmup_mean_nll_by_update", [])) != WARM_UPDATES
            or any(not math.isfinite(float(value))
                   for value in payload["warmup_mean_nll_by_update"])):
        raise ValueError("warm optimizer steps or 32-update loss history is inconsistent")
    if report.get("optimizer_updates") != {
            "shared_decoder_warmup": WARM_UPDATES,
            "encoder_graph": preflight.DISPOSABLE_UPDATES,
            "joint": preflight.DISPOSABLE_UPDATES}:
        raise ValueError("preflight exposure/update counts differ from registration")
    # Every selected family must receive positive gradient credit. Hash-change
    # eligibility is narrower for the three constrained integration scalars:
    # tiny, valid updates can leave their serialized values unchanged.
    required_gradients = {
        "encoder_graph": {"encoder", "graph", "decoder"},
        "joint": {"encoder", "graph", "native_core", "integration_scalars", "decoder"},
    }
    required_changes = {
        "encoder_graph": {"encoder", "graph", "decoder"},
        "joint": {"encoder", "graph", "native_core", "decoder"},
    }
    all_families = {"encoder", "graph", "native_core", "integration_scalars", "decoder"}
    arm_records = report.get("arms", {})
    if set(arm_records) != set(required_gradients):
        raise ValueError("preflight must contain the two registered disposable arms")
    for arm_name, required in required_gradients.items():
        row = arm_records[arm_name]
        if (row.get("status") != "disposable_two_updates_complete"
                or row.get("optimizer_updates") != 2
                or len(row.get("history", [])) != 2
                or row.get("image_ids") != src["ids"][:32]):
            raise ValueError(f"preflight {arm_name} feasibility run is incomplete")
        initial = row.get("initial_parameter_group_sha256", {})
        final = row.get("final_parameter_group_sha256", {})
        changed = row.get("changed_parameter_groups", {})
        if set(initial) != all_families or set(final) != all_families or set(changed) != all_families:
            raise ValueError(f"preflight {arm_name} parameter families are incomplete")
        if any(changed[name] is not (initial[name] != final[name]) for name in all_families):
            raise ValueError(f"preflight {arm_name} changed-family flags disagree with hashes")
        if any(initial[name] == final[name] for name in required_changes[arm_name]):
            raise ValueError(f"preflight {arm_name} failed a registered parameter update")
        if arm_name == "encoder_graph" and any(
                initial[name] != final[name] for name in ("native_core", "integration_scalars")):
            raise ValueError("encoder_graph preflight changed frozen native/scalar families")
        for history_row in row["history"]:
            norms = history_row.get("family_gradient_norms", {})
            if any(name not in norms or not math.isfinite(float(norms[name]))
                   or float(norms[name]) <= 0 for name in required):
                raise ValueError(f"preflight {arm_name} lacks selected-family gradient credit")
            if set(norms) != all_families:
                raise ValueError(f"preflight {arm_name} gradient-family receipt is incomplete")
    return {"report": report, "report_sha256": hashlib.sha256(raw_report).hexdigest(),
            "warm": payload, "warm_sha256": warm_sha, "source": src,
            "assets": assets, "parity": parity_record, "parity_sha256": parity_sha,
            "sigma2": float(expected_warm["sigma2"])}


def families(foundation, decoder):
    return preflight._families(foundation, decoder)


def trainable_parameter_families(family_map, arm):
    if arm not in ARMS:
        raise ValueError("unknown SW0150 arm")
    names = ["encoder", "graph"]
    if arm == "joint":
        names.extend(("native_core", "integration_scalars"))
    names.append("decoder")
    return {name: family_map[name] for name in names}


def _group_optimizer(family_map, arm):
    selected = trainable_parameter_families(family_map, arm)
    rates = {"encoder": 3e-6, "graph": 3e-5, "native_core": 3e-5,
             "integration_scalars": 1e-3, "decoder": 1e-3}
    group_rows = [{"params": selected[name], "lr": rates[name], "name": name}
                  for name in selected]
    params = [parameter for group in group_rows for parameter in group["params"]]
    if (not params or len({id(param) for param in params}) != len(params)
            or any(not parameter.requires_grad for parameter in params)):
        raise ValueError("SW0150 selected optimizer families overlap or are frozen")
    return torch.optim.Adam(group_rows), params


def _state_sha(module):
    return preflight._state_sha(module)


def _group_hashes(foundation, decoder):
    return preflight._group_hashes(foundation, decoder)


def _parameter_sha(parameters):
    return preflight._state_dict_sha({f"{i:08d}": parameter for i, parameter in enumerate(parameters)})


def _family_steps(optimizer, family_map):
    return {name: preflight._optimizer_steps(optimizer, params) for name, params in family_map.items()}


def _gradient_norm(parameters):
    return preflight._grad_norm(parameters)


def _bindings(arm, admission, foundation, initial_hashes, runtime):
    return {
        "experiment": NAME, "arm": arm, "seed": SEED,
        "preflight_sha256": admission["report_sha256"],
        "preflight_implementation_fingerprint": preflight.implementation_fingerprint(),
        "preflight_warm_sha256": admission["warm_sha256"],
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
        "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
        "ordered_train_ids_sha256": ids_sha(admission["source"]["ids"]),
        "saved_native32_parity_sha256": admission["parity_sha256"],
        "sigma2": admission["sigma2"], "warm_updates": WARM_UPDATES,
        "warm_exposures": WARM_EXPOSURES, "initial_parameter_sha256": initial_hashes,
        "batch": BATCH, "microbatch": MICRO, "updates": UPDATES,
        "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "optimizer": {"encoder": 3e-6, "graph": 3e-5, "native_core": 3e-5,
                      "integration_scalars": 1e-3, "decoder": 1e-3,
                      "native_clip_norm": 1.0, "decoder_clip_norm": 1.0},
        "optimizer_groups": list(trainable_parameter_families(families(foundation, foundation.wrapped.spatial_decoder), arm)),
        "runtime_sha256": durable.canonical_sha(runtime),
        "runtime_signature": runtime,
        "implementation_fingerprint": implementation_fingerprint(),
        "implementation_sha256": implementation_sha(),
        "ground_truth_used": False,
    }


def logical_update(update, *, optimizer, family_map, selected, ordered_ids,
                   compute_micro, wrapped, arm, commit=None):
    """One logical update; callback runs only after backward/clip/step/projection."""
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("update index must be a nonnegative integer")
    if len(ordered_ids) != BATCH:
        raise ValueError("each logical update must bind exactly 16 image IDs")
    optimizer.zero_grad(set_to_none=True)
    losses, micro_rows = [], []
    for micro in range(BATCH // MICRO):
        loss, details = compute_micro(micro)
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("SW0150 Gaussian likelihood must be a finite scalar")
        (loss / (BATCH // MICRO)).backward()
        losses.append(float(loss.detach().cpu()))
        micro_rows.append(details or {})
        del details, loss
    norms = {name: _gradient_norm(params) for name, params in family_map.items()}
    native = [param for name, params in selected.items() if name != "decoder" for param in params]
    decoder = selected["decoder"]
    native_with_grad = [param for param in native if param.grad is not None]
    decoder_with_grad = [param for param in decoder if param.grad is not None]
    if native_with_grad:
        clip_grad_norm_(native_with_grad, 1.0)
    if decoder_with_grad:
        clip_grad_norm_(decoder_with_grad, 1.0)
    optimizer.step()
    if arm == "joint":
        with torch.no_grad():
            wrapped.project_integrations_()
    for name, params in family_map.items():
        if any(not torch.isfinite(param).all() for param in params):
            raise FloatingPointError(f"nonfinite SW0150 parameter after update: {name}")
    row = {"update": update + 1, "image_ids": [int(v) for v in ordered_ids],
           "microbatches": micro_rows, "mean_nll": float(np.mean(losses)),
           "family_gradient_norms": norms,
           "optimizer_steps": _family_steps(optimizer, family_map),
           "parameter_group_sha256": {name: _parameter_sha(params)
                                      for name, params in family_map.items()}}
    if commit is not None:
        commit(row)
    return row


def _load_foundation(device):
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.random.default_generator.manual_seed(preflight.CORE_INIT_SEED)
        torch.cuda.manual_seed(preflight.CORE_INIT_SEED)
        foundation = sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)
    return foundation


def _install_shared_warm(foundation, admission, arm, device):
    warm = admission["warm"]
    decoder = make_decoder(8, seed=preflight.DECODER_INIT_SEED).to(device)
    if hasattr(foundation.wrapped, "spatial_decoder"):
        raise ValueError("fresh SW0150 foundation already contains a decoder")
    foundation.wrapped.add_module("spatial_decoder", decoder)
    decoder.load_state_dict(warm["decoder_state"], strict=True)
    family_map = families(foundation, decoder)
    preflight._set_trainability(foundation, decoder, arm)
    optimizer, _params = _group_optimizer(family_map, arm)
    warm_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    warm_optimizer.load_state_dict(warm["optimizer_state"])
    transplanted = preflight.transplant_decoder_adam(
        warm_optimizer, optimizer, list(decoder.parameters()),
        list(decoder.parameters()))
    if transplanted != len(list(decoder.parameters())):
        raise ValueError("not every shared warm decoder Adam state was transferred")
    torch.cuda.set_rng_state(warm["rng_state"]["cuda"].cpu(), device=device)
    torch.random.set_rng_state(warm["rng_state"]["cpu"].cpu())
    return decoder, family_map, optimizer, transplanted


def _batch_rows(raw, rows, start, device):
    values = np.asarray(raw[rows[start:start + MICRO]]).copy()
    if values.dtype != np.uint8 or values.shape != (MICRO, 128, 128, 3):
        raise ValueError("source TRAIN cache yielded a malformed clean RGB microbatch")
    return torch.from_numpy(values).to(device)


def _optimizer_step_summary(optimizer, groups):
    return {name: preflight._optimizer_steps(optimizer, params) for name, params in groups.items()}


def _validate_optimizer_checkpoint(arm, optimizer_state, schema, optimizer_rates,
                                   recorded_summary):
    """Validate actual Adam group membership/state, including inactive families."""
    family_names = ("encoder", "graph", "native_core", "integration_scalars", "decoder")
    expected_selected = trainable_parameter_families(
        {name: [] for name in family_names}, arm)
    groups = optimizer_state.get("param_groups", [])
    state_rows = optimizer_state.get("state", {})
    if not isinstance(schema, list) or len(groups) != len(schema):
        raise ValueError("SW0150 optimizer parameter-group schema is missing")
    if [group.get("name") for group in groups] != list(expected_selected):
        raise ValueError("SW0150 optimizer group names/order differ from the registered arm")
    actual_summary = {name: {"count": 0, "minimum": None, "maximum": None}
                      for name in family_names}
    all_seen_ids = []
    for group, schema_row in zip(groups, schema):
        name = group.get("name")
        param_ids = group.get("params", [])
        shapes = schema_row.get("parameter_shapes", [])
        dtypes = schema_row.get("parameter_dtypes", [])
        if (schema_row.get("name") != name or len(param_ids) != len(shapes)
                or len(param_ids) != len(dtypes)
                or float(group.get("lr", -1)) != float(optimizer_rates[name])):
            raise ValueError(f"SW0150 optimizer group schema differs for {name}")
        all_seen_ids.extend(param_ids)
        steps = []
        for param_id, shape, dtype in zip(param_ids, shapes, dtypes):
            state = state_rows.get(param_id, {})
            if not state:
                continue
            step = state.get("step")
            if torch.is_tensor(step):
                if step.numel() != 1 or not torch.isfinite(step).all():
                    raise ValueError(f"SW0150 optimizer {name} step is malformed")
                step_value = int(step.item())
            else:
                if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
                    raise ValueError(f"SW0150 optimizer {name} step is malformed")
                step_value = int(step)
            maximum = WARM_UPDATES + UPDATES if name == "decoder" else UPDATES
            minimum = WARM_UPDATES + UPDATES if name == "decoder" else 1
            if not minimum <= step_value <= maximum:
                raise ValueError(f"SW0150 optimizer {name} step is outside its registered range")
            steps.append(step_value)
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if (not torch.is_tensor(value) or list(value.shape) != shape
                        or str(value.dtype) != dtype or not torch.isfinite(value).all()):
                    raise ValueError(f"SW0150 optimizer {name} {key} schema/values differ")
        actual_summary[name] = {"count": len(steps), "minimum": min(steps) if steps else None,
                                "maximum": max(steps) if steps else None}
    if len(set(all_seen_ids)) != len(all_seen_ids) or not set(state_rows).issubset(set(all_seen_ids)):
        raise ValueError("SW0150 optimizer parameter IDs are duplicated or unregistered")
    if actual_summary != recorded_summary:
        raise ValueError("SW0150 optimizer state steps disagree with manifest summary")
    if actual_summary["decoder"]["maximum"] != WARM_UPDATES + UPDATES:
        raise ValueError("SW0150 decoder Adam did not continue from shared warm step 32 to 288")
    if (actual_summary["decoder"]["count"] != len(groups[-1].get("params", []))
            or actual_summary["decoder"]["minimum"] != WARM_UPDATES + UPDATES):
        raise ValueError("SW0150 decoder Adam parameters did not all reach step 288")
    for family in expected_selected:
        if actual_summary[family]["maximum"] is None:
            raise ValueError(f"SW0150 selected Adam family has no recorded step: {family}")
    return actual_summary


def _finalize(arm, *, foundation, optimizer, family_map, selected, history,
              bindings, output_dir, work_dir, initial_hashes, before_frozen):
    if len(history) != UPDATES or output_dir.exists():
        raise ValueError("SW0150 cannot finalize incomplete or pre-existing training output")
    final_hashes = _group_hashes(foundation, foundation.wrapped.spatial_decoder)
    if arm == "encoder_graph":
        for name in ("native_core", "integration_scalars"):
            if final_hashes[name] != initial_hashes[name]:
                raise AssertionError(f"encoder_graph arm changed frozen {name}")
    frozen_after = _parameter_sha([p for name in ("native_core", "integration_scalars")
                                   for p in family_map[name]])
    if arm == "encoder_graph" and frozen_after != before_frozen:
        raise AssertionError("encoder_graph source core/scalar parameters changed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint_path = staging / "checkpoint.pt"
        optimizer_path = staging / "optimizer.pt"
        history_path = staging / "history.json"
        with checkpoint_path.open("xb") as stream:
            torch.save({"wrapped": foundation.wrapped.state_dict(),
                        "encoder": foundation.encoder.state_dict()}, stream)
            stream.flush(); os.fsync(stream.fileno())
        with optimizer_path.open("xb") as stream:
            torch.save(optimizer.state_dict(), stream)
            stream.flush(); os.fsync(stream.fileno())
        with history_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(history, stream, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        manifest = {
            "status": "training_complete", "experiment": NAME, "arm": arm, "seed": SEED,
            "updates": UPDATES, "completed_updates": UPDATES, "logical_batch": BATCH,
            "microbatch": MICRO, "exposures": BATCH * UPDATES,
            "shared_warm_updates": WARM_UPDATES, "shared_warm_exposures": WARM_EXPOSURES,
            "train_ids": [image_id for row in history for image_id in row["image_ids"]],
            "source_core_sha256": bindings["source_core_sha256"],
            "source_manifest_sha256": bindings["source_manifest_sha256"],
            "source_training_ids_sha256": bindings["source_training_ids_sha256"],
            "ordered_train_ids_sha256": bindings["ordered_train_ids_sha256"],
            "rgb_asset_validation": bindings["rgb_asset_validation"],
            "preflight_sha256": bindings["preflight_sha256"],
            "preflight_warm_sha256": bindings["preflight_warm_sha256"],
            "initial_parameter_sha256": initial_hashes,
            "final_parameter_sha256": final_hashes,
            "frozen_source_sha256_before": before_frozen,
            "frozen_source_sha256_after": frozen_after,
            "optimizer_step_summary": _optimizer_step_summary(optimizer, family_map),
            "optimizer_schema": durable.optimizer_schema(optimizer),
            "runtime_signature": bindings["runtime_signature"],
            "checkpoint_sha256": sha(checkpoint_path),
            "optimizer_sha256": sha(optimizer_path), "history_sha256": sha(history_path),
            "implementation_fingerprint": bindings["implementation_fingerprint"],
            "implementation_sha256": bindings["implementation_sha256"],
            "bindings": bindings, "ground_truth_used": False,
        }
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        # Retain incomplete unique staging directories for forensic review.
        raise


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT,
              warm_path=WARM_ARTIFACT, output_root=TRAIN, work_root=WORK):
    if arm not in ARMS:
        raise ValueError("unregistered SW0150 training arm")
    admission = validate_training_inputs(preflight_path, warm_path)
    output_dir = Path(output_root) / f"{arm}_seed1"
    work_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    if output_dir.exists():
        raise FileExistsError(f"preserve existing SW0150 completed training output: {output_dir}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0150 main training requires its assigned CUDA device")
    torch.cuda.set_device(dev)
    foundation = _load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != admission["source"]["core_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha(admission["source"]["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != admission["assets"]
            or list(map(int, foundation.image_ids)) != admission["source"]["ids"]
            or not np.array_equal(foundation.pool_indices, admission["source"]["rows"])):
        raise ValueError("fresh SW0150 foundation no longer matches preflight source/order/assets")
    decoder, family_map, optimizer, transferred = _install_shared_warm(
        foundation, admission, arm, dev)
    initial_hashes = _group_hashes(foundation, decoder)
    expected_initial = admission["report"]["arms"][arm]["initial_parameter_group_sha256"]
    if initial_hashes != expected_initial:
        raise ValueError("fresh source plus shared warm decoder differs from disposable initialization")
    selected = trainable_parameter_families(family_map, arm)
    before_frozen = _parameter_sha([p for name in ("native_core", "integration_scalars")
                                    for p in family_map[name]])
    runtime = durable.runtime_signature(dev)
    bindings = _bindings(arm, admission, foundation, initial_hashes, runtime)
    source_ids = admission["source"]["ids"]
    rows = admission["source"]["rows"]
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered SW0150 source TRAIN RGB cache schema changed")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    boundary = work_dir / "boundary.ptx"
    if boundary.exists():
        metadata = durable.inspect_committed_metadata(work_dir)
        if metadata.get("bindings") != bindings or metadata.get("runtime") != runtime:
            raise ValueError("SW0150 checkpoint source/warm/code/runtime binding differs")
        state = durable.load_latest(work_dir, wrapped=foundation.wrapped,
                                    encoder=foundation.encoder, optimizer=optimizer,
                                    expected_bindings=bindings, expected_runtime=runtime,
                                    device=dev, max_updates=UPDATES)
        completed, history = state["completed_updates"], state["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise ValueError("SW0150 checkpoint directory has uncommitted unknown files")
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=False)
        completed, history = 0, []
        durable.save_boundary(work_dir, completed_updates=0, wrapped=foundation.wrapped,
                              encoder=foundation.encoder, optimizer=optimizer, history=history,
                              bindings=bindings, runtime=runtime, device=dev)
    if completed == UPDATES:
        result = _finalize(arm, foundation=foundation, optimizer=optimizer,
                           family_map=family_map, selected=selected, history=history,
                           bindings=bindings, output_dir=output_dir, work_dir=work_dir,
                           initial_hashes=initial_hashes, before_frozen=before_frozen)
        return {"status": "training_complete", "manifest": result}

    foundation.wrapped.train()
    foundation.encoder.train()
    started = time.perf_counter()

    def commit(row):
        history.append(row)
        durable.save_boundary(work_dir, completed_updates=row["update"],
                              wrapped=foundation.wrapped, encoder=foundation.encoder,
                              optimizer=optimizer, history=history, bindings=bindings,
                              runtime=runtime, device=dev)
        if row["update"] % 16 == 0:
            print(json.dumps({"stage": "train", "arm": arm,
                              "update": row["update"], "total": UPDATES,
                              "elapsed_seconds": time.perf_counter() - started,
                              "warm_decoder_states_transferred": transferred}), flush=True)

    raw = cache
    for update in range(completed, UPDATES):
        ids = source_ids[update * BATCH:(update + 1) * BATCH]

        def compute_micro(micro):
            start = update * BATCH + micro * MICRO
            rgb = _batch_rows(raw, rows, start, dev)
            result = forward_assignment(foundation, rgb, live_tail_steps=TAIL)
            loss = gaussian_mixture_nll(result["gamma_content"], result["p"], rgb,
                                        decoder, sigma2=admission["sigma2"])
            if not torch.isfinite(loss):
                raise FloatingPointError("SW0150 main Gaussian mixture likelihood is nonfinite")
            return loss, {"image_ids": ids[micro * MICRO:(micro + 1) * MICRO]}

        logical_update(update, optimizer=optimizer, family_map=family_map,
                       selected=selected, ordered_ids=ids, compute_micro=compute_micro,
                       wrapped=foundation.wrapped, arm=arm, commit=commit)
    result = _finalize(arm, foundation=foundation, optimizer=optimizer,
                       family_map=family_map, selected=selected, history=history,
                       bindings=bindings, output_dir=output_dir, work_dir=work_dir,
                       initial_hashes=initial_hashes, before_frozen=before_frozen)
    return {"status": "training_complete", "manifest": result}


def validate_training(arm, *, output_root=TRAIN, work_root=WORK,
                      preflight_path=PREFLIGHT, warm_path=WARM_ARTIFACT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0150 training arm")
    admission = validate_training_inputs(preflight_path, warm_path)
    output = Path(output_root) / f"{arm}_seed1"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint, optimizer_path, history_path = (output / "checkpoint.pt",
                                                  output / "optimizer.pt",
                                                  output / "history.json")
    expected_ids = admission["source"]["ids"]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    bindings = manifest.get("bindings", {})
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != NAME
            or manifest.get("arm") != arm or manifest.get("seed") != SEED
            or manifest.get("completed_updates") != UPDATES
            or manifest.get("logical_batch") != BATCH or manifest.get("microbatch") != MICRO
            or manifest.get("exposures") != UPDATES * BATCH
            or manifest.get("shared_warm_updates") != WARM_UPDATES
            or manifest.get("shared_warm_exposures") != WARM_EXPOSURES
            or manifest.get("train_ids") != expected_ids
            or manifest.get("checkpoint_sha256") != sha(checkpoint)
            or manifest.get("optimizer_sha256") != sha(optimizer_path)
            or manifest.get("history_sha256") != sha(history_path)
            or manifest.get("preflight_sha256") != admission["report_sha256"]
            or manifest.get("preflight_warm_sha256") != admission["warm_sha256"]
            or manifest.get("source_core_sha256") != admission["source"]["core_sha256"]
            or manifest.get("source_manifest_sha256") != sha(admission["source"]["manifest_path"])
            or manifest.get("source_training_ids_sha256") != ids_sha(expected_ids)
            or manifest.get("rgb_asset_validation") != admission["assets"]
            or manifest.get("implementation_sha256") != implementation_sha()
            or bindings.get("implementation_sha256") != implementation_sha()
            or bindings.get("implementation_fingerprint") != implementation_fingerprint()
            or bindings.get("preflight_sha256") != admission["report_sha256"]
            or bindings.get("preflight_warm_sha256") != admission["warm_sha256"]
            or bindings.get("rgb_asset_validation") != admission["assets"]
            or len(history) != UPDATES
            or [item for row in history for item in row.get("image_ids", [])] != expected_ids):
        raise ValueError(f"invalid SW0150 completed training lineage/artifacts: {arm}")
    boundary = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    metadata = durable.inspect_committed_metadata(boundary)
    if (metadata.get("completed_updates") != UPDATES
            or metadata.get("bindings") != bindings
            or metadata.get("runtime") != bindings.get("runtime_signature")
            or metadata.get("runtime_sha256") != durable.canonical_sha(
                bindings.get("runtime_signature"))):
        raise ValueError("SW0150 durable final checkpoint differs from completed manifest")
    try:
        optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError("SW0150 serialized optimizer artifact cannot be loaded") from exc
    _validate_optimizer_checkpoint(
        arm, optimizer_state, manifest.get("optimizer_schema"), bindings["optimizer"],
        manifest.get("optimizer_step_summary"))
    return {"manifest": manifest, "manifest_sha256": sha(manifest_path),
            "checkpoint": checkpoint, "checkpoint_sha256": sha(checkpoint),
            "optimizer": optimizer_path, "history": history_path}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = train_arm(args.arm, device=args.device)
    print(json.dumps({"status": result["status"], "arm": args.arm,
                      "training_admission": True, "updates": UPDATES,
                      "ground_truth_used": False}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
