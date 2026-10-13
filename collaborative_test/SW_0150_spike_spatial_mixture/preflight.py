"""No-GT native32 calibration, shared warm head, and disposable feasibility check."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding import parity as sw135_parity
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import (
    calibrate_sigma2, gaussian_mixture_nll, make_decoder,
)


EXPERIMENT = "SW0150_spike_spatial_mixture"
SEED = 1
CORE_INIT_SEED = 149001
DECODER_INIT_SEED = 150001
GRID_NODES = 1024
IMAGE_SHAPE = (128, 128, 3)
WARM_UPDATES = 32
WARM_BATCH = 16
MICRO_BATCH = 4
WARM_EXPOSURES = WARM_UPDATES * WARM_BATCH
DISPOSABLE_UPDATES = 2
DISPOSABLE_EXPOSURES = DISPOSABLE_UPDATES * WARM_BATCH
PARITY_PATH = (ROOT / "collaborative_test/SW_0135_native32_spike_binding"
               / "results_archive/native32_zero_adapter_parity_seed1_b4.json")
OUTPUT_DEFAULT = HERE / "results_archive" / "preflight_seed1.json"
WARM_DEFAULT = HERE / "results_archive" / "private_work" / "decoder_warm_seed1.pt"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def implementation_fingerprint():
    """Bind the preflight to the foundation, source readers and new likelihood path."""
    paths = (
        HERE / "preflight.py", HERE / "protocol.json", HERE / "integration.py",
        HERE / "spatial_diffusion.py", HERE / "seed_selection.py", HERE / "likelihood.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/parity.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preflight dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def validate_saved_parity_payload(record, *, source_sha, source_manifest_sha,
                                  training_ids, asset_validation, expected_fingerprint):
    """Fail closed unless the existing source1 B4 1024-step parity receipt binds now."""
    if not isinstance(record, dict):
        raise TypeError("saved native32 parity receipt must be a JSON object")
    ids = [int(value) for value in training_ids]
    if len(ids) != 4096 or len(set(ids)) != 4096:
        raise ValueError("preflight requires the exact 4096-image source TRAIN order")
    if (record.get("experiment") != "SW0135_native32_spike_binding"
            or record.get("stage") != "native32_zero_adapter_parity"
            or record.get("status") != "native32_zero_adapter_parity_passed"
            or record.get("source_seed") != SEED
            or record.get("source_core_sha256") != source_sha
            or record.get("source_manifest_sha256") != source_manifest_sha
            or record.get("training_image_ids") != ids[:4]
            or record.get("training_image_ids_sha256") != _ids_sha(ids[:4])
            or record.get("registered_train_order_sha256") != _ids_sha(ids)
            or record.get("rgb_asset_validation") != asset_validation
            or record.get("implementation_fingerprint") != expected_fingerprint
            or record.get("gamma_shape") != [4, 8, GRID_NODES]
            or record.get("microbatch_size") != 4
            or record.get("ground_truth_used") is not False
            or record.get("masks_read") is not False
            or record.get("optimizer_updates") != 0
            or record.get("training_admission") is not False):
        raise ValueError("saved SW0135 parity receipt does not bind the current source contract")
    horizons = record.get("horizons")
    if not isinstance(horizons, list) or not any(
            isinstance(row, dict) and row.get("steps") == 1024 and row.get("settle") == 512
            for row in horizons):
        raise ValueError("saved SW0135 parity receipt lacks the registered T1024/settle512 check")
    arms = record.get("arms")
    if not isinstance(arms, dict) or not {"phase", "constant"}.issubset(arms):
        raise ValueError("saved SW0135 parity receipt is missing source-adapter arms")
    for name in ("phase", "constant"):
        rows = arms[name]
        match = next((row for row in rows if row.get("steps") == 1024 and row.get("settle") == 512), None)
        if not isinstance(match, dict) or match.get("exact_parity") is not True:
            raise ValueError(f"saved SW0135 {name} arm lacks exact T1024 parity")
        differences = match.get("differences")
        if not isinstance(differences, dict) or not differences or any(
                not isinstance(item, dict) or item.get("exact") is not True
                for item in differences.values()):
            raise ValueError(f"saved SW0135 {name} arm contains a non-exact parity field")
    return True


def load_and_validate_parity(path, *, source_sha, source_manifest_sha,
                             training_ids, asset_validation):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_saved_parity_payload(
        payload, source_sha=source_sha, source_manifest_sha=source_manifest_sha,
        training_ids=training_ids, asset_validation=asset_validation,
        expected_fingerprint=sw135_parity.implementation_fingerprint())
    return payload, sha256_file(path)


def transplant_decoder_adam(warm_optimizer, pilot_optimizer, warm_parameters,
                             pilot_parameters):
    """Merge warm Adam state_dict IDs into the pilot decoder group and reload.

    Optimizer.load_state_dict performs the supported parameter-to-device state
    placement; no state tensors are assigned directly to optimizer internals.
    """
    source = list(warm_parameters)
    target = list(pilot_parameters)
    if not source or len(source) != len(target):
        raise ValueError("warm and pilot decoder parameter lists must have equal nonzero length")
    if len({id(parameter) for parameter in source}) != len(source) or len(
            {id(parameter) for parameter in target}) != len(target):
        raise ValueError("optimizer parameter lists must not contain duplicates")
    if len(warm_optimizer.param_groups) != 1 or len(warm_optimizer.param_groups[0]["params"]) != len(source):
        raise ValueError("warm Adam must have exactly the registered decoder parameters")
    target_decoder_groups = [group for group in pilot_optimizer.param_groups
                             if group.get("name") == "decoder"]
    if (len(target_decoder_groups) != 1
            or len(target_decoder_groups[0]["params"]) != len(target)):
        raise ValueError("pilot Adam must expose exactly one matching decoder group")
    warm_state = warm_optimizer.state_dict()
    pilot_state = pilot_optimizer.state_dict()
    warm_ids = warm_state["param_groups"][0]["params"]
    pilot_group = next(group for group in pilot_state["param_groups"]
                       if group.get("name") == "decoder")
    pilot_ids = pilot_group["params"]
    if len(warm_ids) != len(pilot_ids):
        raise ValueError("warm and pilot decoder state_dict parameter order differs")
    copied = 0
    for old_id, new_id in zip(warm_ids, pilot_ids):
        state = warm_state["state"].get(old_id)
        if state:
            pilot_state["state"][new_id] = copy.deepcopy(state)
            copied += 1
    if not copied:
        raise ValueError("warm Adam has no decoder moments to transfer")
    pilot_optimizer.load_state_dict(pilot_state)
    return copied


def _state_sha(module):
    return _state_dict_sha(module.state_dict())


def _state_dict_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _group_hashes(foundation, decoder):
    core = foundation.wrapped.core
    native_core_state = {name: value for name, value in core.state_dict().items()
                         if not name.startswith("graph_generator.")}
    return {
        "encoder": _state_sha(foundation.encoder),
        "graph": _state_sha(core.graph_generator),
        "native_core": _state_dict_sha(native_core_state),
        "integration_scalars": hashlib.sha256(torch.cat((
            foundation.wrapped.a_d.detach().cpu().reshape(-1),
            foundation.wrapped.a_m.detach().cpu().reshape(-1),
            foundation.wrapped.b.detach().cpu().reshape(-1))).contiguous().numpy().tobytes()).hexdigest(),
        "decoder": _state_sha(decoder),
    }


def _families(foundation, decoder):
    core = foundation.wrapped.core
    return {
        "encoder": list(foundation.encoder.parameters()),
        "graph": list(core.graph_generator.parameters()),
        "native_core": [p for name, p in core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m,
                                foundation.wrapped.b],
        "decoder": list(decoder.parameters()),
    }


def _set_trainability(foundation, decoder, arm):
    if arm not in ("encoder_graph", "joint"):
        raise ValueError("arm must be encoder_graph or joint")
    joint = arm == "joint"
    foundation.encoder.requires_grad_(True)
    foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(joint)
    foundation.wrapped.core.graph_generator.requires_grad_(True)
    decoder.requires_grad_(True)
    foundation.encoder.train()
    foundation.patcher.eval()
    foundation.wrapped.train()
    decoder.train()
    if not joint:
        foundation.wrapped.a_d.requires_grad_(False)
        foundation.wrapped.a_m.requires_grad_(False)
        foundation.wrapped.b.requires_grad_(False)


def _optimizer(foundation, decoder, arm):
    families = _families(foundation, decoder)
    named = [
        ("encoder", families["encoder"], 3e-6),
        ("graph", families["graph"], 3e-5),
    ]
    if arm == "joint":
        named += [
            ("native_core", families["native_core"], 3e-5),
            ("integration_scalars", families["integration_scalars"], 1e-3),
        ]
    named.append(("decoder", families["decoder"], 1e-3))
    groups = [{"params": params, "lr": lr, "name": name} for name, params, lr in named]
    params = [parameter for group in groups for parameter in group["params"]]
    if (not params or len({id(parameter) for parameter in params}) != len(params)
            or any(not parameter.requires_grad for parameter in params)):
        raise ValueError("preflight optimizer groups must be unique and trainable")
    return torch.optim.Adam(groups), families, params


def _grad_norm(parameters):
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise FloatingPointError("preflight received a nonfinite gradient")
            total += float(grad.double().square().sum().cpu())
            found = True
    return math.sqrt(total) if found else 0.0


def _finite_model(*modules):
    for module in modules:
        for value in list(module.parameters()) + list(module.buffers()):
            if not torch.isfinite(value).all():
                raise FloatingPointError("preflight model parameter or buffer is nonfinite")


def _batch_rows(raw, rows, start, device):
    rgb = np.asarray(raw[rows[start:start + MICRO_BATCH]]).copy()
    if rgb.dtype != np.uint8 or tuple(rgb.shape[1:]) != IMAGE_SHAPE:
        raise ValueError("registered source cache returned invalid clean RGB")
    return torch.from_numpy(rgb).to(device)


def _forward_loss(foundation, rgb, decoder, sigma2, *, live_tail_steps):
    result = forward_assignment(foundation, rgb, live_tail_steps=live_tail_steps)
    loss = gaussian_mixture_nll(result["gamma_content"], result["p"], rgb,
                                decoder, sigma2=sigma2)
    if not torch.isfinite(loss):
        raise FloatingPointError("preflight spatial mixture likelihood is nonfinite")
    return loss, result


def _optimizer_steps(optimizer, parameters):
    steps = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if "step" in state:
            step = state["step"]
            steps.append(int(step.detach().cpu()) if torch.is_tensor(step) else int(step))
    return {"count": len(steps), "minimum": min(steps) if steps else None,
            "maximum": max(steps) if steps else None}


def _atomic_bytes(path, write_payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserve existing preflight artifact: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp")
    try:
        with temp.open("xb") as stream:
            write_payload(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        try:
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        if temp.exists():
            temp.unlink()
    return sha256_file(path)


def _atomic_json(path, payload):
    def write(stream):
        stream.write((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return _atomic_bytes(path, write)


def _atomic_torch(path, payload):
    return _atomic_bytes(path, lambda stream: torch.save(payload, stream))


def _rng_snapshot(device):
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).cpu().clone()}


def _rng_restore(state, device):
    torch.random.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device=device)


def _load_foundation(device):
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED)
        torch.cuda.manual_seed(CORE_INIT_SEED)
        return sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)


def _warmup(foundation, decoder, raw, rows, sigma2, device):
    foundation.encoder.requires_grad_(False)
    foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(False)
    foundation.encoder.eval()
    foundation.patcher.eval()
    foundation.wrapped.eval()
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    losses = []
    first_summary = None
    for update in range(WARM_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        update_losses = []
        for micro in range(0, WARM_BATCH, MICRO_BATCH):
            start = update * WARM_BATCH + micro
            rgb = _batch_rows(raw, rows, start, device)
            with torch.no_grad():
                result = forward_assignment(foundation, rgb, live_tail_steps=0)
            loss = gaussian_mixture_nll(result["gamma_content"], result["p"], rgb,
                                        decoder, sigma2=sigma2)
            if not torch.isfinite(loss):
                raise FloatingPointError("warmup likelihood is nonfinite")
            (loss / (WARM_BATCH // MICRO_BATCH)).backward()
            update_losses.append(float(loss.detach().cpu()))
            if first_summary is None:
                first_summary = _assignment_summary(result)
            del result, loss, rgb
        grad_norm = _grad_norm(list(decoder.parameters()))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("warmup decoder gradient norm is nonfinite")
        clip_grad_norm_([p for p in decoder.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        _finite_model(decoder)
        losses.append(float(np.mean(update_losses)))
    return optimizer, losses, first_summary


def _assignment_summary(result):
    p = result["p"].detach()
    labels = result["qcc"].detach().cpu().reshape(p.shape[0], -1)
    return {
        "seed_counts": [len(row) for row in result["seed_ids"]],
        "qcc_region_counts": [int(torch.unique(row).numel()) for row in labels],
        "p_column_mass": p.sum(dim=1).cpu().tolist(),
        "primary_label_counts": [torch.bincount(row, minlength=p.shape[-1]).cpu().tolist()
                                  for row in result["primary_labels"].detach().cpu().reshape(p.shape[0], -1)],
    }


def _event_fraction(trace, wrapped):
    threshold = 0.06 * torch.exp(wrapped.b.detach().reshape(1, 4, 1, 1))
    membrane = trace["component_membrane"].detach()
    gates = trace["component_gates"].detach()
    if membrane.shape != gates.shape or membrane.shape[1] != 4:
        raise ValueError("event diagnostic requires native [B,4,N,T] membrane/gate traces")
    events = (membrane > threshold).to(torch.float32)
    return {"threshold_event_fraction": float(events.mean().cpu()),
            "gate_mean": float(gates.float().mean().cpu())}


def _disposable_arm(arm, warm_payload, raw, rows, sigma2, device,
                    source_initial_hashes):
    foundation = _load_foundation(device)
    decoder = make_decoder(8, seed=DECODER_INIT_SEED).to(device)
    initial_hashes = _group_hashes(foundation, decoder)
    for name in ("encoder", "graph", "native_core", "integration_scalars"):
        if initial_hashes[name] != source_initial_hashes[name]:
            raise ValueError(f"fresh {arm} source initialization differs for {name}")
    decoder.load_state_dict(warm_payload["decoder_state"], strict=True)
    _set_trainability(foundation, decoder, arm)
    optimizer, families, optimizer_parameters = _optimizer(foundation, decoder, arm)
    # Restore the serialized warm moments before copying them into the pilot's
    # decoder group; native optimizer groups intentionally start fresh.
    warm_source = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    warm_source.load_state_dict(warm_payload["optimizer_state"])
    copied = transplant_decoder_adam(warm_source, optimizer,
                                     list(decoder.parameters()), list(decoder.parameters()))
    _rng_restore(warm_payload["rng_state"], device)
    initial_group_hashes = _group_hashes(foundation, decoder)
    history = []
    first_summary = None
    event_before = None
    for update in range(DISPOSABLE_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(0, WARM_BATCH, MICRO_BATCH):
            start = update * WARM_BATCH + micro
            rgb = _batch_rows(raw, rows, start, device)
            loss, result = _forward_loss(foundation, rgb, decoder, sigma2,
                                         live_tail_steps=64)
            (loss / (WARM_BATCH // MICRO_BATCH)).backward()
            losses.append(float(loss.detach().cpu()))
            if first_summary is None:
                first_summary = _assignment_summary(result)
                event_before = _event_fraction(result["trace"], foundation.wrapped)
            del result, loss, rgb
        family_norms = {name: _grad_norm(params) for name, params in families.items()}
        if any(not math.isfinite(value) for value in family_norms.values()):
            raise FloatingPointError(f"{arm} disposable family gradient is nonfinite")
        native_params = [parameter for group in optimizer.param_groups
                         if group.get("name") != "decoder" for parameter in group["params"]]
        decoder_params = families["decoder"]
        if native_params:
            clip_grad_norm_([p for p in native_params if p.grad is not None], 1.0)
        clip_grad_norm_([p for p in decoder_params if p.grad is not None], 1.0)
        optimizer.step()
        with torch.no_grad():
            foundation.wrapped.project_integrations_()
        _finite_model(foundation.wrapped, foundation.encoder, decoder)
        history.append({"update": update + 1, "mean_nll": float(np.mean(losses)),
                        "family_gradient_norms": family_norms,
                        "adam_state": {name: _optimizer_steps(optimizer, params)
                                       for name, params in families.items()}})
    final_hashes = _group_hashes(foundation, decoder)
    changed = {name: final_hashes[name] != initial_group_hashes[name]
               for name in initial_group_hashes}
    if arm == "encoder_graph":
        frozen = ("native_core", "integration_scalars")
        if any(changed[name] for name in frozen):
            raise AssertionError("encoder_graph disposable trial changed frozen native state")
    # A no-grad observation on the same first logical batch records diagnostics,
    # without adding training updates or consulting semantic labels.
    with torch.no_grad():
        rgb = _batch_rows(raw, rows, 0, device)
        observation = forward_assignment(foundation, rgb, live_tail_steps=64)
        event_after = _event_fraction(observation["trace"], foundation.wrapped)
        final_summary = _assignment_summary(observation)
        post_loss = gaussian_mixture_nll(observation["gamma_content"], observation["p"],
                                         rgb, decoder, sigma2=sigma2)
    if not torch.isfinite(post_loss):
        raise FloatingPointError(f"{arm} final disposable observation is nonfinite")
    result = {
        "status": "disposable_two_updates_complete",
        "arm": arm,
        "optimizer_updates": DISPOSABLE_UPDATES,
        "image_ids": [int(v) for v in warm_payload["disposable_image_ids"]],
        "history": history,
        "initial_parameter_group_sha256": initial_group_hashes,
        "final_parameter_group_sha256": final_hashes,
        "changed_parameter_groups": changed,
        "initial_assignment_diagnostics": first_summary,
        "final_assignment_diagnostics": final_summary,
        "initial_event_diagnostic": event_before,
        "final_event_diagnostic": event_after,
        "final_mean_nll": float(post_loss.detach().cpu()),
        "warm_decoder_adam_parameter_states_transplanted": copied,
        "decoder_adam_steps_after_two_updates": _optimizer_steps(optimizer, decoder_params),
    }
    del observation, post_loss, rgb, foundation, decoder, optimizer
    return result


def run_preflight(*, device="cuda:0", output=OUTPUT_DEFAULT, warm_output=WARM_DEFAULT,
                  parity_path=PARITY_PATH):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0150 source preflight requires its assigned CUDA device")
    output, warm_output = Path(output), Path(warm_output)
    if output.exists() or warm_output.exists():
        raise FileExistsError("preflight and warm artifacts are create-once")
    torch.cuda.set_device(device)
    foundation = _load_foundation(device)
    checkpoint, manifest_path, source_manifest, rows, ids, source_sha = sw130.source_contract(SEED)
    if (len(rows) != 4096 or len(ids) != 4096
            or ids != source_manifest.get("training_ids")):
        raise AssertionError("source1 TRAIN order must match the frozen 4096-image manifest")
    asset_validation = sw130.validate_rgb_assets()
    if foundation.provenance.get("rgb_asset_validation") != asset_validation:
        raise AssertionError("foundation did not bind the currently verified RGB assets")
    parity, parity_sha = load_and_validate_parity(
        parity_path, source_sha=source_sha, source_manifest_sha=sha256_file(manifest_path),
        training_ids=ids, asset_validation=asset_validation)
    if foundation.provenance.get("source_core_sha256") != source_sha:
        raise AssertionError("foundation source core differs from source_contract")
    if _ids_sha(ids) != parity.get("registered_train_order_sha256"):
        raise AssertionError("saved parity receipt uses a different source image order")

    raw = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    if tuple(raw.shape) != (70000, 128, 128, 3) or raw.dtype != np.uint8:
        raise ValueError("registered clean TRAIN RGB cache has wrong schema")
    first64 = torch.from_numpy(np.asarray(raw[rows[:64]]).copy())
    sigma2 = calibrate_sigma2(first64)
    del first64
    source_initial_hashes = _group_hashes(foundation, make_decoder(8, seed=DECODER_INIT_SEED))
    warm_decoder = make_decoder(8, seed=DECODER_INIT_SEED).to(device)
    warm_initial_sha = _state_sha(warm_decoder)
    warm_optimizer, warm_losses, warm_assignment = _warmup(
        foundation, warm_decoder, raw, rows, sigma2, device)
    after_warm_hashes = _group_hashes(foundation, warm_decoder)
    for group in ("encoder", "graph", "native_core", "integration_scalars"):
        if after_warm_hashes[group] != source_initial_hashes[group]:
            raise AssertionError(f"frozen-source decoder warmup changed {group}")
    warm_rng = _rng_snapshot(device)
    warm_payload = {
        "schema": 1,
        "experiment": EXPERIMENT,
        "status": "shared_decoder_warmup_complete",
        "seed": SEED,
        "ground_truth_used": False,
        "training_admission": False,
        "optimizer_updates": WARM_UPDATES,
        "warmup_exposures": WARM_EXPOSURES,
        "warmup_image_ids": [int(value) for value in ids[:WARM_EXPOSURES]],
        "warmup_image_ids_sha256": _ids_sha(ids[:WARM_EXPOSURES]),
        "disposable_image_ids": [int(value) for value in ids[:DISPOSABLE_EXPOSURES]],
        "source_core_sha256": source_sha,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_training_ids_sha256": _ids_sha(ids),
        "source_initial_parameter_group_sha256": source_initial_hashes,
        "rgb_asset_validation": asset_validation,
        "saved_sw0135_parity_sha256": parity_sha,
        "saved_sw0135_parity_implementation_fingerprint": parity["implementation_fingerprint"],
        "preflight_implementation_fingerprint": implementation_fingerprint(),
        "sigma2": sigma2,
        "decoder_initial_sha256": warm_initial_sha,
        "decoder_state": {k: v.detach().cpu() for k, v in warm_decoder.state_dict().items()},
        "optimizer_state": warm_optimizer.state_dict(),
        "decoder_optimizer_state_steps": _optimizer_steps(warm_optimizer, list(warm_decoder.parameters())),
        "rng_state": warm_rng,
        "warmup_mean_nll_by_update": warm_losses,
        "first_warmup_assignment_diagnostics": warm_assignment,
    }
    warm_sha = _atomic_torch(warm_output, warm_payload)
    source_initial_hashes = warm_payload["source_initial_parameter_group_sha256"]
    disposable_rows = rows[:DISPOSABLE_EXPOSURES]
    arms = {}
    for arm in ("encoder_graph", "joint"):
        arms[arm] = _disposable_arm(arm, warm_payload, raw, disposable_rows,
                                    sigma2, device, source_initial_hashes)
    record = {
        "experiment": EXPERIMENT,
        "stage": "source1_shared_warm_and_disposable_preflight",
        "status": "source_preflight_complete",
        "seed": SEED,
        "device": str(device),
        "training_admission": False,
        "ground_truth_used": False,
        "masks_read": False,
        "optimizer_updates": {"shared_decoder_warmup": WARM_UPDATES,
                              "encoder_graph": DISPOSABLE_UPDATES,
                              "joint": DISPOSABLE_UPDATES},
        "exposures": {"warmup_images": WARM_EXPOSURES,
                      "each_disposable_arm_images": DISPOSABLE_EXPOSURES,
                      "full_native32_70000_image_training": False},
        "source": {
            "checkpoint_sha256": source_sha,
            "manifest_sha256": sha256_file(manifest_path),
            "training_ids_sha256": _ids_sha(ids),
            "training_ids": [int(value) for value in ids],
            "pool_rows_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
            "rgb_asset_validation": asset_validation,
            "native32_provenance": foundation.provenance,
        },
        "saved_source_parity": {
            "path": str(parity_path), "sha256": parity_sha,
            "status": parity["status"],
            "source1_actual_source_exact_T1024_settle512": True,
            "source_image_order_sha256": parity["registered_train_order_sha256"],
        },
        "likelihood": {"sigma2": sigma2,
                       "calibration": "mean of per-image per-channel spatial population variance over exactly first64 registered clean TRAIN uint8 RGB images"},
        "warm_checkpoint": {"path": str(warm_output), "sha256": warm_sha,
                            "decoder_initial_sha256": warm_initial_sha,
                            "optimizer_state_steps": warm_payload["decoder_optimizer_state_steps"],
                            "shared_exactly_across_disposable_arms": True},
        "implementation_fingerprint": implementation_fingerprint(),
        "arms": arms,
        "limitations": [
            "Two disposable updates per arm are feasibility evidence only, not scientific training.",
            "Warmup exposure is reported separately and is not part of the later registered pilot update count.",
            "This preflight does not establish segmentation quality, gate necessity, or 70,000-image scaling.",
        ],
    }
    _finite_model(foundation.wrapped, foundation.encoder, warm_decoder)
    del raw, foundation, warm_decoder, warm_optimizer
    _atomic_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--warm-output", type=Path, default=WARM_DEFAULT)
    parser.add_argument("--parity", type=Path, default=PARITY_PATH)
    args = parser.parse_args(argv)
    result = run_preflight(device=args.device, output=args.output,
                           warm_output=args.warm_output, parity_path=args.parity)
    print(json.dumps({"status": result["status"], "training_admission": False,
                      "output": str(args.output), "warm_output": str(args.warm_output)},
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
