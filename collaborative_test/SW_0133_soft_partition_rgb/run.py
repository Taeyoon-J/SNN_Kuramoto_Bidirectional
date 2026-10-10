"""SW0133 source-bound preflight for genuine soft-partition RGB learning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]

import numpy as np
import torch

from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import SharedRGBDecoder
from collaborative_test.SW_0110_xy_graph_route import run as source97
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import RelativeRGBDecoder
from collaborative_test.SW_0133_soft_partition_rgb.soft_partition import reconstruct_soft
from snn_kuramoto_bidirectional.loss_function import phase_locking_value

SEEDS = (0, 1, 2)
ARMS = ("phase_live", "constant_live", "phase_detached")
BATCH = 16
STEPS = 64
SETTLE = 32
WARMUP_STEPS = 32
CORE_LR = 3e-5
DECODER_LR = 3e-4
CLIP = 1.0
ARCHIVE = HERE / "results_archive"


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
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def implementation_fingerprint():
    fp = dict(sw130.implementation_fingerprint())
    for path in (HERE / "soft_partition.py", HERE / "run.py", HERE / "protocol.json",
                 ROOT / "collaborative_test/SW_0132_partition_relative_rgb/relative_rgb.py"):
        fp[path.relative_to(ROOT).as_posix()] = sha(path)
    return fp


def source_contract(seed):
    checkpoint, manifest_path, manifest, pool, ids, checkpoint_sha = sw130.source_contract(seed)
    if manifest.get("steps") != 256 or manifest.get("source_model_seed") != seed:
        raise AssertionError("SW0097 source must be the registered completed source model")
    return checkpoint, manifest_path, manifest, pool, ids, checkpoint_sha


def _new_decoder_from_registered_initial(old_decoder, device):
    # Same architecture and module order as the SW0106 decoder; seed 106 makes
    # the starting tensor state exactly the registered source initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(106)
        decoder = RelativeRGBDecoder().to(device)
    decoder.load_state_dict(old_decoder.state_dict(), strict=True)
    return decoder


def load_models(seed, device, arm):
    if arm not in ARMS:
        raise ValueError(f"unregistered SW0133 arm: {arm}")
    integration_arm = "constant" if arm == "constant_live" else "phase"
    wrapped, encoder, patcher, mean, std, clip, legacy_decoder, pool, ids, source_sha = \
        sw130.load_models(seed, device, integration_arm)
    decoder = _new_decoder_from_registered_initial(legacy_decoder, device)
    return wrapped, encoder, patcher, mean, std, clip, decoder, pool, ids, source_sha


def _batch_forward(wrapped, encoder, patcher, mean, std, clip, images):
    criterion = sw130.make_criterion()
    gamma = sw130.encode(encoder, patcher, mean, std, clip, images)
    # The integration arm changes only state integration; the registered old
    # primary objective remains phase PLV for both matched arms.
    result = sw130._forward_with_plv(wrapped, gamma, criterion, SETTLE, "phase", "mean")
    groups, spikes, core_out, plv, theta = result
    components = wrapped.core.last_component_spikes
    q = sw130.spike_synchrony_affinity(components.mean(dim=1), components,
                                       settle=SETTLE, affinity_mode="spike")
    labels, hard, detected = sw130.hard_labels(spikes, components, SETTLE)
    target = images.permute(0, 2, 3, 1).contiguous() / 255.0
    return gamma, result, q, labels, hard, detected, target


def _old_loss(result, q):
    _groups, _spikes, _core_out, plv, theta = result
    criterion = sw130.make_criterion()
    primary, _ = criterion(plv=plv, theta=theta)
    positive_q, _ = criterion(plv=q)
    return primary + 5.0 * positive_q


def _finite_norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite gradient in SW0133 preflight")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def _family_norms(named, grads):
    return sw130.family_norm(named, grads)


def _rgb_batch(q, hard, gamma, target, decoder, *, assignment_live=True,
               chunk_size=1024, checkpoint_chunks=True):
    rows = [reconstruct_soft(q[index], hard[index], gamma[index].transpose(0, 1),
                             target[index], decoder, assignment_live=assignment_live,
                             chunk_size=chunk_size, checkpoint_chunks=checkpoint_chunks)
            for index in range(q.shape[0])]
    return (torch.stack([row[0] for row in rows]),
            torch.stack([row[1] for row in rows]).mean(),
            [row[2] for row in rows])


def _seed0_lambda_record(output_path, seed, ids, assets):
    if seed == 0:
        return None, None
    record_path = Path(output_path).parent / "preflight_seed0.json"
    if not record_path.is_file():
        raise RuntimeError("seed1/2 requires a passed seed0 SW0133 lambda record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source0 = source_contract(0)
    expected_ids = source0[4]
    expected_sha = hashlib.sha256(np.asarray(expected_ids, dtype="<i8").tobytes()).hexdigest()
    if (record.get("status") != "passed" or record.get("seed") != 0
            or record.get("experiment") != "SW0133_soft_partition_rgb"
            or record.get("source_steps") != 256
            or record.get("shuffle_seed") != 117
            or record.get("batch_size") != BATCH
            or record.get("train_time_steps") != STEPS
            or record.get("settle_steps") != SETTLE
            or record.get("lambda_source_seed") != 0
            or record.get("implementation_fingerprint") != implementation_fingerprint()
            or record.get("source_core_sha256") != source97.EXPECTED_SOURCE_SHAS[0]
            or record.get("source_manifest_sha256") != sha(source0[1])
            or record.get("training_ids") != expected_ids
            or record.get("training_ids_sha256") != expected_sha
            or record.get("asset_hashes") != assets):
        raise RuntimeError("seed0 SW0133 lambda provenance does not match this frozen run")
    ratios = record.get("lambda_batch_ratios")
    calibrations = record.get("lambda_seed0_calibration")
    if not isinstance(ratios, list) or len(ratios) != 4 \
            or not isinstance(calibrations, list) or len(calibrations) != 4:
        raise RuntimeError("seed0 shared lambda does not match its four-batch calibration record")
    checked_ratios = []
    for batch_index, row in enumerate(calibrations):
        try:
            old_norm = float(row["old_joint_norm"])
            rgb_norm = float(row["rgb_joint_norm"])
            ratio = float(row["lambda_ratio"])
            recorded_ratio = float(ratios[batch_index])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("seed0 lambda calibration row is malformed") from exc
        if (row.get("batch_index") != batch_index
                or not all(math.isfinite(x) and x > 0 for x in
                           (old_norm, rgb_norm, ratio, recorded_ratio))
                or not math.isclose(ratio, 0.25 * old_norm / rgb_norm,
                                    rel_tol=1e-12, abs_tol=0.0)
                or ratio != recorded_ratio):
            raise RuntimeError("seed0 lambda batch ratio is inconsistent with its gradients")
        checked_ratios.append(ratio)
    value = float(record.get("lambda", float("nan")))
    if not math.isfinite(value) or value <= 0 or value != float(statistics.median(checked_ratios)):
        raise RuntimeError("seed0 lambda is not the exact median of its four registered ratios")
    warm_path = Path(record.get("decoder_warmup_artifact", ""))
    if (not warm_path.is_file()
            or sha(warm_path) != record.get("decoder_warmup_artifact_sha256")):
        raise RuntimeError("seed0 SW0133 warm decoder artifact is missing or changed")
    return value, sha(record_path)


def _warm_decoder(seed, device, pool, ids, asset_hashes, gamma_cache, train_cache,
                  wrapped, encoder, patcher, mean, std, clip, decoder, output_path):
    warm_path = Path(output_path).parent / f"preflight_decoder_seed{seed}.pt"
    if warm_path.exists():
        raise FileExistsError(f"preserve existing warm decoder artifact: {warm_path}")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    source_core = {k: v.detach().clone() for k, v in wrapped.core.state_dict().items()}
    source_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    losses = []
    wrapped.eval(); encoder.eval(); decoder.train()
    for step in range(WARMUP_STEPS):
        rows = pool[step * BATCH:(step + 1) * BATCH].tolist()
        images = sw130.read_batch(train_cache, rows, device)
        with torch.no_grad():
            gamma, _result, q, _labels, hard, _groups, target = _batch_forward(
                wrapped, encoder, patcher, mean, std, clip, images)
        optimizer.zero_grad(set_to_none=True)
        _pred, image_losses, _details = _rgb_batch(q, hard, gamma.detach(), target,
                                                    decoder, checkpoint_chunks=True)
        loss = image_losses.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite SW0133 full-RGB decoder warmup loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
        optimizer.step()
        losses.append(float(loss.detach()))
    if any(not torch.equal(value, wrapped.core.state_dict()[key])
           for key, value in source_core.items()):
        raise AssertionError("SW0133 decoder warmup modified source core")
    if any(not torch.equal(value, encoder.state_dict()[key])
           for key, value in source_encoder.items()):
        raise AssertionError("SW0133 decoder warmup modified source encoder")
    payload = {"decoder_state_dict": decoder.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "source_core_sha256": source97.EXPECTED_SOURCE_SHAS[seed],
               "training_ids_sha256": hashlib.sha256(
                   np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
               "asset_hashes": asset_hashes,
               "warmup_updates": WARMUP_STEPS}
    # torch.save is create-only: a prior warmup attempt is never overwritten.
    with warm_path.open("xb") as stream:
        torch.save(payload, stream)
    return warm_path, sha(warm_path), losses


def _paired_initial_arm_check(seed, device, first_images, phase_model):
    """Prove all three arms share initial traces and soft-P RGB forward."""
    models = {"phase_live": phase_model}
    models.update({arm: load_models(seed, device, arm)
                   for arm in ("constant_live", "phase_detached")})
    reference = phase_model
    ref_core, ref_encoder, ref_patcher, ref_mean, ref_std, ref_clip, ref_decoder, *_ = reference
    outputs = {}
    for arm, model in models.items():
        core, encoder, patcher, mean, std, clip, decoder, *_ = model
        for name, reference_module, candidate_module in (
                ("decoder", ref_decoder, decoder), ("core", ref_core, core),
                ("encoder", ref_encoder, encoder)):
            if set(reference_module.state_dict()) != set(candidate_module.state_dict()) or any(
                    not torch.equal(value, candidate_module.state_dict()[key])
                    for key, value in reference_module.state_dict().items()):
                raise AssertionError(f"initial {name} differs across SW0133 arms")
        with torch.no_grad():
            row = _batch_forward(core, encoder, patcher, mean, std, clip, first_images)
            prediction, rgb, details = _rgb_batch(
                row[2], row[4], row[0], row[6], decoder,
                assignment_live=(arm != "phase_detached"))
        outputs[arm] = (row, prediction, rgb, details)
    reference_row, reference_prediction, reference_rgb, reference_details = outputs["phase_live"]
    for arm, (row, prediction, rgb, details) in outputs.items():
        if not torch.equal(reference_row[0], row[0]):
            raise AssertionError(f"initial gamma differs for {arm}")
        for index in (2, 3):
            if not torch.equal(reference_row[index], row[index]):
                raise AssertionError(f"initial Q or labels differ for {arm}")
        if any(not torch.equal(a, b) for a, b in zip(reference_row[4], row[4])):
            raise AssertionError(f"initial hard partition differs for {arm}")
        if not torch.equal(reference_row[6], row[6]):
            raise AssertionError(f"initial RGB target differs for {arm}")
        if not torch.equal(reference_prediction, prediction) or not torch.equal(reference_rgb, rgb):
            raise AssertionError(f"initial live-P RGB output differs for {arm}")
        if any(not torch.equal(a["P"], b["P"])
               for a, b in zip(reference_details, details)):
            raise AssertionError(f"initial soft assignment P differs for {arm}")
        if not torch.equal(_old_loss(reference_row[1], reference_row[2]),
                           _old_loss(row[1], row[2])):
            raise AssertionError(f"initial old-objective loss differs for {arm}")
    # At zero integration state phase and constant controls have exact native
    # trace equality; the detached arm is the same phase forward path.
    for arm in ("constant_live", "phase_detached"):
        a, b = reference_row[1], outputs[arm][0][1]
        if len(a) != len(b) or any(not torch.equal(x, y) for x, y in zip(a[1:], b[1:])):
            raise AssertionError(f"initial native traces differ for {arm}")
    return {"initial_decoder_equal": True, "initial_gamma_q_hard_traces_equal": True,
            "initial_soft_p_rgb_equal": True,
            "warm_decoder_optimizer_shared_across_arms": True}


def _calibrate_lambda(seed, device, pool, warm_path, wrapped, encoder, patcher,
                      mean, std, clip, decoder, train_cache):
    warm = torch.load(warm_path, map_location=device, weights_only=True)
    _validate_warm_optimizer(warm)
    decoder.load_state_dict(warm["decoder_state_dict"], strict=True)
    named, groups = sw130.joint_parameters(wrapped, encoder)
    params = [parameter for group in groups for parameter in group["params"]]
    if len({id(p) for p in params}) != len(params):
        raise AssertionError("joint optimizer groups contain duplicate parameters")
    ratios, records = [], []
    wrapped.train(); encoder.eval(); decoder.eval()
    for batch_index in range(4):
        rows = pool[batch_index * BATCH:(batch_index + 1) * BATCH].tolist()
        images = sw130.read_batch(train_cache, rows, device)
        gamma, result, q, _labels, hard, _groups, target = _batch_forward(
            wrapped, encoder, patcher, mean, std, clip, images)
        old = _old_loss(result, q)
        _pred, rgb_per_image, _detail = _rgb_batch(q, hard, gamma, target, decoder)
        rgb = rgb_per_image.mean()
        old_grads = torch.autograd.grad(old, params, retain_graph=True, allow_unused=True)
        rgb_grads = torch.autograd.grad(rgb, params, retain_graph=False, allow_unused=True)
        old_norm, rgb_norm = _finite_norm(old_grads), _finite_norm(rgb_grads)
        if old_norm <= 0 or rgb_norm <= 0:
            raise AssertionError(f"seed{seed} lambda batch {batch_index}: empty joint gradient")
        if not bool(torch.isfinite(old)) or not bool(torch.isfinite(rgb)):
            raise FloatingPointError(f"seed{seed} lambda batch {batch_index}: nonfinite loss")
        families = _family_norms(named, rgb_grads)
        ratio = 0.25 * old_norm / rgb_norm
        ratios.append(ratio)
        records.append({"batch_index": batch_index, "old_joint_norm": old_norm,
                        "rgb_joint_norm": rgb_norm, "rgb_family_norms": families,
                        "lambda_ratio": ratio})
    return ratios, records


def _disposable_update(seed, arm, device, pool, fixed_lambda, warm_path):
    wrapped, encoder, patcher, mean, std, clip, decoder, _, _, _ = load_models(seed, device, arm)
    warm = torch.load(warm_path, map_location=device, weights_only=True)
    _validate_warm_optimizer(warm)
    decoder.load_state_dict(warm["decoder_state_dict"], strict=True)
    named, groups = sw130.joint_parameters(wrapped, encoder)
    params = [parameter for group in groups for parameter in group["params"]]
    named_ids = {id(parameter) for _, parameter in named}
    if len({id(parameter) for parameter in params}) != len(params) or {
            id(parameter) for parameter in params} != named_ids:
        raise AssertionError("joint optimizer parameters must equal the unique named gradient union")
    decoder_params = list(decoder.parameters())
    images = sw130.read_batch(np.load(sw130.TRAIN_RGB, mmap_mode="r"),
                              pool[:BATCH].tolist(), device)
    gamma, result, q, _labels, hard, _groups, target = _batch_forward(
        wrapped, encoder, patcher, mean, std, clip, images)
    old = _old_loss(result, q)
    assignment_live = arm != "phase_detached"
    _pred, rgb_per_image, _details = _rgb_batch(
        q, hard, gamma, target, decoder, assignment_live=assignment_live)
    rgb = rgb_per_image.mean()
    if not bool(torch.isfinite(old)) or not bool(torch.isfinite(rgb)):
        raise FloatingPointError(f"{arm} disposable update has a nonfinite loss")
    total = old + float(fixed_lambda) * rgb if assignment_live else old
    q_rgb_gradient, = torch.autograd.grad(rgb, q, retain_graph=True, allow_unused=True)
    q_rgb_norm = 0.0 if q_rgb_gradient is None else _finite_norm([q_rgb_gradient])
    if assignment_live and q_rgb_norm <= 0:
        raise AssertionError(f"{arm} full-RGB loss has no live assignment gradient to Q")
    if not assignment_live and q_rgb_norm != 0:
        raise AssertionError(f"{arm} detached assignment leaked RGB credit to Q")
    rgb_grads = torch.autograd.grad(rgb, [p for _, p in named], retain_graph=True,
                                    allow_unused=True)
    families = _family_norms(named, rgb_grads)
    required = ("encoder", "graph", "oscillator_drive", "kuramoto", "dendrite",
                "membrane", "a_d", "a_m", "b")
    if assignment_live and any(families.get(name, 0.0) <= 0 for name in required):
        raise AssertionError(f"{arm} relative-RGB credit missing: {families}")
    if not assignment_live and any(float(value) != 0.0 for value in families.values()):
        raise AssertionError(f"{arm} detached assignment leaked RGB credit to core/encoder: {families}")
    joint_grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
    decoder_grads = torch.autograd.grad(rgb, decoder_params, allow_unused=True)
    if _finite_norm(joint_grads) <= 0 or _finite_norm(decoder_grads) <= 0:
        raise AssertionError(f"{arm} disposable optimizer gradients are empty")
    joint_optimizer = torch.optim.Adam(groups)
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=DECODER_LR)
    decoder_optimizer.load_state_dict(warm["optimizer_state_dict"])
    for parameter, grad in zip(params, joint_grads):
        parameter.grad = None if grad is None else grad.detach().clone()
    for parameter, grad in zip(decoder_params, decoder_grads):
        parameter.grad = None if grad is None else grad.detach().clone()
    before_core = {name: p.detach().clone() for name, p in wrapped.core.named_parameters()}
    before_encoder = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    before_integration = {name: getattr(wrapped, name).detach().clone()
                          for name in ("a_d", "a_m", "b")}
    before_decoder = [p.detach().clone() for p in decoder_params]
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    torch.nn.utils.clip_grad_norm_(decoder_params, CLIP)
    joint_optimizer.step(); decoder_optimizer.step(); wrapped.project_integrations_()
    changed_core = [name for name, parameter in wrapped.core.named_parameters()
                    if not torch.equal(before_core[name], parameter)]
    changed_encoder = [name for name, parameter in encoder.named_parameters()
                       if not torch.equal(before_encoder[name], parameter)]
    changed_decoder = [str(i) for i, (before, after) in enumerate(
        zip(before_decoder, decoder_params)) if not torch.equal(before, after)]
    changed_integration = [name for name in before_integration
                           if not torch.equal(before_integration[name], getattr(wrapped, name))]
    native_families = {
        "graph": lambda n: n.startswith("graph_generator."),
        "oscillator_drive": lambda n: n.startswith(("gamma_channel_proj.", "gamma_phase_gain.")),
        "kuramoto": lambda n: n.startswith("kuramoto."),
        "dendrite": lambda n: n.startswith("dendric_layer."),
        "membrane": lambda n: n.startswith("membrane_layer."),
    }
    missing = [family for family, predicate in native_families.items()
               if not any(predicate(name) for name in changed_core)]
    if assignment_live and (missing or not changed_encoder or not changed_decoder
                            or "b" not in changed_integration):
        raise AssertionError(f"{arm} disposable update missed registered families: "
                             f"core={missing}, encoder={len(changed_encoder)}, "
                             f"decoder={len(changed_decoder)}, integration={changed_integration}")
    if not assignment_live and (not changed_core or not changed_decoder):
        raise AssertionError(f"{arm} old-objective/decoder disposable update did not change: "
                             f"core={len(changed_core)}, decoder={len(changed_decoder)}")
    if any(not torch.isfinite(parameter).all() for parameter in
           list(wrapped.parameters()) + list(encoder.parameters()) + decoder_params):
        raise FloatingPointError(f"{arm} disposable update produced nonfinite parameters")
    return {"arm": arm, "assignment_live": assignment_live,
            "rgb_to_q_gradient_norm": q_rgb_norm,
            "rgb_gradient_norms_by_family": families,
            "old_loss": float(old.detach()), "rgb_loss": float(rgb.detach()),
            "joint_preclip_norm": _finite_norm(joint_grads),
            "decoder_preclip_norm": _finite_norm(decoder_grads),
            "joint_and_decoder_changed": True,
            "changed_core_families": {
                family: any(predicate(name) for name in changed_core)
                for family, predicate in native_families.items()},
            "changed_encoder_parameter_count": len(changed_encoder),
            "changed_decoder_parameter_count": len(changed_decoder),
            "changed_integration_parameters": changed_integration,
            "integration_parameter_gradient_abs_by_component": {
                name: ([0.0] * int(parameter.numel()) if grad is None else
                       [float(value) for value in grad.detach().abs().reshape(-1).cpu().tolist()])
                for (name, parameter), grad in zip(named, rgb_grads)
                if name in ("core.a_d", "core.a_m", "core.b")},
            "rgb_gradient_unused_integration": {
                name: grad is None for (name, _parameter), grad in zip(named, rgb_grads)
                if name in ("core.a_d", "core.a_m", "core.b")},
            "optimizer_updates": 1,
            "throwaway_only": True}


def _validate_warm_optimizer(warm):
    optimizer = warm.get("optimizer_state_dict")
    if not isinstance(optimizer, dict):
        raise ValueError("warmup artifact lacks the paired decoder Adam state")
    states = optimizer.get("state", {})
    steps = []
    for state in states.values():
        step = state.get("step")
        steps.append(int(step.item() if torch.is_tensor(step) else step))
    if len(steps) != 6 or set(steps) != {WARMUP_STEPS}:
        raise ValueError("warm decoder Adam state must contain exactly six step-32 entries")


def preflight(seed, device, output):
    if seed not in SEEDS:
        raise ValueError("seed must be registered seed 0, 1, or 2")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0133 preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = sw130.validate_rgb_assets()
    gamma_cache, gamma_manifest = source97.validate_gamma_cache(
        source97.GAMMA_TRAIN, source97.GAMMA_TRAIN_MANIFEST)
    checkpoint, source_manifest_path, source_manifest, pool, ids, source_sha = source_contract(seed)
    fixed_lambda, seed0_record_sha = _seed0_lambda_record(output, seed, ids, assets)
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    rows = []
    failure = None
    warm_path = None
    warm_sha = None
    warm_losses = None
    lambda_ratios, lambda_records = [], []
    disposable = []
    try:
        wrapped, encoder, patcher, mean, std, clip, decoder, loaded_pool, loaded_ids, loaded_sha = \
            load_models(seed, device, "phase_live")
        if loaded_sha != source_sha or loaded_ids != ids or not np.array_equal(loaded_pool, pool):
            raise AssertionError("SW0133 model loader differs from registered source/order")
        first_rows = pool[:BATCH].tolist()
        first_images = sw130.read_batch(train_cache, first_rows, device)
        with torch.no_grad():
            cached = gamma_cache[torch.as_tensor(pool[:BATCH], dtype=torch.long)].to(device)
            live = sw130.encode(encoder, patcher, mean, std, clip, first_images)
            gamma_diff = float((live - cached).abs().max())
        if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
            raise AssertionError(f"live/cached source gamma mismatch {gamma_diff}")
        parity = sw130.verify_zero_initial_parity(seed, cached[:2], device)
        paired_arm_check = _paired_initial_arm_check(
            seed, device, first_images,
            (wrapped, encoder, patcher, mean, std, clip, decoder, pool, ids, source_sha))
        warm_path, warm_sha, warm_losses = _warm_decoder(
            seed, device, pool, ids, assets, gamma_cache, train_cache, wrapped,
            encoder, patcher, mean, std, clip, decoder, output)

        # The registered row shuffle is descriptive in SW0133; it is not an
        # initial-warmup eligibility threshold, unlike closed SW0132.
        scramble_by_arm = {}
        core, enc, pat, mu, sd, clip_value, dec = \
            wrapped, encoder, patcher, mean, std, clip, decoder
        dec.load_state_dict(torch.load(warm_path, map_location=device,
                                       weights_only=True)["decoder_state_dict"], strict=True)
        deltas = []
        core.eval(); enc.eval(); dec.eval()
        with torch.no_grad():
            for batch_index in range(4):
                start = batch_index * BATCH
                ix = pool[start:start + BATCH].tolist()
                batch = sw130.read_batch(train_cache, ix, device)
                gamma, _result, q, _labels, hard, _groups, target = _batch_forward(
                    core, enc, pat, mu, sd, clip_value, batch)
                for local in range(BATCH):
                    global_index = start + local
                    gen = torch.Generator(device="cpu").manual_seed(130 + global_index)
                    permutation = torch.randperm(256, generator=gen, device="cpu").to(device)
                    _, native_loss, _ = reconstruct_soft(
                        q[local], hard[local], gamma[local].transpose(0, 1), target[local], dec)
                    _, shuffled_loss, _ = reconstruct_soft(
                        q[local], hard[local][permutation], gamma[local].transpose(0, 1),
                        target[local], dec)
                    deltas.append(float(shuffled_loss - native_loss))
        scramble_by_arm["phase_live"] = {"per_image_excess": deltas,
                                         "mean_excess": float(np.mean(deltas)),
                                         "positive_count": int(sum(x > 0 for x in deltas))}

        if seed == 0:
            lambda_ratios, lambda_records = _calibrate_lambda(
                seed, device, pool, warm_path, wrapped, encoder, patcher, mean, std,
                clip, decoder, train_cache)
            fixed_lambda = float(statistics.median(lambda_ratios))
            seed0_record_sha = None
        elif fixed_lambda is None:
            raise RuntimeError("seed1/2 has no immutable seed0 lambda record")
        if not math.isfinite(float(fixed_lambda)) or fixed_lambda <= 0:
            raise AssertionError("shared lambda must be finite and positive")
        for arm in ARMS:
            disposable.append(_disposable_update(
                seed, arm, device, pool, fixed_lambda, warm_path))
        fp = implementation_fingerprint()
        status = "passed" if failure is None else "failed_scientific_guard"
        report = {
            "status": status, "scientific_failure": failure,
            "experiment": "SW0133_soft_partition_rgb", "seed": seed,
            "device": str(device), "created_unix": time.time(),
            "implementation_fingerprint": fp,
            "source_core_sha256": source_sha,
            "source_manifest_sha256": sha(source_manifest_path),
            "source_steps": int(source_manifest["steps"]),
            "training_ids": ids,
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "pool_indices_sha256": hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest(),
            "shuffle_seed": 117 + seed, "batch_size": BATCH,
            "train_time_steps": STEPS, "settle_steps": SETTLE,
            "asset_hashes": assets,
            "gamma_train_cache_sha256": gamma_manifest["gamma_sha256"],
            "gamma_train_manifest_sha256": sha(source97.GAMMA_TRAIN_MANIFEST),
            "live_source_gamma_max_abs_diff_first_batch": gamma_diff,
            "native_zero_initial_parity": parity,
            "paired_zero_initial_arm_check": paired_arm_check,
            "decoder_warmup_updates": WARMUP_STEPS,
            "decoder_warmup_loss_first_last": [warm_losses[0], warm_losses[-1]],
            "decoder_warmup_artifact": str(warm_path.resolve()),
            "decoder_warmup_artifact_sha256": warm_sha,
            "row_scramble_by_arm": scramble_by_arm,
            "row_scramble_count_per_arm": 64,
            "lambda_source_seed": 0, "lambda": float(fixed_lambda),
            "seed0_lambda_record_sha256": seed0_record_sha,
            "lambda_seed0_calibration": lambda_records if seed == 0 else "reused frozen seed0 preflight",
            "lambda_batch_ratios": lambda_ratios if seed == 0 else None,
            "disposable_updates": disposable,
            "optimizer_updates": 0,
            "ground_truth_used": False,
            "source_core_checkpoint_modified": False,
            "source_encoder_modified": False,
        }
        write_once(output, report)
        return report
    except Exception as exc:
        # Persist a failed attempt only when useful warmup/scramble evidence exists;
        # never overwrite it with a retry. Runtime failures remain explicit.
        if warm_path is not None and not output.exists():
            fail = {
                "status": "failed_preflight", "experiment": "SW0133_soft_partition_rgb",
                "seed": seed, "device": str(device), "created_unix": time.time(),
                "implementation_fingerprint": implementation_fingerprint(),
                "source_core_sha256": source_sha, "asset_hashes": assets,
                "decoder_warmup_artifact": str(warm_path.resolve()),
                "decoder_warmup_artifact_sha256": warm_sha,
                "row_scramble_by_arm": locals().get("scramble_by_arm", {}),
                "lambda": fixed_lambda, "disposable_updates": disposable,
                "failure_type": type(exc).__name__, "failure_message": str(exc),
                "ground_truth_used": False, "optimizer_updates": 0,
            }
            write_once(output, fail)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight",), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = preflight(args.seed, torch.device(args.device), args.output)
    print(json.dumps({"status": report["status"], "seed": args.seed,
                      "output": str(args.output), "lambda": report.get("lambda")},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
