"""SW0149 native-source preflight; writes evidence only and never opens GT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding import parity as sw135_parity
from collaborative_test.SW_0135_native32_spike_binding import resolution as sw135_resolution
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0149_spike_graph_modularity.modularity import (
    degree_corrected_modularity_loss, source_adjacency,
)
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

NAME = "SW0149_spike_graph_modularity"
SEED = 1
N, BATCH, MICRO = 1024, 16, 4
STEPS, SETTLE, TAIL = 1024, 512, 64
UPDATES = 2
PARITY_PATH = HERE.parent / "SW_0135_native32_spike_binding" / "results_archive" / "native32_zero_adapter_parity_seed1_b4.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint():
    result = dict(sw135.implementation_fingerprint())
    result.update(sw135_parity.implementation_fingerprint())
    paths = [HERE / "protocol.json", HERE / "modularity.py", HERE / "preflight.py",
             ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
             ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/protocol.json"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def implementation_sha():
    raw = json.dumps(fingerprint(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def ordered_train_contract(source):
    checkpoint, manifest_path, manifest, pool_rows, image_ids, source_sha = source
    rows = np.asarray(pool_rows, dtype=np.int64)
    ids = [int(value) for value in image_ids]
    if (len(rows) != 4096 or len(ids) != 4096 or len(np.unique(rows)) != 4096
            or len(set(ids)) != 4096):
        raise ValueError("seed1 source contract must contain the exact unique ordered4096 TRAIN rows")
    return {"checkpoint": Path(checkpoint), "manifest": Path(manifest_path),
            "source_manifest": manifest, "pool_rows": rows, "image_ids": ids,
            "source_core_sha256": source_sha}


def validate_existing_parity(src):
    """Require the already-executed exact SW0135 seed1 zero-adapter parity proof."""
    if not PARITY_PATH.is_file():
        raise FileNotFoundError(f"registered SW0135 seed1 parity report missing: {PARITY_PATH}")
    row = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    return validate_existing_parity_record(
        row, source_core_sha256=src["source_core_sha256"],
        source_manifest_sha256=sha(src["manifest"]),
    )


def validate_existing_parity_record(row, *, source_core_sha256, source_manifest_sha256):
    """Validate a decoded parity receipt against current source bindings."""
    if (row.get("status") != "native32_zero_adapter_parity_passed"
            or row.get("source_seed") != SEED
            or row.get("source_core_sha256") != source_core_sha256
            or row.get("source_manifest_sha256") != source_manifest_sha256
            or row.get("implementation_fingerprint") != sw135_parity.implementation_fingerprint()
            or row.get("ground_truth_used") is not False
            or row.get("optimizer_updates") != 0
            or row.get("training_admission") is not False):
        raise ValueError("existing SW0135 source parity proof is stale or invalid")
    checks = []
    for arm, rows in row.get("arms", {}).items():
        if arm not in {"phase", "constant"}:
            raise ValueError("unexpected SW0135 parity arm")
        for horizon in rows:
            checks.append((horizon.get("steps"), horizon.get("exact_parity")))
    if (len(checks) != 4 or any(steps not in {64, 1024} or exact is not True
                                for steps, exact in checks)):
        raise ValueError("SW0135 parity does not cover both exact native32 horizons/arms")
    return {"path": str(PARITY_PATH), "sha256": sha(PARITY_PATH),
            "implementation_fingerprint": row["implementation_fingerprint"],
            "exact_arm_horizons": checks}


def _groups(foundation):
    core = foundation.wrapped.core
    groups = {
        "encoder": list(foundation.encoder.parameters()),
        "graph": list(core.graph_generator.parameters()),
        "native_core": [p for name, p in core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m,
                                 foundation.wrapped.b],
    }
    flat = [p for family in groups.values() for p in family]
    if len({id(p) for p in flat}) != len(flat):
        raise ValueError("native32 parameter families overlap")
    return groups


def _state_digest(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _set_trainability(foundation, groups, arm):
    selected_names = {"encoder", "graph"}
    if arm == "joint":
        selected_names.update(("native_core", "integration_scalars"))
    selected = {id(p) for name in selected_names for p in groups[name]}
    for parameter in foundation.wrapped.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    for parameter in foundation.encoder.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    foundation.wrapped.train()
    foundation.encoder.train()
    return selected_names


def _teacher_modules(foundation):
    encoder = copy.deepcopy(foundation.encoder).eval()
    patcher = copy.deepcopy(foundation.patcher).eval()
    graph = copy.deepcopy(foundation.wrapped.core.graph_generator).eval()
    for module in (encoder, patcher, graph):
        module.requires_grad_(False)
    return encoder, patcher, graph


def _teacher_adjacency(foundation, teacher, rgb):
    encoder, patcher, graph = teacher
    with torch.no_grad():
        gamma = sw135.encode_native32(encoder, patcher, foundation.feature_mean,
                                      foundation.feature_std, foundation.feature_clip, rgb)
        adjacency = source_adjacency(graph(gamma))
    return adjacency


def _student_q(foundation, rgb):
    gamma = foundation.encode(rgb)
    trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS,
                         live_tail_steps=TAIL)
    q = sw145._actual_q(trace)
    return q, trace


def _source_q_parity(foundation, teacher, source_contract, rgb):
    """Compare fresh mapped adapter Q/A with native frozen source on exact B4."""
    params = list(foundation.wrapped.parameters()) + list(foundation.encoder.parameters())
    requires_grad = [parameter.requires_grad for parameter in params]
    for parameter in params:
        parameter.requires_grad_(False)
    wrapped_training = foundation.wrapped.training
    encoder_training = foundation.encoder.training
    foundation.wrapped.eval()
    foundation.encoder.eval()
    try:
        with torch.no_grad():
            gamma = foundation.encode(rgb)
            a_student = source_adjacency(foundation.wrapped.core.graph_generator(gamma))
            a_teacher = _teacher_adjacency(foundation, teacher, rgb)
            if not torch.equal(a_student, a_teacher):
                raise AssertionError("fresh student source graph differs from frozen source graph teacher")
            # late_rollout enables autograd for its live tail, so the mapped
            # foundation is explicitly frozen while checking the fresh-source Q.
            trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS,
                                 live_tail_steps=TAIL)
            q_student = sw145._actual_q(trace)
            core_state = torch.load(source_contract["checkpoint"], map_location=rgb.device,
                                    weights_only=True)
            native = sw135_resolution.build_core(core_state, device=rgb.device).eval()
            native._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            native_out = native(gamma, return_core_out=True, return_theta=True,
                                num_time_steps=STEPS)
            q_native = spike_synchrony_affinity(native_out[1], native.last_component_spikes,
                                                 settle=SETTLE, affinity_mode="spike")
            if not torch.equal(q_student, q_native):
                raise AssertionError("fresh source adapter actual-spike Q differs from native source core")
            result = {"images": int(rgb.shape[0]), "source_graph_exact": True,
                      "actual_spike_q_exact": True,
                      "source_graph_sha256": _tensor_sha(a_teacher),
                      "actual_spike_q_sha256": _tensor_sha(q_student),
                      "q_shape": list(q_student.shape)}
    finally:
        for parameter, flag in zip(params, requires_grad):
            parameter.requires_grad_(flag)
        foundation.wrapped.train(wrapped_training)
        foundation.encoder.train(encoder_training)
    del native, native_out, trace, q_student, q_native, a_student, a_teacher
    if rgb.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _tensor_sha(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_credit_screen(foundation, teacher, seed_contract, rgb_cache, device):
    params = list(foundation.wrapped.parameters()) + list(foundation.encoder.parameters())
    states = [parameter.requires_grad for parameter in params]
    for parameter in params:
        parameter.requires_grad_(False)
    foundation.wrapped.eval()
    foundation.encoder.eval()
    result = []
    try:
        with torch.no_grad():
            for start in range(0, 64, MICRO):
                rows = seed_contract["pool_rows"][start:start + MICRO]
                rgb_np = np.asarray(rgb_cache[rows]).copy()
                if rgb_np.dtype != np.uint8 or rgb_np.shape != (MICRO, 128, 128, 3):
                    raise ValueError("registered TRAIN cache returned unexpected native128 RGB")
                rgb = torch.from_numpy(rgb_np).to(device)
                adjacency = _teacher_adjacency(foundation, teacher, rgb)
                q, trace = _student_q(foundation, rgb)
                loss = degree_corrected_modularity_loss(adjacency, q)
                row = _credit_summary(adjacency, q)
                row["image_ids"] = seed_contract["image_ids"][start:start + MICRO]
                row["modularity_loss"] = float(loss.detach().cpu())
                result.append(row)
                del rgb, adjacency, q, trace, loss
    finally:
        for parameter, requires_grad in zip(params, states):
            parameter.requires_grad_(requires_grad)
    foundation.wrapped.train()
    foundation.encoder.train()
    return result


def _credit_summary(adjacency, q):
    z = adjacency.sum((-2, -1))
    degree = adjacency.sum(-1)
    modularity = adjacency - degree.unsqueeze(-1) * degree.unsqueeze(-2) / z[:, None, None]
    offdiag = ~torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device)
    values = modularity[:, offdiag]
    affinities = q[:, offdiag]
    positive = values > 0
    negative = values < 0
    return {
        "positive_m_entries": int(positive.sum().item()),
        "negative_m_entries": int(negative.sum().item()),
        "positive_m_loss_contribution": float((-(values[positive] * affinities[positive]).sum()).item()),
        "negative_m_loss_contribution": float((-(values[negative] * affinities[negative]).sum()).item()),
    }


def _family_norms(groups):
    out = {}
    for name, parameters in groups.items():
        sq = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
        out[name] = math.sqrt(float(torch.stack(sq).sum().cpu())) if sq else 0.0
        if not math.isfinite(out[name]):
            raise FloatingPointError(f"nonfinite gradient norm for {name}")
    return out


def _arm_trial(arm, *, dev, seed_contract, rgb_cache, credit=None, expected_init=None):
    # Reconstruct each arm from the exact same random initializer seed, never
    # from any disposable or prior trained model state.
    torch.manual_seed(149001)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(149001)
    foundation = sw135.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
    if (foundation.provenance.get("source_core_sha256") != seed_contract["source_core_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha(seed_contract["manifest"])
            or foundation.provenance.get("rgb_asset_validation") != sw130.validate_rgb_assets()
            or list(map(int, foundation.image_ids)) != seed_contract["image_ids"]
            or not np.array_equal(foundation.pool_indices, seed_contract["pool_rows"])):
        raise ValueError("native32 foundation source/order/assets differ from exact seed1 contract")
    groups = _groups(foundation)
    init_hashes = {name: _state_digest(ps) for name, ps in groups.items()}
    if expected_init is not None and init_hashes != expected_init:
        raise AssertionError("fresh arms did not reconstruct the same source97 initialization")
    teacher = _teacher_modules(foundation)
    rgb_parity_np = np.asarray(rgb_cache[seed_contract["pool_rows"][:MICRO]]).copy()
    rgb_parity = torch.from_numpy(rgb_parity_np).to(dev)
    source_parity = _source_q_parity(foundation, teacher, seed_contract, rgb_parity)
    del rgb_parity, rgb_parity_np
    trainable = _set_trainability(foundation, groups, arm)
    rates = {"encoder": 3e-6, "graph": 3e-5,
             "native_core": 3e-5, "integration_scalars": 1e-3}
    optim_groups = [{"params": groups[name], "lr": rates[name], "name": name}
                    for name in groups if name in trainable]
    params = [p for group in optim_groups for p in group["params"]]
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("preflight optimizer parameters are empty or duplicated")
    optimizer = torch.optim.Adam(optim_groups)
    arm_credit = _source_credit_screen(foundation, teacher, seed_contract, rgb_cache, dev) if credit is not None else None
    before = {name: _state_digest(ps) for name, ps in groups.items()}
    frozen_before = _state_digest(groups["native_core"] + groups["integration_scalars"])
    update_rows = []
    for update in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for micro in range(BATCH // MICRO):
            offset = update * BATCH + micro * MICRO
            rows = seed_contract["pool_rows"][offset:offset + MICRO]
            rgb_np = np.asarray(rgb_cache[rows]).copy()
            if rgb_np.dtype != np.uint8 or rgb_np.shape != (MICRO, 128, 128, 3):
                raise ValueError("registered TRAIN cache returned unexpected native128 RGB")
            rgb = torch.from_numpy(rgb_np).to(dev)
            a = _teacher_adjacency(foundation, teacher, rgb)
            q, trace = _student_q(foundation, rgb)
            loss = degree_corrected_modularity_loss(a, q)
            del a, q, trace
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite modularity loss in disposable preflight update")
            (loss / (BATCH // MICRO)).backward()
            micro_losses.append(float(loss.detach().cpu()))
            del loss, rgb
        norms = _family_norms(groups)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if arm == "joint":
            with torch.no_grad():
                foundation.wrapped.a_d.clamp_(0, 1)
                foundation.wrapped.a_m.clamp_(0, 1)
        for name, family in groups.items():
            if not all(torch.isfinite(p).all() for p in family):
                raise FloatingPointError(f"nonfinite updated {name} parameter")
        after_step = {name: _state_digest(ps) for name, ps in groups.items()}
        update_rows.append({"update": update + 1, "mean_microbatch_loss": float(np.mean(micro_losses)),
                            "gradient_norms": norms,
                            "changed_groups": [name for name in groups if after_step[name] != before[name]],
                            "parameter_sha256_after": after_step})
        before = after_step
    frozen_after = _state_digest(groups["native_core"] + groups["integration_scalars"])
    if arm == "encoder_graph" and frozen_before != frozen_after:
        raise AssertionError("encoder_graph preflight changed frozen native core/integration scalars")
    final_hashes = {name: _state_digest(ps) for name, ps in groups.items()}
    del optimizer, teacher, foundation
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if arm_credit is not None:
        credit.extend(arm_credit)
    return {"arm": arm, "trainable_groups": sorted(trainable),
            "fresh_source_parity": source_parity,
            "initial_parameter_sha256": init_hashes,
            "updates": update_rows, "final_parameter_sha256": final_hashes,
            "frozen_native_sha256_before": frozen_before,
            "frozen_native_sha256_after": frozen_after}


def run_preflight(*, device="cuda:0", output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0149 preflight: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0149 actual-source preflight requires an exclusive CUDA device")
    torch.cuda.set_device(dev)
    torch.cuda.init()
    src = ordered_train_contract(sw130.source_contract(SEED))
    assets = sw130.validate_rgb_assets()
    parity = validate_existing_parity(src)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered clean TRAIN RGB cache shape/dtype mismatch")
    credit_rows = []
    arms = []
    init_hashes = None
    for arm in ("encoder_graph", "joint"):
        result = _arm_trial(arm, dev=dev, seed_contract=src, rgb_cache=cache,
                            credit=credit_rows if arm == "encoder_graph" else None,
                            expected_init=init_hashes)
        if init_hashes is None:
            init_hashes = result["initial_parameter_sha256"]
        arms.append(result)
    if len(credit_rows) != 16:
        raise AssertionError("credit screen must cover the first64 TRAIN images in B4 groups")
    positive = sum(row["positive_m_entries"] for row in credit_rows)
    negative = sum(row["negative_m_entries"] for row in credit_rows)
    if positive <= 0 or negative <= 0:
        raise AssertionError("first64 source modularity lacks both positive and negative off-diagonal entries")
    record = {
        "status": "preflight_complete", "experiment": NAME, "seed": SEED,
        "disposable_optimizer_updates_per_arm": UPDATES,
        "training_admission": False,
        "ground_truth_used": False,
        "source": {"core_sha256": src["source_core_sha256"],
                   "manifest_sha256": sha(src["manifest"]),
                   "training_ids": src["image_ids"],
                   "training_ids_sha256": hashlib.sha256(np.asarray(src["image_ids"], dtype="<i8").tobytes()).hexdigest(),
                   "rgb_asset_validation": assets},
        "existing_native32_parity": parity,
        "first64_modularity_credit": {"images": 64, "microbatch": MICRO,
            "positive_M_offdiagonal_entries": positive,
            "negative_M_offdiagonal_entries": negative,
            "per_b4_batch": credit_rows},
        "arms": arms,
        "implementation_fingerprint": fingerprint(),
        "implementation_sha256": implementation_sha(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"preserve pre-existing preflight output: {output}")
    else:
        temporary.unlink()
        if os.name == "posix":
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    row = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": row["status"], "experiment": NAME,
                      "training_admission": False, "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
