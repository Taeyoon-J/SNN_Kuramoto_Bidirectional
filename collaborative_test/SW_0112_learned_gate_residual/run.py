"""Matched SW0112 nine-parameter delayed raw-gate residual pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import ASSETS
from collaborative_test.SW_0110_xy_graph_route import run as common
from gate_residual import actual_gate_binding, attach_gate_residual

SEEDS = common.SEEDS
ARMS = ("control", "gate_candidate")
BATCH = common.BATCH
UPDATES = common.UPDATES
LR = common.LR
OUT = ROOT / "trained_models/SW0112_learned_gate_residual"
ARCHIVE = HERE / "results_archive"
RUNNER = HERE / "run.py"
PROTOCOL = HERE / "protocol.json"
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_source(seed, arm, device):
    core, checkpoint, manifest_path, manifest = common.load_source(seed, "control", device)
    if arm == "gate_candidate":
        attach_gate_residual(core)
    elif arm != "control":
        raise ValueError(f"unregistered arm {arm}")
    core.graph_generator.requires_grad_(False)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core, checkpoint, manifest_path, manifest


def loss_parts(core, gamma, lossfn):
    if hasattr(core, "gate_residual"):
        with actual_gate_binding(core):
            return common.loss_parts(core, gamma, lossfn)
    return common.loss_parts(core, gamma, lossfn)


def preflight(seed, arm, device="cuda"):
    implementation = common.implementation_fingerprint(
        (RUNNER, HERE / "gate_residual.py", PROTOCOL))
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    core, source, source_manifest, source_record = load_source(seed, arm, device)
    ids, rows = common.train_indices(seed)
    gamma, gamma_manifest = common.validate_gamma_cache(common.GAMMA_TRAIN,
                                                         common.GAMMA_TRAIN_MANIFEST)
    control, _, _, _ = load_source(seed, "control", device)
    candidate, _, _, _ = load_source(seed, "gate_candidate", device)
    probe = gamma[torch.as_tensor(rows[:BATCH])].to(device)
    control.eval(); candidate.eval()
    with torch.no_grad():
        control_drive = control.gamma_to_drive(probe, control.gamma_channel_proj,
                                                control.gamma_phase_gain)
        candidate_drive = candidate.gamma_to_drive(probe, candidate.gamma_channel_proj,
                                                    candidate.gamma_phase_gain)
        if not torch.equal(control_drive, candidate_drive):
            raise AssertionError("SW0112 changed native gamma-to-drive")
        if not torch.equal(control.graph_generator(probe), candidate.graph_generator(probe)):
            raise AssertionError("SW0112 changed frozen legacy graph output")
        phi_test = torch.linspace(-math.pi, math.pi, steps=64, device=device).reshape(2, 4, 8).transpose(1, 2)
        base_test = 0.5 * (1.0 + torch.sin(phi_test.mean(dim=-1)))
        corrected_test = candidate.gate_residual(phi_test, base_test)
        if (not torch.isfinite(corrected_test).all()
                or torch.any(corrected_test < base_test.square() - 1e-7)
                or torch.any(corrected_test > 2 * base_test - base_test.square() + 1e-7)):
            raise AssertionError("SW0112 residual violates finite bounded gate contract")
        control_gates, candidate_gates = [], []
        control_hook = control.membrane_layer.register_forward_pre_hook(
            lambda module, inputs: control_gates.append(inputs[1].detach().clone()))
        try:
            expected = control(probe, return_core_out=True, return_theta=True,
                               num_time_steps=common.TRAIN_STEPS)
        finally:
            control_hook.remove()
        candidate_hook = candidate.membrane_layer.register_forward_pre_hook(
            lambda module, inputs: candidate_gates.append(inputs[1].detach().clone()))
        try:
            with actual_gate_binding(candidate):
                actual = candidate(probe, return_core_out=True, return_theta=True,
                                   num_time_steps=common.TRAIN_STEPS)
        finally:
            candidate_hook.remove()
        if (len(control_gates) != common.TRAIN_STEPS
                or len(candidate_gates) != common.TRAIN_STEPS
                or any(g.shape != (BATCH * 4, 256) for g in candidate_gates)
                or any(not torch.equal(a, b) for a, b in zip(control_gates, candidate_gates))):
            raise AssertionError("zero residual changed scalar gate or folded [B*4,N] ordering")
        for label, left, right in zip(("spikes", "membrane", "phase"),
                                      expected[1:], actual[1:]):
            if not torch.equal(left, right):
                raise AssertionError(f"zero-initialized residual changed rollout {label}")
        if not torch.equal(control.last_component_spikes, candidate.last_component_spikes):
            raise AssertionError("zero-initialized residual changed actual component spikes")

    roundtrip = io.BytesIO()
    torch.save(candidate.state_dict(), roundtrip)
    roundtrip.seek(0)
    restored, _, _, _ = load_source(seed, "gate_candidate", device)
    restored.load_state_dict(torch.load(roundtrip, map_location=device, weights_only=True), strict=True)
    with torch.no_grad(), actual_gate_binding(restored):
        replay = restored(probe, return_core_out=True, return_theta=True,
                          num_time_steps=common.TRAIN_STEPS)
    if not all(torch.equal(a, b) for a, b in zip(actual[1:], replay[1:])):
        raise AssertionError("strict SW0112 adapter checkpoint roundtrip changed rollout")
    initial = {k: v.detach().clone() for k, v in core.state_dict().items()}
    lossfn = common.criterion()
    batches = []
    core.train(); core.graph_generator.eval()
    for batch_index in range(4):
        batch_rows = rows[batch_index * BATCH:(batch_index + 1) * BATCH]
        x = gamma[torch.as_tensor(batch_rows)].to(device)
        total, primary, positive = loss_parts(core, x, lossfn)
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0112 loss in seed{seed} batch{batch_index}")
        core.zero_grad(set_to_none=True)
        total.backward()
        params = [p for p in core.parameters() if p.requires_grad]
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("SW0112 preflight produced empty or nonfinite gradients")
        total_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        residual_norm = None
        if arm == "gate_candidate":
            gate_params = (core.gate_residual.w, core.gate_residual.b)
            gate_grads = [p.grad for p in gate_params]
            if any(g is None or not torch.isfinite(g).all() for g in gate_grads):
                raise AssertionError("SW0112 residual gradients missing or nonfinite")
            residual_norm = math.sqrt(sum(float(g.detach().double().square().sum())
                                          for g in gate_grads))
            if not math.isfinite(residual_norm) or residual_norm <= 0:
                raise AssertionError(f"SW0112 residual is inert at batch{batch_index}")
        batches.append({"batch": batch_index, "global_ids": ids[batch_index*BATCH:(batch_index+1)*BATCH].tolist(),
                        "total_loss": float(total.detach()), "phase_primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "core_gradient_norm": total_norm,
                        "gate_residual_gradient_norm": residual_norm})
    if any(not torch.equal(v, core.state_dict()[k]) for k, v in initial.items()):
        raise AssertionError("read-only SW0112 preflight modified source parameters")
    update_core, _, _, _ = load_source(seed, arm, device)
    update_params = [p for p in update_core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(update_params, lr=LR)
    before = {k: v.detach().clone() for k, v in update_core.state_dict().items()}
    total, _, _ = loss_parts(update_core, gamma[torch.as_tensor(rows[:BATCH])].to(device), lossfn)
    opt.zero_grad(set_to_none=True); total.backward()
    norm = torch.nn.utils.clip_grad_norm_(update_params, 1.0)
    if not torch.isfinite(total) or not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError("invalid throwaway SW0112 Adam update")
    opt.step()
    after = update_core.state_dict()
    if not any(not torch.equal(v, after[k]) for k, v in before.items()
               if not k.startswith("graph_generator.")):
        raise AssertionError("throwaway Adam preflight changed no eligible core/gate parameters")
    if any(not torch.equal(v, after[k]) for k, v in before.items()
           if k.startswith("graph_generator.")):
        raise AssertionError("throwaway Adam update changed frozen legacy graph")
    if arm == "gate_candidate" and all(torch.equal(before[k], after[k])
            for k in ("gate_residual.w", "gate_residual.b")):
        raise AssertionError("throwaway update did not change gate residual parameters")
    return {"status": "passed", "experiment": "SW0112", "seed": seed, "arm": arm,
            "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
            "implementation_sha256": implementation,
            "source_record": str(common.SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"),
            "source_evidence_manifest_sha256": sha(common.SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"),
            "encoder_sha256": sha(ASSETS / "input_encoder/input_layer_encoder.pt"),
            "preprocessing_sha256": sha(ASSETS / "feature_preprocessing.pt"),
            "gamma_train_sha256": sha(common.GAMMA_TRAIN),
            "gamma_train_manifest_sha256": sha(common.GAMMA_TRAIN_MANIFEST),
            "training_ids": ids.tolist(),
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "matched_shuffle_seed": 117 + seed, "batch_size": BATCH,
            "time_steps": common.TRAIN_STEPS, "settle": common.TRAIN_SETTLE,
            "phase_delay_steps": 2, "ground_truth_used": False, "batches": batches,
            "zero_residual_legacy_equivalence": True, "graph_frozen": True,
            "bounded_gate_formula_finite": True,
            "membrane_gate_fold_order_exact": True,
            "strict_checkpoint_roundtrip_bitwise": True, "throwaway_b16_adam_update": True,
            "throwaway_update_gradient_norm_preclip": float(norm)}


def train(seed, arm, output, device="cuda", steps=UPDATES):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"preserve existing attempt without overwrite: {out}")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    pf = json.loads(pf_path.read_text())
    if pf.get("status") != "passed" or pf.get("seed") != seed or pf.get("arm") != arm:
        raise AssertionError("matching successful SW0112 preflight required")
    if pf.get("implementation_sha256") != common.implementation_fingerprint(
            (RUNNER, HERE / "gate_residual.py", PROTOCOL)):
        raise AssertionError("implementation changed since preflight; revalidate before training")
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    core, source, source_manifest, source_record = load_source(seed, arm, device)
    ids, rows = common.train_indices(seed)
    gamma, gamma_manifest = common.validate_gamma_cache(common.GAMMA_TRAIN,
                                                         common.GAMMA_TRAIN_MANIFEST)
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    lossfn = common.criterion()
    graph_before = {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "training", "experiment": "SW0112", "seed": seed, "arm": arm,
                "source_core": str(source), "source_core_sha256": sha(source),
                "source_manifest_sha256": sha(source_manifest),
                "source_evidence_manifest_sha256": sha(common.SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"),
                "training_ids": ids.tolist(),
                "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "encoder_sha256": sha(ASSETS / "input_encoder/input_layer_encoder.pt"),
                "preprocessing_sha256": sha(ASSETS / "feature_preprocessing.pt"),
                "gamma_train_sha256": sha(common.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(common.GAMMA_TRAIN_MANIFEST),
                "matched_shuffle_seed": 117 + seed, "updates": steps, "batch_size": BATCH,
                "lr": LR, "clip_norm": 1.0, "time_steps": common.TRAIN_STEPS,
                "settle": common.TRAIN_SETTLE, "phase_delay_steps": 2,
                "loss": "phase_primary + 5 * positive_actual_spike_product",
                "gate_formula": "g0 + g0*(1-g0)*tanh([sin(phi),cos(phi)] dot w + b)",
                "gate_adapter_config": {"class": "SharedGateResidual", "components": 4,
                    "w_shape": [8], "b_shape": [], "parameters": 9,
                    "zero_init": arm == "gate_candidate", "phase_delay_steps": 2},
                "ground_truth_used_for_training": False}
    write(out / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.eval()
    for step in range(steps):
        index = rows[step * BATCH:(step + 1) * BATCH]
        total, primary, positive = loss_parts(core, gamma[torch.as_tensor(index)].to(device), lossfn)
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0112 loss at update {step + 1}")
        opt.zero_grad(set_to_none=True); total.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"invalid SW0112 gradient at update {step + 1}")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise FloatingPointError("invalid SW0112 preclip norm")
        opt.step()
        if any(not torch.equal(v, core.graph_generator.state_dict()[k])
               for k, v in graph_before.items()):
            raise AssertionError("frozen SW0112 graph changed during training")
        history.append({"update": step + 1, "total": float(total.detach()),
                        "phase_primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "preclip_norm": float(norm)})
        if (step + 1) % 32 == 0:
            write(out / "history.json", history)
            torch.save(core.state_dict(), out / "core.pt.tmp")
            (out / "core.pt.tmp").replace(out / "core.pt")
    write(out / "history.json", history)
    torch.save(core.state_dict(), out / "core.pt.tmp")
    (out / "core.pt.tmp").replace(out / "core.pt")
    manifest.update({"status": "training_complete", "core_sha256": sha(out / "core.pt"),
                     "ground_truth_used_for_training": False})
    write(out / "manifest.json", manifest)
    (out / "TRAINING_COMPLETED").write_text("SW0112 256 matched updates complete\n")


def _eval_core(device, checkpoint, steps, gate):
    core = common.make_core(device, steps=steps)
    if gate:
        attach_gate_residual(core)
    core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    core.eval()
    if hasattr(core, "gate_residual"):
        original_forward = core.forward
        def forward_with_gate_adapter(*args, **kwargs):
            with actual_gate_binding(core):
                return original_forward(*args, **kwargs)
        core.forward = forward_with_gate_adapter
    return core


def evaluate(seed, arm, checkpoint, output, device="cuda"):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    import evaluate_fixed_split as fixed
    old_factory, old_argv = fixed._core, sys.argv[:]
    fixed._core = lambda dev, ckpt, steps, *args, **kwargs: _eval_core(
        dev, ckpt, steps, gate=(arm == "gate_candidate"))
    evaluator_path = ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"
    try:
        spec = importlib.util.spec_from_file_location("sw0112_shared_evaluate", evaluator_path)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        common.validate_gamma_cache(common.GAMMA_VAL, common.GAMMA_VAL_MANIFEST, validation=True)
        sys.argv = [str(evaluator_path), "--checkpoint", str(checkpoint),
            "--gamma-path", str(common.GAMMA_VAL), "--gamma-global-start", "1320",
            "--gamma-manifest", str(common.GAMMA_VAL_MANIFEST),
            "--dataset-path", str(common.DATASET), "--output-path", str(output),
            "--start", "1320", "--count", "320", "--steps", "1024", "--settle", "512",
            "--thresholds", ".50", "--min-group-size", "2", "--background", "largest_component",
            "--dendritic-projection", "shared", "--graph-spatial-decay", ".35",
            "--geodesic-steps", "3", "--geodesic-radius", "1.5", "--geodesic-contrast", "2",
            "--geodesic-temperature", ".5", "--geodesic-cap", "16", "--kuramoto-backend", "factorized",
            "--gate-mode", "raw", "--batch-size", "8", "--membrane-vth", ".06", "--device", str(device)]
        evaluator.main()
    finally:
        fixed._core = old_factory
        sys.argv = old_argv
    report = json.loads(Path(output).read_text())
    if (report.get("ids") != [1320, 1639] or report.get("images") != 320
            or report.get("ground_truth_used_for_prediction") is not False):
        raise AssertionError("SW0112 evaluator output does not match registered validation contract")
    write(Path(output).with_name("evaluation_manifest.json"), {
        "status": "complete", "experiment": "SW0112", "seed": seed, "arm": arm,
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha(checkpoint),
        "evaluation_sha256": sha(output), "gamma_sha256": sha(common.GAMMA_VAL),
        "gamma_manifest_sha256": sha(common.GAMMA_VAL_MANIFEST), "ids": [1320, 1639],
        "images": 320, "ground_truth_used_for_prediction": False,
        "batch_size": 8, "steps": 1024, "settle": 512,
        "membrane_vth": 0.06, "synchrony_threshold": 0.5})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sp.add_argument("--arm", choices=ARMS, required=True)
        sp.add_argument("--device", default="cuda")
        if name in ("preflight", "train"):
            sp.add_argument("--output", type=Path, required=True)
        if name == "train":
            sp.add_argument("--steps", type=int, default=UPDATES)
        if name == "evaluate":
            sp.add_argument("--checkpoint", type=Path, required=True)
            sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        if args.output.exists():
            raise FileExistsError(args.output)
        write(args.output, preflight(args.seed, args.arm, args.device))
    elif args.command == "train":
        train(args.seed, args.arm, args.output, args.device, args.steps)
    else:
        evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
    print(json.dumps({"status": "complete", "stage": args.command,
                      "seed": args.seed, "arm": args.arm}, flush=True))


if __name__ == "__main__":
    main()
