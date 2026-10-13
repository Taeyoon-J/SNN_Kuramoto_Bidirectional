"""No-GT SW0152 native32 zero-head parity and two-update feasibility preflight."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np
import torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training
from collaborative_test.SW_0152_source_preserving_partition_equivariance.partition_objective import (
    alignment_js, unflip_probabilities)
from collaborative_test.SW_0152_source_preserving_partition_equivariance.training import (
    configure_trainability, install_deep_encoder, logical_update, make_optimizer)

EXPERIMENT = "SW0152_source_preserving_partition_equivariance"
SEED, CORE_INIT_SEED, RESIDUAL_INIT_SEED = 1, 149001, 152001
TRAIN_COUNT, BATCH, MICRO = 2048, 16, 4
UPDATES, STEPS, SETTLE, TAIL = 2, 1024, 512, 64
OUTPUT_DEFAULT = HERE / "results_archive" / "preflight_seed1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def ids_sha(ids): return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def implementation_fingerprint():
    out = dict(sw135.implementation_fingerprint())
    paths = [HERE / n for n in ("protocol.json", "deep_encoder.py", "view_augmentation.py",
                                  "partition_objective.py", "training.py", "preflight.py")]
    paths += [ROOT / p for p in (
        "collaborative_test/SW_0150_spike_spatial_mixture/integration.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/spatial_diffusion.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/seed_selection.py",
        "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py")]
    missing = [p for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("missing SW0152 dependency: " + str(missing[0]))
    out.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    return out


def source_contract():
    ckpt, manifest_path, manifest, rows, ids, source_sha = sw130.source_contract(SEED)
    ids = [int(x) for x in ids]; rows = np.asarray(rows, dtype=np.int64)
    if len(ids) != 4096 or len(set(ids)) != 4096 or rows.shape != (4096,) or len(np.unique(rows)) != 4096:
        raise ValueError("requires exact ordered 4096-image source1 TRAIN contract")
    return {"checkpoint": Path(ckpt), "manifest_path": Path(manifest_path), "manifest": manifest,
            "rows": rows, "ids": ids, "source_sha256": source_sha}


def load_source(device):
    index = torch.device(device).index
    if index is None and torch.device(device).type == "cuda": index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED)
        if index is not None: torch.cuda.manual_seed(CORE_INIT_SEED)
        return sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)


def read_rows(cache, rows, start, count):
    rgb = np.asarray(cache[rows[start:start + count]]).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (count, 128, 128, 3):
        raise ValueError("source TRAIN RGB rows have invalid schema")
    return torch.from_numpy(rgb)


def padded_mass(p):
    m = p.detach().mean(1)
    if m.shape[-1] > 11: raise ValueError("source P exceeds eleven slots")
    return torch.nn.functional.pad(m, (0, 11 - m.shape[-1]))


def frozen_state_sha(foundation):
    core = {k: v for k, v in foundation.wrapped.core.state_dict().items()
            if not k.startswith("graph_generator.")}
    scalars = {"a_d": foundation.wrapped.a_d, "a_m": foundation.wrapped.a_m,
               "b": foundation.wrapped.b}
    return {"native_core": training.state_sha(core), "integration_scalars": training.state_sha(scalars)}


def source_masses(foundation, cache, rows, device, count):
    out = []
    with torch.no_grad():
        for start in range(0, count, MICRO):
            rgb = read_rows(cache, rows, start, MICRO).to(device)
            out.append(padded_mass(forward_assignment(foundation, rgb, live_tail_steps=TAIL)["p"]).cpu())
    return torch.cat(out)


def color_permutation_diagnostic(foundation, cache, rows, ids, device):
    base_losses, perm_losses, eligible, moved, aug_records = [], [], 0, 0, []
    with torch.no_grad():
        for start in range(0, 64, MICRO):
            rgb_cpu = read_rows(cache, rows, start, MICRO)
            ids_batch = [int(x) for x in ids[start:start + MICRO]]
            clean_cpu, aug_cpu, records = training.make_paired_views(
                rgb_cpu, ids_batch, source_seed=SEED)
            canonical, transformed = training.paired_assignments(
                foundation, clean_cpu.to(device), aug_cpu.to(device), live_tail_steps=TAIL)
            p_clean, p_aug = canonical["p"].detach(), transformed["p"].detach()
            p_aug_unflipped = unflip_probabilities(p_aug)
            shuffled, stats = training.color_bin_row_permutation(rgb_cpu, p_aug_unflipped)
            p_aug_corrupted = shuffled.reshape(MICRO, 32, 32, shuffled.shape[-1]).flip(2).reshape_as(p_aug)
            base_losses.append(float(alignment_js(p_clean, p_aug).cpu()))
            perm_losses.append(float(alignment_js(p_clean, p_aug_corrupted).cpu()))
            eligible += stats["eligible_patch_count"]
            moved += stats["changed_hard_assignment_count"]
            aug_records.extend(records)
    return {"training_ids": [int(x) for x in ids[:64]],
            "baseline_alignment_js_mean": float(np.mean(base_losses)),
            "within_color_permuted_alignment_js_mean": float(np.mean(perm_losses)),
            "eligible_patch_count": eligible, "changed_hard_assignment_count": moved,
            "moved_fraction": moved / eligible if eligible else 0.0,
            "augmentation_records": aug_records,
            "ground_truth_used": False, "diagnostic_only": True}


def run_preflight(*, device="cuda:0", output=OUTPUT_DEFAULT):
    output = Path(output)
    if output.exists(): raise FileExistsError("preserve existing SW0152 preflight output")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual-source preflight requires assigned CUDA")
    torch.cuda.set_device(dev)
    src = source_contract(); assets = sw130.validate_rgb_assets()
    saved_parity, parity_sha = sw150.load_and_validate_parity(
        sw150.PARITY_PATH, source_sha=src["source_sha256"],
        source_manifest_sha=sha(src["manifest_path"]), training_ids=src["ids"],
        asset_validation=assets)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered RGB TRAIN cache schema changed")

    foundation = load_source(dev)
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("rgb_asset_validation") != assets
            or list(map(int, foundation.image_ids)) != src["ids"]):
        raise ValueError("fresh source foundation differs from registered source/assets/order")
    rgb0 = read_rows(cache, src["rows"], 0, MICRO).to(dev)
    with torch.no_grad():
        ref = forward_assignment(foundation, rgb0, live_tail_steps=TAIL)
        ref_gamma = foundation.encode(rgb0).detach().cpu()
    source_encoder_sha = training.state_sha(foundation.encoder)
    source_core_sha = frozen_state_sha(foundation)
    masses = source_masses(foundation, cache, src["rows"], dev, BATCH * UPDATES)
    color_diagnostic = color_permutation_diagnostic(foundation, cache, src["rows"], src["ids"], dev)
    ref_cpu = {k: ref[k].detach().cpu() for k in ("q", "qcc", "p", "gamma")}
    for name in ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates"):
        ref_cpu[name] = ref["trace"][name].detach().cpu()
    del ref, ref_gamma

    arms = {}
    for arm in ("equivariance", "mass_control"):
        del foundation
        foundation = load_source(dev)
        if training.state_sha(foundation.encoder) != source_encoder_sha:
            raise ValueError("source encoder initialization differs between preflight arms")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(RESIDUAL_INIT_SEED)
            training.install_deep_encoder(foundation)
        foundation.encoder.to(dev)
        student = forward_assignment(foundation, rgb0, live_tail_steps=TAIL)
        student_values = {k: student[k].detach().cpu() for k in ("q", "qcc", "p", "gamma")}
        for name in ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates"):
            student_values[name] = student["trace"][name].detach().cpu()
        exact = all(torch.equal(ref_cpu[k], student_values[k]) for k in ref_cpu)
        del student
        if not exact: raise AssertionError("zero-head model failed exact source QCC/P/gamma parity")
        family_map = training.configure_trainability(foundation, arm)
        optimizer = training.make_optimizer(family_map)
        before = {name: training.parameter_hash(ps) for name, ps in family_map.items()}
        frozen_before = training.state_sha(foundation.encoder.base_encoder)
        history = []
        for update in range(UPDATES):
            ids = src["ids"][update * BATCH:(update + 1) * BATCH]
            def compute_micro(micro):
                start = update * BATCH + micro * MICRO
                clean_cpu = read_rows(cache, src["rows"], start, MICRO)
                clean, aug, _ = training.make_paired_views(clean_cpu, ids[micro * MICRO:(micro + 1) * MICRO], source_seed=SEED)
                clean, aug = clean.to(dev), aug.to(dev)
                a, b = training.paired_assignments(foundation, clean, aug, live_tail_steps=TAIL)
                return training.pair_loss(a, b, masses[start:start + MICRO].to(dev), arm=arm)
            history.append(training.logical_update(optimizer=optimizer, family_map=family_map,
                                                   compute_micro=compute_micro,
                                                   wrapped=foundation.wrapped))
        after = {name: training.parameter_hash(ps) for name, ps in family_map.items()}
        changed = {name: before[name] != after[name] for name in before}
        if not all(changed.values()): raise AssertionError("selected preflight parameters did not all change")
        if training.state_sha(foundation.encoder.base_encoder) != frozen_before:
            raise AssertionError("preflight changed frozen source encoder parameters")
        arms[arm] = {"status": "two_actual_updates_complete", "image_ids": src["ids"][:32],
                     "parameter_sha256_before": before, "parameter_sha256_after": after,
                     "changed_parameter_families": changed, "history": history,
                     "frozen_base_encoder_sha256_before_after": frozen_before,
                     "source_zero_head_parity_exact": exact}

    report = {"experiment": EXPERIMENT, "stage": "source1_two_update_preflight",
        "status": "source_preflight_complete", "seed": SEED, "training_admission": False,
        "ground_truth_used": False, "masks_read": False,
        "source": {"checkpoint_sha256": src["source_sha256"], "manifest_sha256": sha(src["manifest_path"]),
            "training_ids": src["ids"], "training_ids_sha256": ids_sha(src["ids"]),
            "source_rgb_asset_validation": assets, "source_encoder_sha256": source_encoder_sha,
            "source_native_core_sha256": source_core_sha, "saved_sw0135_parity_sha256": parity_sha},
        "saved_source_parity": {"path": str(sw150.PARITY_PATH), "sha256": parity_sha,
            "status": saved_parity["status"], "source1_actual_source_exact_T1024_settle512": True},
        "registered": {"logical_batch": BATCH, "microbatch": MICRO, "preflight_updates": UPDATES,
            "total_steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
            "optimizer_learning_rates": {"spatial_residual": 3e-6, "graph": 3e-5,
                                         "native_core": 3e-5, "integration_scalars": 1e-3}},
        "within_color_permutation_diagnostic": color_diagnostic, "arms": arms,
        "implementation_fingerprint": implementation_fingerprint()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    result = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": result["status"], "training_admission": False,
                      "arms": list(result["arms"]), "ground_truth_used": False}, allow_nan=False), flush=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
