"""Paired first-four-batch SW0097/SW0102 GT-free diagnostic; no optimizer."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"), str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import GAMMA, hparams, aligned_affinity
from SW_0102_cannot_link_draft.calibration_preflight import actual_101_negative_masks, make_criterion, expected_training_ids
from SW_0102_cannot_link_draft.forest_loss import cannot_link_hinge, same_component_pairs
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv

SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
CANDIDATE = ROOT / "trained_models/SW0102_cannot_link_draft"
OUT = ROOT / "collaborative_test/SW_0102_cannot_link_draft/results_archive/paired_train_audit.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    gamma_path = ROOT / GAMMA.relative_to(ROOT)
    gamma_cache = torch.load(gamma_path, map_location="cpu", weights_only=True, mmap=True)
    result = {"status": "running", "purpose": "paired first four registered train batches, no GT/no optimizer",
              "device": str(device), "gamma_cache_sha256": sha(gamma_path), "started_unix": time.time(),
              "contract": {"ids_per_seed": 4096, "batches": 4, "batch": 16,
                           "train_steps": 64, "settle": 32, "shuffle": [117, 118, 119]},
              "seeds": {}}
    for seed in range(3):
        ids = expected_training_ids(seed)
        control_dir = CONTROL / f"seed{seed}_positive_frozen"
        candidate_dir = CANDIDATE / f"seed{seed}"
        control_manifest = json.loads((control_dir / "manifest.json").read_text())
        candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text())
        if control_manifest["training_ids"] != ids or candidate_manifest["training_ids"] != ids:
            raise AssertionError(f"seed{seed}: registered train ID order mismatch")
        if control_manifest["source_sha256"] != candidate_manifest["source_sha256"]:
            raise AssertionError(f"seed{seed}: candidate/control sources differ")
        source_sha = sha(SOURCE / f"seed{seed}/core.pt")
        if source_sha != candidate_manifest["source_sha256"]:
            raise AssertionError(f"seed{seed}: source SHA mismatch")
        outputs = {}
        graph_states = {}
        for label, checkpoint in (("control", control_dir / "core.pt"),
                                  ("candidate", candidate_dir / "core.pt")):
            core = S2NetCore(hparams("raw"), device=device).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            core.load_state_dict(state, strict=True)
            graph_states[label] = {k: v.detach().cpu().clone() for k, v in core.graph_generator.state_dict().items()}
            core.graph_generator.requires_grad_(False)
            core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            core.train()
            rows = []
            criterion = make_criterion()
            generator = torch.Generator().manual_seed(117 + seed)
            train_indices = torch.randperm(70000, generator=generator)[:4096]
            for batch_index in range(4):
                ix = train_indices[batch_index * 16:(batch_index + 1) * 16]
                gamma = gamma_cache[ix].to(device)
                adjacency, projected = [], []
                h1 = core.graph_generator.register_forward_hook(lambda _m, _i, o: adjacency.append(o.detach()))
                h2 = core.graph_generator.projection.register_forward_hook(lambda _m, _i, o: projected.append(o.detach()))
                try:
                    with torch.no_grad():
                        _, spikes, _, plv, theta = _forward_with_plv(core, gamma, criterion, 32, "phase", "mean")
                finally:
                    h1.remove(); h2.remove()
                if len(adjacency) != 1 or len(projected) != 1:
                    raise AssertionError("expected one graph and projection call")
                masks, mask_counts = actual_101_negative_masks(adjacency[0], projected[0])
                q = aligned_affinity(core, settle=32).detach()
                cut, count_stats = cannot_link_hinge(q, masks)
                sym = torch.maximum(q, q.transpose(1, 2))
                per_bin = []
                for neg in masks:
                    cc = same_component_pairs(sym, neg, threshold=.50)
                    off = ~torch.eye(256, dtype=torch.bool, device=device).unsqueeze(0)
                    decisions = neg & off
                    direct = decisions & (sym >= .50)
                    indirect = decisions & cc & (sym <= .40)
                    per_bin.append({"selected_negative_decisions": int(decisions.sum()),
                                    "same_cc_q_ge_050": int((decisions & cc).sum()),
                                    "direct_q_ge_050": int(direct.sum()),
                                    "indirect_same_cc_q_le_040": int(indirect.sum())})
                primary, _ = criterion(plv=plv, theta=theta)
                positive_spike = criterion(plv=q)[0]
                rows.append({"batch": batch_index,
                    "training_ids": ids[batch_index * 16:(batch_index + 1) * 16],
                    "mask_counts": mask_counts, "negative_loss_counts": count_stats,
                    "cut_loss": float(cut), "primary_loss": float(primary),
                    "unweighted_positive_spike_loss": float(positive_spike),
                    "weighted_positive_spike_loss": float(5.0 * positive_spike),
                    "negative_conflicts_by_bin": per_bin,
                    "positive_product_density_ge_050": float((sym >= .50).float().mean()),
                    "positive_product_mean": float(sym.mean()),
                    "positive_product_variance": float(sym.var(unbiased=False)),
                    "component_spike_occupancy": float((core.last_component_spikes > 0).float().mean()),
                    "component_spike_variance": float(core.last_component_spikes.var(unbiased=False)),
                    "theta_variance": float(theta.var(unbiased=False)),
                    "finite": bool(torch.isfinite(q).all() and math.isfinite(float(cut)) and math.isfinite(float(primary)) and math.isfinite(float(positive_spike)))})
                del gamma, spikes, plv, theta, q, cut
            outputs[label] = rows
            del core, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
        graph_equal = all(torch.equal(graph_states["control"][k], graph_states["candidate"][k])
                          for k in graph_states["control"])
        if not graph_equal:
            raise AssertionError(f"seed{seed}: graph tensors differ between SW0097 and SW0102")
        for b in range(4):
            left, right = outputs["control"][b], outputs["candidate"][b]
            if left["training_ids"] != right["training_ids"] or left["mask_counts"] != right["mask_counts"]:
                raise AssertionError(f"seed{seed} batch{b}: exact frozen masks/ID mismatch")
        result["seeds"][str(seed)] = {"source_sha256": source_sha,
            "control_checkpoint_sha256": sha(control_dir / "core.pt"),
            "candidate_checkpoint_sha256": sha(candidate_dir / "core.pt"),
            "matched_ids_and_order": True, "graph_state_bitwise_equal": graph_equal,
            "control": outputs["control"], "candidate": outputs["candidate"]}
    result["status"] = "complete"
    result["finished_unix"] = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"preserve existing audit: {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "seeds": list(result["seeds"]),
                      "output": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
