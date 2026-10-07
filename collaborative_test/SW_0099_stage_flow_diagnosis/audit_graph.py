"""CPU equality audit for frozen graph-generator checkpoint parameters."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import hparams
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore


def state(condition, seed):
    path = ROOT / "trained_models" / {
        "SW0095": f"SW0095_full70k_aligned_loss/seed{seed}/core.pt",
        "SW0097": f"SW0097_graph_adaptation/seed{seed}_positive_frozen/core.pt",
        "SW0098": f"SW0098_long_window/seed{seed}/core.pt",
    }[condition]
    core = S2NetCore(hparams(), device="cpu")
    core.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    values = core.graph_generator.state_dict()
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode())
        digest.update(values[key].contiguous().numpy().tobytes())
    return {k: v.detach().clone() for k, v in values.items()}, digest.hexdigest()


def main():
    result = {"status": "complete", "device": "cpu", "per_seed": {}}
    for seed in range(3):
        states = {condition: state(condition, seed) for condition in ("SW0095", "SW0097", "SW0098")}
        hashes = {condition: value[1] for condition, value in states.items()}
        equal = all(torch.equal(states["SW0095"][0][k], states[c][0][k])
                    for c in ("SW0097", "SW0098")
                    for k in states["SW0095"][0])
        result["per_seed"][str(seed)] = {"hashes": hashes, "exactly_equal": equal}
    result["all_seeds_equal"] = all(v["exactly_equal"] for v in result["per_seed"].values())
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["all_seeds_equal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
