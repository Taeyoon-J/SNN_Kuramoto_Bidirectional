#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


GROUPS = {
    "drive0": lambda k: k == "gamma_phase_gain" or k.startswith("gamma_channel_proj."),
    "graph0": lambda k: k.startswith("graph_generator."),
    "kuramoto0": lambda k: k.startswith("kuramoto."),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0", type=Path, required=True)
    p.add_argument("--seed2", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    manifest = args.output_dir / "manifest.json"
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    s0 = torch.load(args.seed0, map_location="cpu", weights_only=True)
    s2 = torch.load(args.seed2, map_location="cpu", weights_only=True)
    if tuple(s0) != tuple(s2):
        raise ValueError("checkpoint keys differ")
    args.output_dir.mkdir(parents=True)
    rows = {}
    for name, selector in GROUPS.items():
        selected = [key for key in s2 if selector(key)]
        if not selected:
            raise ValueError(f"empty group {name}")
        hybrid = {key: (s0[key].clone() if key in selected else s2[key].clone()) for key in s2}
        for key in s2:
            expected = s0[key] if key in selected else s2[key]
            if not torch.equal(hybrid[key], expected):
                raise ValueError(f"hybrid verification failed: {name}/{key}")
        path = args.output_dir / f"seed2_{name}.pt"
        torch.save(hybrid, path)
        rows[name] = {"path": str(path), "sha256": sha(path), "copied_keys": selected}
    result = {
        "experiment": "SW0065 causal seed component swaps",
        "base": {"seed0": str(args.seed0), "seed2": str(args.seed2), "seed0_sha256": sha(args.seed0), "seed2_sha256": sha(args.seed2)},
        "hybrids": rows,
        "training_or_data_change": False,
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: len(row["copied_keys"]) for key, row in rows.items()}))


if __name__ == "__main__":
    main()
