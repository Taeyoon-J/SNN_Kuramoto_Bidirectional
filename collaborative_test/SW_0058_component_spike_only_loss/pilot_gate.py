#!/usr/bin/env python3
"""Require the completed, validator-approved SW0054 count-32 pilot."""
import argparse
import importlib.util
import json
from pathlib import Path


def validate_pilot(folder: Path) -> dict:
    raw_path = folder / "seed0_epoch25_short_n32_pilot_T256_settle64.json"
    summary_path = folder / "seed0_epoch25_short_n32_pilot_summary.json"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("SW0054 raw pilot and validated summary are both required")
    spec = importlib.util.spec_from_file_location("sw54_summary", folder.parent / "summarize.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load SW0054 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    module.validate_report(raw)
    expected = module.summarize(raw)
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    if expected != stored:
        raise ValueError("stored SW0054 summary does not match raw pilot through current validator")
    if stored.get("count") != 32 or stored.get("ids") != [1320, 1351]:
        raise ValueError("SW0054 gate requires the completed 32-image aligned pilot")
    return stored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    args = p.parse_args()
    result = validate_pilot(args.folder)
    print(json.dumps({"pilot_validated": True, "checkpoint_sha256": result["checkpoint_sha256"],
                      "count": result["count"], "ids": result["ids"]}))


if __name__ == "__main__":
    main()
