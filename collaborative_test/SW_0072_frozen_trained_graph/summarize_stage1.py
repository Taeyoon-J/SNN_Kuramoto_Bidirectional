#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

M = ("fg_ari", "foreground_iou", "matched_object_iou")
p = argparse.ArgumentParser()
p.add_argument("--candidate-dir", type=Path, required=True)
p.add_argument("--baseline-dir", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = {}
for seed in (1, 2):
    rows[str(seed)] = {}
    for kind, path in (
        ("candidate", a.candidate_dir / f"candidate_seed{seed}_n32.json"),
        ("baseline", a.baseline_dir / f"candidate_seed{seed}_n32.json"),
    ):
        report = json.loads(path.read_text())
        assert report["ids"] == [1320, 1351] and report["ground_truth_used_for_prediction"] is False
        found = [row for row in report["sweep"] if row["affinity_mode"] == "spike" and row["synchrony_threshold"] == .35]
        assert len(found) == 1
        metrics = found[0]["scored_targets"]["our_hdf5"]["metrics"]
        rows[str(seed)][kind] = {key: float(metrics[key]) for key in M}
    rows[str(seed)]["delta"] = {key: rows[str(seed)]["candidate"][key] - rows[str(seed)]["baseline"][key] for key in M}
assert all(math.isfinite(value) for seed in rows.values() for group in seed.values() for value in group.values())
out = {
    "experiment": "SW0072 frozen trained seed0 graph stage1",
    "rows": rows,
    "advance": all(rows[str(seed)]["delta"][key] > 0 for seed in (1, 2) for key in M),
    "gate": "both weak seeds improve all three metrics over SW0066 graph-init0",
}
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
