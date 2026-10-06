import argparse, json, statistics
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
epochs = {}
for epoch in (1, 3, 10):
    seeds = {}
    for seed in range(3):
        report = json.loads((a.results / f"seed{seed}_epoch{epoch}.json").read_text())
        assert report["ids"] == [1320, 1639] and not report["ground_truth_used_for_prediction"]
        row = report["sweep"][0]
        assert row["synchrony_threshold"] == .5
        seeds[str(seed)] = row["scored_targets"]["our_hdf5"]["metrics"]
    epochs[str(epoch)] = {"seeds": seeds, "mean": {
        key: statistics.mean(seeds[str(seed)][key] for seed in range(3)) for key in METRICS}}
out = {
    "experiment": "SW0090 large unique-scene scaling",
    "train_scenes": 70000,
    "validation_ids": [1320, 1639],
    "readout": "spike connected components; 1024 steps; settle 512; threshold 0.50",
    "epochs": epochs,
    "baseline_sw0072_2500x10_mean": {
        "fg_ari": .7814074870445242,
        "foreground_iou": .47803820815434037,
        "matched_object_iou": .6167461822430291,
    },
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

