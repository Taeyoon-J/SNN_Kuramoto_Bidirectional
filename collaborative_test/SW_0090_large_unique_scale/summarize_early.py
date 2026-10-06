import json, statistics, sys
from pathlib import Path

metrics = ("fg_ari", "foreground_iou", "matched_object_iou")
root, output = Path(sys.argv[1]), Path(sys.argv[2])
epochs = {}
for epoch in (1, 3, 10):
    seeds = {}
    for seed in (0, 1):
        report = json.loads((root / f"seed{seed}_epoch{epoch}.json").read_text())
        assert report["ids"] == [1320, 1639] and not report["ground_truth_used_for_prediction"]
        row = report["sweep"][0]
        assert row["synchrony_threshold"] == .5
        seeds[str(seed)] = row["scored_targets"]["our_hdf5"]["metrics"]
    epochs[str(epoch)] = {"seeds": seeds, "mean_seed01": {
        key: statistics.mean(seeds[str(seed)][key] for seed in (0, 1)) for key in metrics}}
summary = {"experiment": "SW0090 early seed0/1 evaluation", "epochs": epochs,
           "readout": "fixed full320, steps1024, settle512, spike threshold .50"}
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

