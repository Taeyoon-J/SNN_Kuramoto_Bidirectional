import argparse
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    initial = json.loads((ROOT / "collaborative_test/SW_0090_large_unique_scale/results/full320/seed0_epoch1.json").read_text())
    initial_scores = initial["sweep"][0]["scored_targets"]["our_hdf5"]["per_image"]
    rows = {"unchanged_source": {k: mean(v[:80]) for k, v in initial_scores.items()}}
    for arm in ("absolute_frozen", "positive_frozen", "positive_graph", "positive_joint"):
        d = json.loads((args.results / arm / "evaluation.json").read_text())
        assert d["images"] == 80 and d["ids"] == [1320, 1399]
        assert d["ground_truth_used_for_prediction"] is False
        rows[arm] = d["sweep"][0]["scored_targets"]["our_hdf5"]["metrics"]
    result = {"pilot_only": True, "source_model_seed": 0, "shuffle_seed": 17,
              "images": 80, "ids": [1320, 1399], "rows": rows,
              "not_a_full70000_or_three_seed_success_claim": True}
    (args.results / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
