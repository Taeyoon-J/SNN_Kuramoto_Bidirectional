"""CPU-only diagnosis of completed runs; never uses GT to form predictions."""
import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    rows = []
    changes = []
    for seed in range(3):
        log = (HERE / f"results/training_seed{seed}.log").read_text(encoding="utf-8-sig")
        losses = {int(e): float(v) for e, v in re.findall(
            r"Epoch\s+(\d+)/\d+\s+\|\s+loss=([0-9.]+)", log)}
        assert set(losses) == set(range(1, 11)), (seed, losses)
        per_epoch = {}
        for epoch in (1, 3, 10):
            data = load(ROOT / f"collaborative_test/SW_0090_large_unique_scale/results/full320/seed{seed}_epoch{epoch}.json")
            assert data["images"] == 320 and data["ids"] == [1320, 1639]
            assert data["ground_truth_used_for_prediction"] is False
            sweep = data["sweep"][0]
            scored = sweep["scored_targets"]["our_hdf5"]
            values = scored["per_image"]["fg_ari"]
            assert len(values) == 320 and all(math.isfinite(v) for v in values)
            assert abs(mean(values) - scored["metrics"]["fg_ari"]) < 1e-12
            row = {
                "seed": seed, "epoch": epoch, "train_loss": losses[epoch],
                "metrics": scored["metrics"],
                "object_count": scored["object_count"],
                "predicted_foreground_fraction": sweep["predicted_foreground_fraction"],
                "target_foreground_fraction": scored["target_foreground_fraction"],
                "phase_product_plv_mean": data["phase_diagnostic"]["product_plv_mean"],
            }
            rows.append(row)
            per_epoch[epoch] = values
        changes.append({
            "seed": seed,
            "epoch10_minus_epoch1_fg_ari": mean(per_epoch[10]) - mean(per_epoch[1]),
            "images_with_lower_fg_ari": sum(b < a for a, b in zip(per_epoch[1], per_epoch[10])),
            "images_with_higher_fg_ari": sum(b > a for a, b in zip(per_epoch[1], per_epoch[10])),
            "epoch10_minus_epoch1_train_loss": losses[10] - losses[1],
        })
    epochs = []
    for epoch in (1, 3, 10):
        selected = [r for r in rows if r["epoch"] == epoch]
        epochs.append({
            "epoch": epoch,
            "metrics": {k: mean(r["metrics"][k] for r in selected)
                        for k in selected[0]["metrics"]},
            "object_count": {k: mean(r["object_count"][k] for r in selected)
                             for k in selected[0]["object_count"]},
        })
    # Counterexample matching the two formulas found in repository code.
    # Perfectly opposite binary traces have centered correlation -1.
    anticorrelated_components = [-1.0] * 4
    result = {
        "scope": "Completed SW0090 artifacts and code audit; no new training",
        "held_out_ids": [1320, 1639], "images_per_evaluation": 320,
        "seeds": [0, 1, 2], "rows": rows, "epoch_means": epochs,
        "paired_image_changes": changes,
        "optimizer_update_budget": {
            "batch_size": 16, "small_2500_epochs10": math.ceil(2500 / 16) * 10,
            "large_70000_epochs1": math.ceil(70000 / 16),
            "large_70000_epochs10": math.ceil(70000 / 16) * 10,
            "large10_to_small10_ratio": math.ceil(70000 / 16) / math.ceil(2500 / 16),
            "caveat": "SW0072 seed0 reused SW0055 with a trained graph; other seeds froze that graph. Not an identical scaling control.",
        },
        "confirmed_affinity_formula_mismatch": {
            "training": "product(abs(centered_correlation_per_component))",
            "classifier": "product(max(centered_correlation_per_component, 0))",
            "training_function": "training/train_s2net_core.py:_component_spike_synchrony -> loss_function.py:signal_synchrony",
            "classifier_function": "spike_classifier.py:spike_synchrony_affinity",
            "four_anticorrelated_components": {
                "training_affinity": math.prod(abs(c) for c in anticorrelated_components),
                "classifier_affinity": math.prod(max(c, 0) for c in anticorrelated_components),
            },
            "limitation": "Counterexample proves semantic mismatch, not its causal share of observed degradation. Actual signed affinities require checkpoint diagnostics.",
        },
    }
    target = HERE / "results/diagnosis.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"epoch_means": epochs, "paired_image_changes": changes}, indent=2))


if __name__ == "__main__":
    main()
