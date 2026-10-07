"""Reproducibly summarize SW0099's corrected small validation traces."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INITIAL = HERE / "results"
CORRECTED = HERE / "results_corrected_dendrite_order"
OUT = HERE / "summary.json"
CONDITIONS = ("sw0095", "sw0097", "sw0098")
STAGES = (
    "graph_adjacency", "phase_theta_product", "sinusoidal_carrier_product",
    "delayed_gate", "exact_gated_carrier_product", "dendritic_output_product",
    "membrane_positive_product", "classifier_spike_positive_product",
)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
DISTANCE_BINS = ("0-2", "2-5", "5-10", "10-inf")


def load(path):
    return json.loads(path.read_text())


def images(record):
    return [row for batch in record["batches"] for row in batch["rows"]]


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def subtract(a, b):
    return None if a is None or b is None else a - b


def trace_mean(record, stage, field, distance=None):
    values = []
    for row in images(record):
        stats = row["stages"][stage]
        if field in ("auc", "same_minus_different"):
            stats = stats["overall"] if distance is None else stats["distance_bins_euclidean_patches"][distance]
            values.append(stats.get(field))
    return mean(values)


def main():
    graph_audit = load(HERE / "graph_frozen_audit.json")
    assert graph_audit["all_seeds_equal"]
    corrected = {}
    initial = {}
    endpoint_by_condition = {}
    for condition in CONDITIONS:
        corrected[condition], initial[condition], endpoint_by_condition[condition] = {}, {}, {}
        for seed in range(3):
            name = f"{condition}_seed{seed}.json"
            new = load(CORRECTED / name)
            old = load(INITIAL / name)
            assert new["status"] == "complete" and old["status"] == "complete"
            assert new["runner_sha256"] == "651f930319541382ffc620806a18e83b518c9242378ce09ed2aea50f17d44680"
            assert len(new["ids"]) == 16 and new["ids"] == list(range(1320, 1336))
            assert [len(batch["rows"]) for batch in new["batches"]] == [8, 8]
            assert all(row["image_id"] == iid for row, iid in zip(images(new), range(1320, 1336)))
            # Correcting fold-unflattening changes dendritic metrics only.
            for old_row, new_row in zip(images(old), images(new)):
                for stage in STAGES:
                    if stage == "dendritic_output_product":
                        continue
                    assert old_row["stages"][stage] == new_row["stages"][stage], (
                        f"non-dendrite trace changed: {condition} seed{seed} {stage}"
                    )
            corrected[condition][str(seed)] = new
            initial[condition][str(seed)] = old
            endpoint_by_condition[condition][str(seed)] = {
                key: mean(row["stages"]["endpoint"][key] for row in images(new))
                for key in (*METRICS, "predicted_groups", "target_groups")
            }

    stage_results = {}
    for stage in STAGES:
        conditions = {}
        for condition in CONDITIONS:
            conditions[condition] = {}
            for seed in range(3):
                r = corrected[condition][str(seed)]
                item = {"foreground_pair_auc": trace_mean(r, stage, "auc"),
                        "same_minus_different": trace_mean(r, stage, "same_minus_different"),
                        "distance_bins": {bin_name: {
                            "foreground_pair_auc": trace_mean(r, stage, "auc", bin_name),
                            "same_minus_different": trace_mean(r, stage, "same_minus_different", bin_name),
                        } for bin_name in DISTANCE_BINS}}
                if stage in ("membrane_positive_product", "classifier_spike_positive_product"):
                    rows = images(r)
                    edges = [row["stages"][stage]["within_object_edge_recall_and_cross_object_false_edge_rate"]
                             for row in rows]
                    objects = [obj for row in rows for obj in row["stages"][stage]["threshold_0p5_edges"]]
                    item["threshold_0p5"] = {
                        "mean_image_within_object_edge_recall": mean(x["within_object_edge_recall"] for x in edges),
                        "mean_image_cross_object_false_edge_rate": mean(x["cross_object_false_edge_rate"] for x in edges),
                        "mean_object_largest_induced_component_fraction": mean(x["largest_component_fraction"] for x in objects),
                        "mean_object_induced_fragments": mean(x["fragments"] for x in objects),
                        "ground_truth_objects": len(objects),
                    }
                conditions[condition][str(seed)] = item
        deltas = {}
        for seed in range(3):
            a, b = conditions["sw0098"][str(seed)], conditions["sw0097"][str(seed)]
            deltas[str(seed)] = {
                "foreground_pair_auc": subtract(a["foreground_pair_auc"], b["foreground_pair_auc"]),
                "same_minus_different": subtract(a["same_minus_different"], b["same_minus_different"]),
                "distance_bins": {k: {
                    "foreground_pair_auc": subtract(a["distance_bins"][k]["foreground_pair_auc"], b["distance_bins"][k]["foreground_pair_auc"]),
                    "same_minus_different": subtract(a["distance_bins"][k]["same_minus_different"], b["distance_bins"][k]["same_minus_different"]),
                } for k in DISTANCE_BINS},
            }
        stage_results[stage] = {"condition_seed_means": conditions, "SW0098_minus_SW0097": deltas,
                                "three_seed_mean_delta": {
                                    "foreground_pair_auc": mean(x["foreground_pair_auc"] for x in deltas.values()),
                                    "same_minus_different": mean(x["same_minus_different"] for x in deltas.values()),
                                },
                                "seeds_with_lower_auc": sum(x["foreground_pair_auc"] < 0 for x in deltas.values()),
                                "seeds_with_lower_separation": sum(x["same_minus_different"] < 0 for x in deltas.values())}

    earliest = None
    for stage in STAGES:
        r = stage_results[stage]
        if r["three_seed_mean_delta"]["foreground_pair_auc"] < 0 and r["seeds_with_lower_auc"] >= 2:
            earliest = {"stage": stage, "criterion": "SW0098-SW0097 foreground-only pair AUC lower in at least two seed means and three-seed mean",
                        "evidence": r}
            break

    paired_endpoints = {}
    for seed in range(3):
        old97 = images(corrected["sw0097"][str(seed)])
        new98 = images(corrected["sw0098"][str(seed)])
        paired_endpoints[str(seed)] = [{
            "image_id": a["image_id"],
            **{m: new["stages"]["endpoint"][m] - a["stages"]["endpoint"][m] for m in METRICS},
        } for a, new in zip(old97, new98)]

    endpoint_means = {}
    for condition in CONDITIONS:
        endpoint_means[condition] = {
            m: mean(endpoint_by_condition[condition][str(s)][m] for s in range(3))
            for m in METRICS
        }
    endpoint_delta = {m: endpoint_means["sw0098"][m] - endpoint_means["sw0097"][m] for m in METRICS}
    non_dendrite_equal = True
    runner_hashes = {corrected[c][str(s)]["runner_sha256"] for c in CONDITIONS for s in range(3)}
    assert len(runner_hashes) == 1
    result = {
        "experiment": "SW0099_stage_flow_diagnosis",
        "status": "complete_corrected_dendrite_order",
        "validation_ids": [1320, 1335],
        "images_per_checkpoint": 16,
        "batches_per_checkpoint": [8, 8],
        "inference_steps": 1024,
        "settle": 512,
        "classifier": "positive per-component Pearson product; threshold0.50; min group2; largest component background",
        "gt_usage": "loaded after forward, diagnostic scoring only",
        "distance_bins_euclidean_patch_distance": {"labels": list(DISTANCE_BINS), "definition": "(lower, upper], diagonal excluded"},
        "provenance": {"diagnostic_runner_sha256": next(iter(runner_hashes)),
                       "executed_coordinator_sha256": "0f5d54bfa2ec0871ca08e3334bf5229a7b0c190a01f8cea044a122e9a863c5c2",
                       "current_coordinator_sha256": "28047981ef45dea81c787135e281e85462634e44e8a0725c519c206affeb0e8c",
                       "patch_v2_sw_head": "e4ca2fe", "patch_v2_reference_head": "e2389f5",
                       "current_sandbox_gpu_authorization": "user authorized GPUs0-3; only genuinely free devices used",
                       "actual_gpu": 0},
        "execution_audits": {
            "corrected_synthetic_dendrite_order_check": "passed",
            "corrected_smoke_endpoint_max_abs_diff_vs_archived": {"fg_ari": 0.0, "foreground_iou": 0.0, "matched_object_iou": 0.0},
            "initial_suite_remote_runner_sha256": "97860e09464e83417c5cc7d0a0a8c9b16300a8cff0e91eccce21d933572869b7",
            "initial_suite_dendritic_stage": "invalidated due to folded [B*D,T,N] being reinterpreted as [T,B,D,N]; original JSON retained in results/",
            "non_dendrite_stage_and_endpoint_initial_vs_corrected_exact_equal": non_dendrite_equal,
            "corrected_coordinator_state_path_issue": "executed coordinator updated shared state.json rather than tagged state_corrected_dendrite_order.json; process terminal state was confirmed complete, all corrected outputs used isolated results_corrected_dendrite_order paths, and the local coordinator was fixed",
            "graph_generator_state_exact_equal_SW0095_SW0097_SW0098": graph_audit,
        },
        "per_condition_seed_endpoint_means_over16": endpoint_by_condition,
        "three_seed_endpoint_means": endpoint_means,
        "SW0098_minus_SW0097_three_seed_endpoint_delta": endpoint_delta,
        "SW0098_minus_SW0097_per_image_endpoint_deltas": paired_endpoints,
        "stage_results": stage_results,
        "earliest_consistent_degradation": earliest,
        "limitations": ["validation diagnostic subset contains 16 images, not the full320 contract",
                        "no holdout or reserved split was read", "no threshold or model setting was tuned"],
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"means": endpoint_means, "endpoint_delta": endpoint_delta,
                      "earliest_consistent_degradation": None if earliest is None else earliest["stage"],
                      "non_dendrite_equal": non_dendrite_equal}, indent=2))


if __name__ == "__main__":
    main()
