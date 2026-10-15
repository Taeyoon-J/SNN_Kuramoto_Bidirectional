# SW0163 evaluation recovery

The original predictor compared a SW0162-family source digest with a SW0161-family training binding. A read-only audit of the actual checkpoint matched all nine cache/preflight/training bindings with the SW0161 algorithm. Both algorithms produce different hashes for the same source; no model change or additional training is needed.

This separate entrypoint uses the exact SW0161 digest before and after prediction. It rejects changed parameters, preprocessing clip values, inconsistent head bindings, and the wrong digest family. Original experiment files, trained heads and failed attempts are preserved. Outputs go to the original experiment's `results_archive/source_lineage_recovery_20261015/`, which is physically centralized locally through the existing results junction.

The source rollout, actual Q/A capture, four trained heads, source-T replay control, fixed320 IDs, metrics, bootstrap and registered scientific gate are unchanged. Every output records its recovery provenance. Scoring validates all five prediction artifacts before opening evaluation masks, then independently recomputes the metrics.

Seven CPU regressions passed locally and on Frontier with zero skips and no CUDA initialization. Root AST review additionally verified ten forward/input/metric helpers unchanged and the entire scientific score function identical except additional provenance metadata. Prediction and score coordinators were launched as PIDs3615371/3615376 and both exact `/proc` module command lines were verified. This confirms registration, not completion or improved performance.

Evidence: `../SW_0163_relation_context_assignment/results_archive/source_lineage_recovery_launch_20261015/`. Failed-parent diagnosis is recorded in `source_digest_family_failure_20261015/` under the same experiment archive.
