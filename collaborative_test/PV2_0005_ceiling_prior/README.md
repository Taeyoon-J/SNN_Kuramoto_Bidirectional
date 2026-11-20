# PV2_0005 — the ceiling, re-measured, and two graph priors

**completed.** Parent: `PV2_0004`. Contract v1, split v1. Validation, 100 images,
seed 0. Exploration: no 3-seed mean here, so nothing below is a score.

## Why

`PV2_0004` showed that a threshold carried over from a different measure had made
per-component spikes look useless. The 0.610 ceiling from a ground-truth coupling
graph, which had been steering the work, was measured with that same stale
threshold and the old readout, so it could not be trusted either.

## The ceiling moved down, and it changes the plan

ORACLE diagnostic -- it builds the coupling graph from the true patch labels, so
it is not a score. Per-component spikes, swept over eight thresholds:

| sync | fg_ari | foreground_iou | matched_object_iou | predicted_fg |
| --- | --- | --- | --- | --- |
| 0.0005 | 0.2218 | 0.3163 | 0.1914 | 0.0345 |
| 0.002 | 0.2681 | 0.4151 | 0.2539 | 0.0524 |
| 0.005 | 0.3077 | 0.4954 | 0.2948 | 0.0708 |
| **0.010** | 0.3291 | **0.5612** | 0.3287 | 0.0918 |
| 0.020 | 0.3420 | 0.5556 | 0.3485 | 0.1036 |
| 0.040 | 0.3637 | 0.5408 | 0.3717 | 0.1213 |
| 0.080 | 0.3652 | 0.4944 | **0.3975** | 0.1409 |

The ceiling is **0.561, not 0.610**, and the learned graph already reaches about
0.49. A perfect coupling graph is worth roughly **+0.06**, so **0.700 is not
reachable by improving the graph** -- which is what the last several experiments
were doing.

Foreground ARI is the sharper signal: under the oracle graph it is **0.329**,
*below* the 0.511 the learned graph gets. Perfect coupling makes grouping worse.
Whatever limits the result is downstream of the graph, in how spikes are turned
into masks.

One caveat on reading the oracle row: ORCPC is its own training run, trained with
the graph forced, so its spiking layers adapted to a coupling the learned model
never sees. It bounds what this readout extracts given ideal coupling; it is not
`PV2_0004` with its graph swapped.

## The two priors

| run | sync | fg_ari | foreground_iou | matched_object_iou | predicted_fg |
| --- | --- | --- | --- | --- | --- |
| `d35` (decay 0.35) | 0.002 | 0.4867 | **0.4894** | 0.3139 | 0.1050 |
| `d35` | 0.030 | **0.5888** | 0.4474 | **0.4092** | 0.2322 |
| `g5` (geodesic 5) | 0.010 | 0.4180 | 0.4178 | 0.2735 | 0.2022 |
| `g5` | 0.030 | 0.4402 | 0.3668 | 0.2946 | 0.2562 |

Tightening the spatial prior from 0.55 to 0.35 helps on all three. Running five
geodesic relaxation rounds instead of three hurts all three and inflates
predicted foreground to 0.26 against a target of 0.126 -- more rounds let the
coupling leak across boundaries rather than sharpening them.

## A bug this experiment produced, and the fix

The first ceiling sweep returned an empty table. It passed `--oracle-graph`,
which did not exist on `evaluate_model.py` -- I had assumed the training flag had
an evaluation counterpart -- and `2>/dev/null` swallowed the argparse error, so a
failed run came back looking like a finished one with no rows.

Worse, the evaluator built its core with `graph_mode="learned"` unconditionally
and had no oracle path at all, so even a correct invocation would have scored the
forced-graph checkpoint on a graph generator that was never trained.

Fixed: `evaluate_model.py` now takes `--oracle-graph` and sets `core.forced_graph`
per batch from the true patch labels with the same row normalisation training
uses, and records `oracle_graph` in its report so the result stays separable as a
diagnostic. Sweeps no longer discard stderr and print FAILED with the tail of the
output instead of an empty row. The earlier `GEOPCBD` run that "produced no
output" is likely the same failure mode.

## Next

`PV2_0006`: train the spiking layers towards preserving synchrony with
`--spike-plv-weight`, on `d35`'s graph. Until now the dendritic and membrane time
constants were drawn from sigmoid(U(-4,0)) and never trained towards keeping
synchrony, because the objective read the phases -- which is the readout-side
gap the ceiling measurement points at.
