# SW0190: both oracle-count spectral recipes fail admission

Three frozen SW0186 source models completed the same16 TRAIN images each.
Both global spectral alternatives perform worse in every source seed despite
receiving the scalar true visible-object count. Neither is admitted to a
learned-count expansion. Close these two specified recipes without a sweep.

| TRAIN16 method, three-seed mean | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Existing production baseline | .872016 | .561423 | .751356 |
| Spectral, existing foreground support | .782037 | .561423 | .581457 |
| Spectral, all patches and model-chosen BG | .771040 | .551537 | .533890 |

| Seed | Baseline FG | Existing-support spectral FG | All-patch spectral FG |
|---|---:|---:|---:|
| 0 | .880259 | .758035 | .800179 |
| 1 | .843351 | .782438 | .740958 |
| 2 | .892438 | .805639 | .771983 |

FG differences are -.089979 and -.100976; positive seed count is0 for both.
The fixed-support alternative preserves Foreground IoU exactly, so its drop
cannot be explained by changed foreground membership. This points against this
particular global partition of Q; it does not prove all count-aware or global
assignment models are unhelpful. The original graph/geodesic production structure
should be retained in the next training-loss experiment.

These are TRAIN diagnostic numbers, not fixed320 validation results. The .8720
baseline is not a new .87 benchmark achievement. Count was supplied, not learned;
no count-classifier accuracy, automatic object counting, unsupervised improvement
or gate causality is claimed. Oracle-count feasibility here is not a mathematical
upper bound on every possible model or partition algorithm.

Eight tests and the actual CLI passed locally/server. All three actual frozen
CPU models matched the pinned SW0186 incumbent. GPU workers preserved their
entire model tensor hashes and made zero optimizer updates. TRAIN masks supplied
only scalar counts to the alternative readouts; pixel memberships were used only
for scoring. No validation RGB/masks were read. Source hashes, per-seed report
hashes, metric-vector arithmetic and the preregistered admission rule were checked.
No independent saved-mask replay was performed for this small pilot; do not
describe the arithmetic check as an independent metric recomputation.

[Public cohort summary](results_archive/summary_v1/summary.json).
Raw imagewise TRAIN reports remain private and ignored by Git. No incumbent
changes: SW0186 supervised FG and SW0165 unsupervised anchors remain separate.
