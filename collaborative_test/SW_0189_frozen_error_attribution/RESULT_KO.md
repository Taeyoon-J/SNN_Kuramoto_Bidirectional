# SW0189: completed frozen-mask error attribution

No model training or inference was performed. All 6,720 endpoint/seed/image
rows from the completed SW0186, SW0187 and comparable Slot archives were checked
against their already published registered metric vectors. Maximum discrepancy
was 3.3306690738754696e-16. Six semantic tests passed locally and on server,
including predicted-background merging, splitting, background leakage, arbitrary
positive label IDs and undefined denominators.

Descriptive three-seed/image means for the production standalone endpoints:

| Setting | FG-ARI | FG missed | BG false positive | Positive labels/image | Raw count exact |
|---|---:|---:|---:|---:|---:|
| SW0186 incumbent | .8347 | 7.21% | 26.28% | 9.90 | 15.83% |
| SW0187 repeated512 | .8295 | 7.36% | 22.65% | 9.56 | 16.77% |
| SW0187 unique1024 | .8266 | 7.92% | 25.15% | 10.12 | 14.06% |
| Comparable Slot70k | .7817 | 20.64% | 79.74% | 9.71 | 7.19% |

The fixed320 GT has 6.3125 visible objects per image on the modal patch grid.
Positive label count includes background-only false-positive segments and
fragments. Raw count exact is equality of that count with the GT object count,
not the accuracy of an independently trained count classifier. Do not attribute
the entire count excess to foreground fragmentation. Slot background rates
depend on the existing registered perimeter-background convention; no remapping
or evaluation change was made for this diagnostic.

| Setting | Same-object patch-pair split | Different-object patch-pair merge | Merge via predicted BG |
|---|---:|---:|---:|
| SW0186 incumbent | 11.70% | 4.55% | .76% |
| SW0187 repeated512 | 11.66% | 4.82% | .89% |
| SW0187 unique1024 | 12.23% | 4.71% | 1.00% |
| Comparable Slot70k | 22.35% | 1.98% | .37% |

Split and merge columns have different denominators. Their percentages cannot
be directly compared as shares of total FG-ARI error. Predicted background
remains a cluster for these foreground pair errors, matching FG-ARI semantics.
The private raw report also contains object-balanced errors, all conditional
endpoints, each seed/image and GT object-count strata. Only cohort aggregates
are shared publicly; image-level GT-derived diagnostics remain local/server-side.

The findings support testing whole-object consistency and explicit assignment/
count handling rather than another small QCE learning-rate or temperature
sweep. Both fragmentation and cross-object merging remain; support prediction
also has room to improve. The failed local connectivity auxiliary from SW0185
should not be repeated under a different name. A new hypothesis should test a
global partition/assignment mechanism with clear training-to-production binding.
SW0188's incumbent continuation remains the running model experiment and is
not altered or delayed by this CPU diagnostic.

This analysis identifies observable errors, not their architectural cause.
No incumbent changes, .87 result, gate-causality claim or data-scaling success
are established. These validation masks were used for posthoc diagnosis only.

Report SHA256: `6636d76f8613597fce918a018914c1107f1f87c591572373c268add27960ca1e`.
[Public cohort summary](results_archive/analysis_v1/summary.json),
[execution receipt](results_archive/analysis_v1/receipt.json).

The image-level report is preserved privately and excluded from publication.
Automatic approval review rejected its public push; the public commit contains
code, aggregate statistics and execution provenance only.
