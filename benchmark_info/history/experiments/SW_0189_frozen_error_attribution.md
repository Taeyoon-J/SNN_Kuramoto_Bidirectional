# SW_0189_frozen_error_attribution

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `a89c6155295e4a7a6f3e965156aa3371ff7f62895413da15be0d907841198911`

````markdown
# SW0189: posthoc frozen-mask error attribution

While SW0188 is training, inspect only the already completed SW0186 and SW0187
fixed320/native32 masks and comparable Slot70k archives. No new GPU inference,
optimizer steps, readout thresholds or metrics are introduced. Hash-pin both
complete source scores, every prediction archive and protocol before opening GT.
Recompute the three registered metrics against every saved seed/image vector.

Separately describe foreground misses, background false positives, raw positive
label count accuracy, same-object split pairs and cross-object merged pairs.
Predicted background label0 remains a cluster for FG pair errors, as in FG-ARI;
also identify cross-object pairs merged specifically through that background
cluster. Report patch-pair-weighted and object-balanced rates distinctly.
Report all three seeds and actual GT object-count strata, without selecting
favorable images. These diagnostics are descriptive, not a causal explanation,
a trained count classifier or a replacement FG-ARI evaluation.

Use the completed experiment comparisons to choose between support prediction,
fragmentation and inter-object separation as the next hypothesis. Do not promote
a new incumbent from this analysis or use validation masks for gradient updates.
No claim of unsupervised improvement, gate causality, unseen-data scaling or
full native32 joint70k training follows from this diagnostic.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `01054f2251e3a73c763f526e53482e349d60cfa73f14185b727d880a3b5989e4`

````markdown
# SW0189 ? completed frozen-mask error attribution

CPU-only analysis of 6,720 completed endpoint/seed/image rows. No model forward passes or optimizer steps. Six tests passed; all registered metrics recomputed within 3.34e-16. Fragmentation, cross-object merging, foreground misses and background leakage remain visible. Raw mask-label count is not a trained count classifier.

See [findings and limits](RESULT_KO.md). SW0188 remains the running model experiment; no incumbent changes.

Publication scope: code, cohort aggregate summary and execution receipt only. The image-level report remains private and is ignored by Git. Automatic approval review rejected its public push; no image-level diagnostic payload is included in the replacement public commit.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `774872d5f59db9b3e91410fa6d654c2736aa6f015e0897c0a749e246bf220d2f`

````markdown
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

````

</details>

## results_archive/analysis_v1/registered_README.md

<details>
<summary>registered_README.md</summary>

Source SHA256: `1c8990e54b32f859dae18679ec14e336893c69ecd1578028c065f5a8a8815801`

````markdown
# SW0189 — frozen-mask error attribution

CPU diagnostic prepared while SW0188 trains. Uses only completed SW0186/SW0187
and Slot fixed320 predictions; no new model training or inference.
See [scope and definitions](DESIGN_KO.md). Results pending.

````

</details>
