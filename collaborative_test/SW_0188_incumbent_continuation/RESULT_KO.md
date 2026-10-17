# SW0188: incumbent continuation did not improve FG

All three training runs, both fixed320/native32 prediction endpoints and the
independent actual-mask audit are complete. Retaining the complete SW0186 weights
and adding another supervised TRAIN512 twice/64-proposal stage with reset Adam
does not pass the improvement gate. Close this exact continuation recipe.

| Model | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0188 continuation | .831401 | .559170 | .681908 |
| SW0186 incumbent | .834681 | .569040 | .689802 |
| Comparable Slot70k epoch10 | .781652 | .204360 | .203165 |

| Seed | FG-ARI | Foreground IoU | Object IoU | New accepted updates |
|---|---:|---:|---:|---:|
| 0 | .835862 | .790242 | .680340 | 33 |
| 1 | .813989 | .432728 | .686333 | 28 |
| 2 | .844351 | .454540 | .679052 | 25 |

Against SW0186, FG changes by -.003280, paired-image95% CI
[-.009268,.002801]. This establishes neither an FG improvement nor a definite
FG decrease. Foreground IoU changes by -.009870, CI[-.019725,-.000340];
Object IoU changes by -.007894, CI[-.013903,-.001883]. Only seed2 improves FG.
All three overall means still exceed comparable Slot under the fixed contract,
but no incumbent change, .87 result or benefit of extra optimization is claimed.

Conditional means are .799926/.609975/.653802. Both endpoints and all three seeds
were frozen before the new GT score. The independent audit recomputed actual
1,920 candidate endpoint/image rows and960 Slot rows, modal GT, means, confidence
intervals, source encoder learning and gates. Local archival also bound all
three actual training manifests, frozen protocols and original SW0186 starts.

All three source/CNN/Adam stage bindings passed: each delivered run starts exactly
from its same-seed completed SW0186 weights, not fresh Source097. The five-group
Adam reset is explicit. Its effect is not separated from the extra64 proposals.
The old resource-interrupted seed1 partial (11 proposals,6 accepted) is preserved
and excluded from delivered weights. Delivered runs total192 proposals, plus11
completed proposals of disclosed overhead. Replacement seed1 passed its own
GPU admission. No SW0188 validation score selected the replacement.

These are extra supervised native32 adaptations; the inherited Source097 core
already saw the full70k RGB pool. A new native32 joint-encoder70k pass remains
outstanding. Comparison uses the same evaluation contract, not identical model
training targets/budgets. Unsup improvement, gate causality and optimal gamma
ordering are not established. SW0165 remains the separate unsupervised anchor.

SW0187 data expansion and this simple continuation both failed to improve FG.
SW0190 separately rejects two oracle-count spectral readouts on TRAIN16.
Keep the successful production partition and investigate a training objective
that responds to actual partition errors; do not rescue these recipes with
another small learning-rate, temperature or clustering sweep.

Score SHA256: `f7899e1215e3bb4cd341184ceb941663d221b3a1e0bcfe9d12f5577475375490`.
Audit SHA256: `f9bb6b2d8e825291f683b6b6cec52f8a672ed95a35ff621f88400ee90b5c4d93`.
[Public aggregate score](results_archive/score_v1/summary.json),
[independent audit](results_archive/score_v1/completion_audit.json),
[archive receipt](results_archive/score_v1/archive_receipt.json).
The imagewise score is preserved privately and excluded from Git publication.
