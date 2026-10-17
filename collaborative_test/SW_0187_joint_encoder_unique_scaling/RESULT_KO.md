# SW0187: additional unique images did not improve FG-ARI

All three seeds, both trained arms and all fixed320/native32 predictions are
complete. The independent v2 audit recomputed 3,840 candidate endpoint-image
rows and 960 Slot rows, all means, paired confidence intervals and gates.

| Model / additional adaptation data | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0187 control: repeated512 | 0.829479 | 0.608872 | 0.680583 |
| SW0187 candidate: unique1024 | 0.826556 | 0.577652 | 0.679338 |
| SW0186 incumbent | 0.834681 | 0.569040 | 0.689802 |
| Comparable Slot70k epoch10 | 0.781652 | 0.204360 | 0.203165 |

| Seed | control FG-ARI | candidate FG-ARI |
|---|---:|---:|
| 0 | 0.848430 | 0.854629 |
| 1 | 0.787443 | 0.782767 |
| 2 | 0.852564 | 0.842273 |

The primary candidate-minus-contemporaneous-control FG difference is -0.002923;
paired-image 95% CI [-0.007999, 0.002256]. This does not establish an FG decrease
or increase. Foreground IoU falls by 0.031219, CI [-0.039247, -0.023166].
Object IoU changes by -0.001246, CI [-0.006058, 0.003537]. The scaling gate fails.
Against historical SW0186, candidate FG changes by -0.008124,
CI [-0.014093, -0.002209]. No incumbent changes and no FG.87 claim are made.
Both SW0187 arms exceed the comparable Slot means in all three metrics.

The original RGB CNN genuinely learned in all six models; accepted control/
candidate updates are 37/39, 31/35 and 47/45. An encoder frozen by mistake is
therefore ruled out for this run. This does not establish that the loss, graph
or discrete assignment uses the additional data effectively. Close this exact
recipe; do not spend another cycle on small learning-rate/temperature changes.
Any next experiment should address the objective-to-assignment mechanism or
test continued learning from the actual incumbent with an explicit budget.

The 512/1024 counts are additional supervised adaptation data, not total
foundation training data. Source097 already inherits full70k RGB training.
See [verified foundation provenance](FOUNDATION_PROVENANCE.md).
This is not a fresh native32 joint-encoder70k run, unsupervised improvement or
gate-causality proof. The seed1 resource requeue excluded and preserved 15 old
proposals, which remain explicitly reported as extra compute.

The original audit v1 expected an obsolete experiment identifier and failed
before metric recomputation. Its source and failure receipt remain preserved.
Separate v2 corrects only that identifier and uses a separate output path;
the producer contract passes actual-entry-point tests locally and on server.
Training, masks, score code, metric definitions and thresholds were unchanged.
The v2 independent audit then passed. No GPU training or inference was repeated
to correct the audit. Maximum metric absolute error is recorded in the raw audit.

Score SHA256: `31bcb373a41fe533d0f2218c5861e6c7022996396536c79f9e8279d0a8716b11`.
Audit SHA256: `583b770aaa41b878b14e4de87092437d062263fb6823cc8118a59edc94e998ce`.

[Raw score](results_archive/score_v2/evaluation.json),
[independent audit](results_archive/score_v2/completion_audit.json),
[archive receipt](results_archive/score_v2/archive_receipt.json).
