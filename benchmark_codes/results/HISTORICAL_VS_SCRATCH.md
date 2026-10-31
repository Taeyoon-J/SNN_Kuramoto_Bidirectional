# Historical checkpoints versus the current scratch benchmark

Model numbers identify a model family here, not an identical training history.

| Item | Historical SW0253 / SW0165 transfer | Current SW0255 variant253 |
|---|---|---|
| Initial state | Trained SW0097 source core and trained reconstruction encoder | Random initialization |
| Grid | Trained16-grid core mapped to native32 | Fresh native64 (previous scratch condition:32) |
| Target-dataset training | None in the frozen transfer evaluation | 3500 unique TRAIN images x10 epochs |
| Encoder/core updates in this experiment | Both frozen for transfer | Encoder, graph and active core/gates trained |
| Training masks | None | None; runtime audit confirms TRAIN labels never opened |
| Loss reaching encoder | None during frozen transfer | Phase and positive-spike losses |
| Final supervised mask loss | None | None |

Variant254 additionally sends final CE/Dice mask loss through activity features
into the encoder/body. Variant252 detaches those features for its final mask
head, while its other body objectives still train the encoder/body. Connecting
an encoder to unsupervised phase/spike loss is different from connecting it to
supervised final-mask loss.

## Verified historical budget

The original seed0 SW0097 manifest points to SW0095 seed0; SW0095 points to the
selected SW0090 epoch01 checkpoint. The archived SW0090/SW0095/SW0097 methods
record the following downstream chain:

| Stage | Image exposures | Adam updates |
|---|---:|---:|
| SW0090 selected epoch01 | 70000 | 4375 |
| SW0095 continuation on the same70000 pool | 70000 | 4375 |
| SW0097 continuation from that pool | 4096 | 256 |
| Confirmed downstream subtotal | 144096 | 9006 |

This is70000 distinct images, not144096 distinct images. Earlier core training
and encoder pretraining are excluded, so this is not a complete lifetime total.
Server seed0 manifest SHA256 values checked during the audit:

- SW0097: b57c0e5a83440dc81582d2ea9433505008041e5182c9e680c5524feff8b10ff9
- SW0095: 68d7d93462738c8668c594fa1fe0a12700255c0f8719450456988189e97b09fb

Current scratch runs use3500 distinct images,35000 exposures and2190updates.
They also differ in trainable scope and optimization trajectory, so exposure
ratios alone cannot establish the cause of the score gap.

## What the completed scratch result proves

Previous native32 scratch253/CLEVR/seed0, fixed320 original-pixel evaluation:
FG-ARI0.428118 /Foreground IoU0.277787 /matched-object IoU0.240651.
The same fresh predictions scored on the historical fixed64 patch contract give
0.457315 /0.273522 /0.248168. Neither row is the historical frozen model's score.

The actual epoch-mean training loss decreased from46.1868 at epoch1 to28.4788
at epoch10; epoch9 was29.0075. All active parameter gradients/updates, including
encoder.kernels and encoder.bias, were audited. This is consistent with further
optimization being possible, but does not show that more training will improve
FG-ARI. The losses do not directly supervise final masks in253.

More exposure is not guaranteed to recover the historical score. In the archived
SW0090 three-seed comparison, FG-ARI decreased from0.774323 at epoch1 to0.677605
at epoch10. That is evidence against assuming monotonic improvement, not a causal
control for the current scratch run. Additional data diversity and repeated
passes must be distinguished; validation segmentation quality must be measured
before claiming either explanation or recovery.

## Native64 revision2026-10-31

Current253 uses4096 patches and top-k512 with spacing16/64. It keeps the same
224-parameter base encoder and unsupervised objectives. It starts fresh in a
separate output root; previous native32 checkpoints/results are preserved.
No64-grid benchmark score is inferred from the32-grid row. Actual CUDA
forward/backward, active parameter updates and optimizer restoration passed;
see `../validation/native64_253_revision.json`.
