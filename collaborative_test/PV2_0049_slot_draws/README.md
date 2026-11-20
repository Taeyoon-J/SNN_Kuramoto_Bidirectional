# PV2_0049 — the matched Slot baseline is 0.6635, and we beat it on all three

**completed. This supersedes `PV2_0047`, which was two seeds and read the wrong
number.** Parent `PV2_0047`. Validation 300, contract v1, the same split and the
same evaluation function our own scores use.

## What `PV2_0047` got wrong

It reported matched Slot Attention at **0.7307** from two seeds and concluded that
Slot Attention was ahead of us on fg_ari by 0.006. Two things were wrong.

**The third seed was missing.** Seed 2 had died twice to CUDA OOM. It finished here
and scores **0.5377**, far below seeds 0 and 1.

**Inference was never seeded.** Slot Attention samples its slots from
`slots_mu + exp(slots_log_sigma) * tf.random.normal(...)` on every forward pass,
and `run_slot_attention.py` seeded nothing, so each run of one checkpoint produced
different masks. Re-scoring the same two checkpoints moved seed 0 from 0.7223 to
0.7234 and seed 1 from 0.7391 to 0.7279. The spread being attributed to training
seeds contained inference noise of unknown size.

## What was run

A `--seed` was added to `run_slot_attention.py`, placed after the local
`import tensorflow as tf` inside `main()` and before `build_model`. Three
checkpoints were then each evaluated under three seeded draws — nine inference
runs, 300 validation images each. Nothing about the models changed.

Verified first: two runs at the same seed produce **bitwise identical** patch
labels.

## Result: the draw is not the story

| checkpoint | draw 10 | draw 11 | draw 12 | mean | draw std |
| --- | --- | --- | --- | --- | --- |
| seed 0 | 0.7214 | 0.7227 | 0.7261 | **0.7234** | 0.0024 |
| seed 1 | 0.7267 | 0.7344 | 0.7272 | **0.7294** | 0.0043 |
| seed 2 | 0.5354 | 0.5459 | 0.5319 | **0.5377** | 0.0073 |

Inference noise is **~0.005**; variance across training checkpoints is **0.1090**.
A factor of twenty. Seed 2's 0.54 is a property of that checkpoint, not an unlucky
sample.

## Headline comparison

| metric | ours (3-seed) | matched SA (3-seed) | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7250** ± 0.0022 | 0.6635 ± 0.1090 | **+0.0615** | **+9.3%** |
| `patch_foreground_iou` | **0.7188** | 0.1184 ± 0.0093 | **+0.6004** | **+507%** |
| `patch_matched_object_iou` | **0.4953** | 0.0869 ± 0.0037 | **+0.4084** | **+470%** |

**We are ahead on all three.** Previous claim (`PV2_0047`): behind on fg_ari by
0.0057. New: ahead by 0.0615.

Our score comes from the spike classifier's masks, as the contract requires;
`PV2_0043` measured that binary spike timing carries the affinity at r = 0.9822
(0.9808 / 0.9807 on these graph-frozen checkpoints).

## A scoring fix made in the same test

`score_slot_attention.py` chose its background slot **once for the whole split**
(`np.bincount(labels.reshape(-1)).argmax()`). With slots resampled per forward
pass, slot index k has no consistent role between images, so one global index is
background in some and an object in others — and our own readout never has that
handicap, since it takes the largest component *of each image*.

Fixed to choose per image, with the global rule kept in the report for
comparison. The effect is small: foreground IoU 0.1146 → 0.1146–0.1208, predicted
background fraction 0.15 → 0.28 against a target of 0.8824. fg_ari is unchanged,
being invariant to relabelling outside the foreground patches.

So Slot Attention calling ~72% of patches foreground is a property of the model,
not an artifact of the scoring. It groups foreground patches about as well as we
do and cannot locate objects.

## Limitations

- **Budget.** 60,000 steps at batch 32 = 320 epochs, against the official 457.
  This is a **lower bound** on Slot Attention. Seed 2 at 0.5377 is what drags the
  mean down, and more training could lift it.
- **Our three seeds share one graph** (`PV2_0044`'s freeze), so the variance the
  graph contributes is removed by construction and this is not three fully
  independent runs.
- **Validation, not test.** Test is read once; the last test figure remains
  `PV2_0008` at 0.6075 / 0.6755 / 0.4298.

## Next

The goal's first condition is still unmet: 0.7250 against 0.75. Closed since:
`PV2_0052` (integration window, +0.0005). Open: `PV2_0054`, position in the code
the graph compares, motivated by `PV2_0053`.
