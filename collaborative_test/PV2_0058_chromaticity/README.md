# PV2_0058 — chromaticity fails, and shows that fg_ari follows foreground extent

**failed as a feature route, but it produced the clue that led to `PV2_0059`-`0061`
and the goal.** Parent `PV2_0057`. Seed 0, validation 300, contract v1.

## What was tried

`PV2_0057` measured that chromaticity cuts the scale-invariant separation ratio by
46% (0.2030 -> 0.1093), targeting the within-object spread that `PV2_0036` valued at
+0.170. Two tensors were built and trained on rows 0-999, the images the baseline
used.

## Result

| arm | fg_ari | fg_iou | obj_iou | pred_fg | training loss |
| --- | --- | --- | --- | --- | --- |
| **bar, 8 channels** | **0.7239** | **0.7251** | **0.4951** | 0.129 | 32.49 -> **17.86** |
| chroma3, lr 1e-3 | 0.6143 | 0.2825 | 0.3153 | 0.228 | 28.15 -> **51.48** |
| chroma3, lr 3e-4 | **0.7380** | 0.1968 | 0.4064 | **0.509** | 28.70 -> 35.24 |
| chroma3, lr 1e-4 | 0.7305 | 0.1929 | 0.4005 | 0.530 | 28.85 -> 26.32 |
| chroma11 + graph-only | 0.3599 | 0.1362 | 0.0907 | 0.255 | 63.36 -> **17.71** |

**Closed.** chroma3 never converges at any of three learning rates, and chroma11
converges to the same loss as the baseline (17.71 against 17.86) yet scores 0.3599.

## Two things it settled, both of which mattered later

**My separation diagnostics do not predict the score in the middle range.** Both the
Euclidean ratio (`PV2_0057`) and a cosine margin computed afterwards say the chroma
tensors are *better* than the 8-channel gamma, which also has the lowest effective
rank of the three (1.68 against 1.98 and 2.64):

| tensor | eff. rank | cos within | cos between | ratio | fg_ari |
| --- | --- | --- | --- | --- | --- |
| wm10k_full | 1.68 | 0.0142 | 0.2483 | 0.0571 | **0.7239** |
| chroma3 | 1.98 | 0.0402 | 0.9524 | 0.0422 | 0.6143 |
| chroma11 | 2.64 | 0.0110 | 0.7416 | 0.0149 | 0.3599 |

The ordering is **inverted**. At the extreme the measures still hold — oracle codes
have within 0.0000 and score 0.9754 — but "the ratio improved 46%, so the score
should rise" does not follow. I had also used the ratio in `PV2_0051` to close data
scale; that conclusion stands because the score fell there too, but the ratio argument
is weaker than I presented it.

I also proposed, in the message reporting this, that chromaticity failed because
r+g+b = 1 makes it rank 2 and the graph uses a cosine. **The rank column refutes
that** — the winning tensor has the lowest rank.

**fg_ari follows predicted foreground extent.** chroma3 at lr 3e-4 reached fg_ari
0.7380, *above* the bar, while predicting foreground at 0.509 against a target of
0.118 and dropping foreground IoU to 0.1968. fg_ari is computed over the true
foreground patches only, so a wider predicted foreground can only help it. That is
Slot Attention's regime exactly (`PV2_0049`: pred_fg ~0.85, fg_ari 0.6635, fg_iou
0.1184), and recognising it as a dial rather than an accident is what `PV2_0059`
swept.
