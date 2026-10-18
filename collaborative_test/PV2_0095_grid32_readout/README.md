# PV2_0095 — the 32x32 readout knobs, and what they do not fix

**Status:** completed. **Parent:** `PV2_0094`. **Contract:** v2.
**Hypothesis:** `PV2_0094`'s fragmentation comes from two readout knobs measured in grid
units — `--spatial-sigma 1.0` covers half the image distance at grid 32 that it did at
grid 16, and `--min-group-size 2` is a patch count while an object covers ~4x more
patches. Both are pure evaluation-time arguments, so this costs minutes.
**Outcome: half right, and the half that was wrong was wrong in an informative way.**

## Results, validation 300, on `G32_POP40`

Column order is the evaluator's own: fg_ari, foreground_iou, matched_object_iou,
**groups_per_image**, predicted_foreground.

| setting | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| sigma 1.0, min 2 (`PV2_0094`) | 0.6044 | 0.6661 | 0.5840 | 19.45 | 0.1268 |
| sigma 1.5, min 2 | 0.6170 | 0.6725 | 0.5861 | 19.67 | 0.1295 |
| sigma 2.0, min 2 | 0.6205 | 0.6566 | 0.5875 | 20.49 | 0.1336 |
| sigma 3.0, min 2 | 0.6255 | 0.6603 | 0.5872 | 18.97 | 0.1345 |
| **sigma 4.0, min 2** | **0.6293** | **0.7091** | **0.5885** | 16.76 | **0.1276** |
| sigma 2.0, min 4 | 0.5747 | 0.6691 | 0.5009 | 8.02 | 0.1061 |
| sigma 2.0, min 8 | 0.4848 | 0.6158 | 0.3583 | 3.66 | 0.0849 |
| sigma 2.0, min 12 | 0.4332 | 0.5591 | 0.2920 | 2.74 | 0.0766 |
| sigma 3.0, min 8 | 0.4928 | 0.6083 | 0.3606 | 3.90 | 0.0875 |
| **grid 16 reference** | **0.7239** | **0.7251** | 0.4951 | — | 0.1290 |

## 1. Sigma helps, monotonically, and not enough

1.0 -> 4.0 gives +0.0249 fg_ari and +0.0430 foreground IoU, bringing the latter to
**0.7091 against grid 16's 0.7251** — within 0.016. `obj_iou` is flat at ~0.588
throughout. So the kernel width was genuinely mis-scaled and fixing it costs nothing,
but fg_ari stays **0.0946** behind.

## 2. `min-group-size` is the opposite of what I argued, and the error is instructive

The reasoning was "an object covers ~4x more patches, so the floor should scale". Every
raise makes everything worse: at min 8 fg_ari falls to 0.4848 and `obj_iou` to 0.3583,
and groups collapse from 20 to 3.7 against 6.49 true objects. **The predicted groups are
already too small**, so a floor deletes real objects rather than noise. The knob was
never admitting fragments; it was the only thing keeping small true objects alive.

## 3. The number that matters

**`groups_per_image` is 19.5 against 6.49 true objects per image — three times too many,
and sigma 4.0 only brings it to 16.8.**

That is the fragmentation, measured rather than read off a figure, and it is the whole of
the remaining fg_ari gap. Stated cleanly, at sigma 4.0 and matched extent (0.1276 against
0.1290):

| | change vs grid 16 |
| --- | --- |
| `patch_foreground_iou` | **-0.0160** — a tie |
| `patch_matched_object_iou` | **+0.0934** |
| `patch_fg_ari` | -0.0946, entirely over-segmentation |

## What this says about the architecture

Grid 16's error mode is **coherent object merges** — `PV2_0069` showed the graph is a
fixed point at the accuracy it starts from, and `ROUTE_MAP_0.9.md`'s finding 2 says its
own errors are merges. Grid 32's error mode is **the opposite**: it splits single objects
into three clusters on average.

The same model, same objective, same readout family, at two grid resolutions, fails in
two opposite directions. Whatever sets that balance is not the spatial kernel and is not
the group-size floor, because both were swept here.

## Next

`PV2_0096` tests the third parameter in the wrong units. `--graph-top-k` is 32 in both
configurations: 12.5% of the nodes at grid 16 and **3.1%** at grid 32, so the coupling
graph is four times sparser relative to the node count and groups may simply be unable to
link across an object. The graph is rebuilt from gamma at evaluation, so it is testable on
this checkpoint — with the caveat that the checkpoint was trained at top_k 32, making a
gain strong evidence and a loss ambiguous.
