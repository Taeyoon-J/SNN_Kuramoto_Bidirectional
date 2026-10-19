# PV2_0104 — the freeze closes the spread at 1024 nodes, and 32x32 is a different operating point, not a better model

**Status:** completed. **Parent:** `PV2_0103`. **Contract:** v2.
**Hypothesis:** `PV2_0044`'s graph freeze — "reduced seed variance; it did not improve a
seed" (`PV2_0081`) — closes the 32x32 seed spread the way it closed grid 16's.
**Confirmed, and the resulting mean does not beat the published configuration.**

## Result, three seeds sharing seed 0's graph, validation 300

| seed | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7699 | 0.7531 | 0.7017 | 15.84 | 0.1512 |
| 1 | 0.6777 | 0.6895 | 0.6357 | 16.81 | 0.1306 |
| 2 | 0.6853 | 0.6963 | 0.6390 | 16.86 | 0.1288 |
| **mean** | **0.7110 +- 0.0512** | **0.7130 +- 0.0349** | **0.6588 +- 0.0372** | 16.50 | 0.1369 |

**The spread closes: fg_ari standard deviation 0.2599 unfrozen (`PV2_0102`) -> 0.0512
frozen.** A fivefold reduction, which is exactly what `PV2_0044` did at 256 nodes and
exactly what `PV2_0081` said it does — it constrains the solution rather than improving a
seed. Seed 1, which collapsed to 0.2916 unfrozen, lands at 0.6777.

## Against the published configuration

| metric | 32x32 frozen | 16x16 published | change |
| --- | --- | --- | --- |
| `patch_fg_ari` | 0.7110 +- 0.0512 | **0.7239** | -0.0129 |
| `patch_foreground_iou` | 0.7130 +- 0.0349 | **0.7251** | -0.0121 |
| `patch_matched_object_iou` | **0.6588 +- 0.0372** | 0.4951 | **+0.1637 (+33%)** |

**Not a win.** Both losses sit inside their own standard deviations — 0.0129 against
0.0512 and 0.0121 against 0.0349 — so on fg_ari and foreground IoU these are ties. Only
the per-object gain is large relative to its spread, at 4.4 standard deviations.

So 32x32 is **a different operating point, not a better model**: it trades about a
hundredth on figure/ground and grouping for a third more per-object accuracy, inside the
calibrated regime at predicted foreground 0.1369 against a target of ~0.121.

The estimate made before the run — 0.718 / 0.686 / 0.635 — was close to the measurement
and reached the same conclusion, which is recorded because the prediction was made in
advance rather than after.

## What the 32x32 line established

Three hyperparameters tuned at 256 oscillators failed to transfer to 1024, and three did
not fail:

| parameter | transfers? |
| --- | --- |
| membrane threshold, absolute (`PV2_0093`) | **no** — the spike path dies at epoch 17 |
| epoch budget, 40 (`PV2_0101`) | **no** — 8 beats 40 by +0.1406 at seed 0 |
| seed robustness (`PV2_0102`, here) | **no** — needs the freeze, as at grid 16 |
| spatial decay in grid units (`PV2_0093` B) | yes — halving it is worse |
| graph top-k as an absolute count (`PV2_0096`) | yes — 8x denser moves +0.0019 |
| the spatial kernel's width (`PV2_0095`) | partly — sigma 1.0 -> 4.0 is worth +0.0249 |

Half the "it is in grid units so it will not transfer" guesses were wrong. The pattern
that did hold is narrower: **quantities with an absolute scale** — a membrane threshold
in volts, a number of epochs, the variance of a random initialisation — did not transfer,
while **quantities defined relative to the graph** mostly did.

## What this is not

**It is not a route to 0.9.** The three-seed mean is 0.7110, below the published 0.7239,
and `ROUTE_MAP_0.9_v2.md` records that the best anything has scored on this data is
0.8102. The finer grid buys per-object accuracy; it does not move fg_ari.

## Limitations

All three seeds share seed 0's graph, so they are not three independent runs — the same
structure the published three seeds have at grid 16 and disclosed the same way in
`STATUS.md`. Validation only; contract v2's test split is unread. **No Slot Attention
baseline exists on contract v2**, so none of this is a claim against anything external;
building one needs `--preprocess resize` per `PV2_0090`. The 8-epoch budget is inherited
rather than selected and no sweep has located the peak.
