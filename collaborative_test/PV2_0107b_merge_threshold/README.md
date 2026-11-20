# PV2_0107b — the merge threshold on its real scale

Status: **completed** (one cell failed on a filesystem fault; see Limitations)
Parent: `PV2_0107` (merged nothing at threshold 0.05) · Control: `PV2_0104` (grid 32, graph frozen)
Ceiling it was chasing: `PV2_0106` oracle, 0.8242 fg_ari / 0.7217 obj_iou at 11.94 groups
Data: `wm10k_full_grid32.pt`, targets `targets_v2_grid32.pt`, split manifest v1, contract **v2**
Evaluated on **validation**, 300 images — a tuning split, not a test read.

## Hypothesis

`PV2_0106`'s oracle decomposition said a second, agglomerative readout stage is worth
+0.1133 fg_ari on its own. `PV2_0107` tried it at threshold 0.05 and merged nothing,
because the affinity is spike correlation **times** a spatial Gaussian and its real scale
is ~1e-4: the measured inter-group maximum was 0.00083 against 0.68401 within groups.
This run sweeps tau across that scale at three seeds.

Success criterion, registered before the run: all three patch metrics above the
same-grid control, at predicted foreground at or below 0.20.

## Result — the stage FAILS the criterion

| | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| control, merge off (3 seeds) | 0.7110 ± 0.0512 | 0.7130 ± 0.0349 | 0.6588 ± 0.0372 | 16.50 | 0.1369 |
| merge, tau 0.0001 (3 seeds) | **0.7592 ± 0.0365** | 0.7130 ± 0.0349 | **0.6303 ± 0.0346** | 10.96 | 0.1369 |
| absolute change | +0.0483 | +0.0000 | **−0.0285** | | |
| relative change | +6.79% | +0.00% | **−4.33%** | | |

fg_ari rises at **every** seed (+0.0313, +0.0577, +0.0556) and matched-object IoU falls at
**every** seed (−0.0315, −0.0263, −0.0278). Foreground IoU is unchanged to four decimals at
every seed, which is the arithmetic working: merging repartitions the foreground set and
cannot change it, so `pred_fg` is identical and the calibration gate is passed trivially.

This is a trade, not an improvement. The rule is all three metrics, so it does not promote.

## Why — the right number of merges, the wrong pairs

tau is a clean dial on the trade: as it falls, groups fall (13.37 → 9.98) and fg_ari rises
while obj_iou drops. At tau 0.0002 the group count is **11.92**, which is the oracle's
11.94 almost exactly — and obj_iou there is still 0.6491, below the control's 0.6588.

So at the oracle's own operating point, matched-object IoU is still worse. The stage is
not merging too much or too little; **it is merging the wrong pairs.** The oracle, which
chose pairs with the truth, gained on both columns from the same number of merges.

That localises the next move: not the threshold, and not the affinity, but the **linkage**.
Mean inter-group affinity lets a large group absorb a small neighbouring one cheaply,
which is exactly the merge that raises fg_ari (fewer labels over true-foreground patches)
while destroying a matched object. Candidates, one variable each: complete linkage; mean
normalised by group size; a mutual-nearest-neighbour requirement before any merge.

## Two errors in this run's own summary, corrected here

1. **The script compared to the wrong control.** It printed
   `against the published 16x16: +0.0353 / −0.0121 / +0.1352`, comparing a grid-32 merged
   result to the grid-16 published model. That mixes a grid change with the stage being
   tested and turns a −0.0285 obj_iou regression into a +0.1352 gain. The control for a
   readout stage is the same checkpoints with the stage off — `PV2_0104`.
2. **It ranked rows computed over different numbers of seeds.** The two best-looking rows,
   tau 0.00002 ("75% of ceiling") and tau 0.0005 ("69%"), are **seed 0 only**. Seed 0 is the
   strongest seed in this family — its control fg_ari is 0.7699 against a 3-seed mean of
   0.7110 — so those rows are high because of the seed, not the threshold.

This is the sixth time an automated verdict in this project read fg_ari alone and reached
the opposite conclusion from the three metrics together.

## Limitations

- Validation, 300 images. No test read was taken and none is warranted: the stage failed.
- `seed 2, tau 0.0002` did not run. `/export_home` faulted mid-sweep with `OSError: [Errno 5]`
  and the import of `snn_kuramoto_bidirectional` failed. That cell is the only n=2 row; it
  does not affect the tau 0.0001 verdict, which has all three seeds.
- The oracle ceiling it is measured against (`PV2_0106`) is an oracle result and is reported
  as a diagnostic only, never as a baseline.

## Next

`PV2_0111`: linkage, one variable. Complete linkage against mean linkage at the tau that
matched the oracle's group count (0.0002), three seeds, same checkpoints. If matched-object
IoU still falls, the affinity itself is what cannot separate the pairs and the
agglomerative route closes.
