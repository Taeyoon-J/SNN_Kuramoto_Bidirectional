# PV2_0094 — the 32x32 model, trained and read

**Status:** completed. **Parent:** `PV2_0089` (failed), `PV2_0093` (the fix).
**Contract:** v2 — 32x32 cells on the native 240x320 masks, a rule verified identical to
contract v1 at grid 16.
**Change from `PV2_0089`:** one variable, `--membrane-threshold-mode population
--membrane-threshold-k 0.5`. Everything else is `PV2_0089`'s command.
**Regression row passed first:** contract v1 at 0.7239 / 0.7251 / 0.4951.

## The spike path is healthy

`spike_plv_collapse` 4.72 -> ... -> 1.90 -> 1.86 -> **1.860177** over 40 epochs, against
grid 16's 1.90 and `PV2_0089`'s dead 9.210340. `plv_mean` 0.7320 (grid 16: 0.753),
`spike_rate` 0.1827. 592 minutes on one GPU.

So the 1024-node model trains to the same health as the 256-node one once the threshold
is made relative. `PV2_0093` called this correctly and `PV2_0089`'s failure is fully
explained.

## Scores, validation 300

| readout | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| **connected components** | 0.6044 | **0.6661** | **0.5840** | **0.1268** |
| spectral k=10 | 0.5671 | 0.5662 | 0.3486 | 0.1793 |
| spectral k=14 | 0.6150 | 0.6179 | 0.4445 | 0.1823 |
| spectral k=20 | 0.6589 | 0.5933 | 0.5314 | 0.2575 |

## Against grid 16, at matched extent

The components readout predicts **0.1268** foreground here against **0.1290** at grid 16 —
a difference of 0.0022, so for once this comparison needs no calibration caveat.

| | grid 16 (v1) | grid 32 (v2) | change |
| --- | --- | --- | --- |
| `pred_fg` | 0.1290 | 0.1268 | -0.0022 |
| `patch_fg_ari` | 0.7239 | 0.6044 | **-0.1195** |
| `patch_foreground_iou` | 0.7251 | 0.6661 | -0.0590 |
| `patch_matched_object_iou` | 0.4951 | **0.5840** | **+0.0889** |

**A finer grid buys per-object accuracy and sells grouping coherence.** The figures show
why: the 32x32 predictions are visibly more fragmented — many small spurious groups, and
single true objects split across several predicted clusters. `patch_fg_ari` penalises
exactly that, while Hungarian-matched `patch_matched_object_iou` collects the benefit of
finer boundaries on the objects it does match.

## The second appearance of one signature

Two unrelated interventions now move the three metrics the same way:

| intervention | d fg_ari | d fg_iou | d obj_iou |
| --- | --- | --- | --- |
| signed coupling, balance 1.0 (`PV2_0092`) | +0.0325 | -0.2592 | **+0.0778** |
| the 32x32 grid (here) | -0.1195 | -0.0590 | **+0.0889** |

`patch_matched_object_iou` cannot be bought by over-predicting foreground — that is why
`ROUTE_MAP_0.9_v2.md`'s gate uses it as the honest witness — so both rises are real. What
both changes appear to do is sharpen object-level separation while loosening whatever
holds a single object's patches in one cluster. Whether that is one mechanism or two
coincidences is not established here, and it is the most concrete lead the project has
that is not a closed route.

## Limitations, stated

- **There is no Slot Attention baseline on contract v2.** This is a diagnostic, not a
  success claim. Building one means re-running the checkpoints at `--grid 32` and, per
  `PV2_0090`, necessarily with `--preprocess resize`.
- **v1 and v2 are different contracts.** 256 vs 1024 cells, foreground 11.5% vs 11.9%,
  6.20 vs 6.49 objects per image, mean purity 0.957 vs 0.978. The cross-contract table
  above is informative because the predicted extent matches and both use the same scoring
  functions on the same images, but it is not a within-contract comparison.
- Seed 0 only, validation only. Test has not been read on contract v2.

## Artifacts

`/work/USERS/tkim1/runs/G32_POP40/core.pt`, `G32_POP40.log`,
`runs/pv2_0094_results.txt`, `figures/g32_cc.png`, `figures/g32_sp14.png`,
`figures/g32_{cc,sp10,sp14,sp20}.npz`.

## Next

A contract-v2 Slot baseline with `--preprocess resize`, and seeds 1 and 2, before any
32x32 number is compared with anything.
