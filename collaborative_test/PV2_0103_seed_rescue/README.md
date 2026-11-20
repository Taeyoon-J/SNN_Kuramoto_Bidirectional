# PV2_0103 — seed 1 is rescued by both fixes, and the automated verdict picked the wrong one

**Status:** completed. **Parent:** `PV2_0102`. **Contract:** v2.
**Hypotheses, both from this project's own record, tested on the one seed that fails:**
**A** the fragility is variance, and `PV2_0044`'s graph freeze — which `PV2_0081`
characterised as "reduced seed variance; it did not improve a seed" — has never been
applied at 1024 nodes. **B** 8 epochs was inherited from `PV2_0093`'s screening budget and
seed 1 simply has not converged.

## Result, validation 300

| | fg_ari | fg_iou | obj_iou | groups | pred_fg | wall | collapse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed 1 as it stands, 8 ep | 0.2916 | 0.0947 | 0.2326 | 11.44 | **0.3101** | 116 min | 3.70 |
| **A: graph frozen, 8 ep** | 0.6777 | **0.6895** | **0.6357** | 16.81 | 0.1306 | **54 min** | 2.22 |
| B: 16 epochs, no freeze | **0.6867** | 0.6008 | 0.4887 | 8.56 | 0.1087 | 341 min | 2.87 |
| seed 0 (works), for scale | 0.7699 | 0.7531 | 0.7017 | 15.84 | 0.1512 | | 2.58 |
| seed 2 (works), for scale | 0.7068 | 0.6151 | 0.5677 | 11.09 | 0.1140 | | 3.08 |
| 16x16 published | 0.7239 | 0.7251 | 0.4951 | | 0.1290 | | |

**Both rescue it**, from 0.2916 to 0.6777 and 0.6867. The fragility is real and fixable,
and it is not one thing: a frozen graph and a longer budget both work.

## The automated verdict was wrong, for the fifth time

The script printed "SEED 1 IS RESCUED BY: B: 16 epochs" because B's fg_ari is higher by
0.0090. On all three metrics A is clearly better:

| | A minus B |
| --- | --- |
| `patch_fg_ari` | -0.0090 |
| `patch_foreground_iou` | **+0.0887** |
| `patch_matched_object_iou` | **+0.1470** |

`patch_matched_object_iou` is the metric over-prediction cannot buy, which is why
`ROUTE_MAP_0.9_v2.md`'s gate uses it as its witness. A also costs **54 minutes against
341** — a sixth of the compute — and lands closer to the target foreground, 0.1306
against B's 0.1087 and a target of ~0.121.

This project has now had five automated verdicts that read fg_ari alone and were wrong:
`PV2_0062`, `PV2_0067`, `PV2_0078`, `PV2_0097`'s weight-20 arm, and this. The scripts keep
printing a one-metric conclusion even when the gate is written in the same file.

## What the freeze does and does not promise

The estimate, using seed 2's unfrozen score as a stand-in for its frozen one:

| | 32x32 frozen (est.) | 16x16 published |
| --- | --- | --- |
| `patch_fg_ari` | ~0.718 | **0.7239** |
| `patch_foreground_iou` | ~0.686 | **0.7251** |
| `patch_matched_object_iou` | **~0.635** | 0.4951 |

A tie on fg_ari, a loss on foreground IoU, and a large win on per-object IoU. **Not a
clean win** — a different operating point. `PV2_0104` replaces the stand-in with a
measurement.

## What is now established about the 32x32 line

1. The 40-epoch budget is wrong (`PV2_0101`): 8 epochs beats it at seeds 0 and 2.
2. 8 epochs alone is unstable (`PV2_0102`): three-seed mean 0.5894 +- 0.2599.
3. The instability is the one `PV2_0012` recorded at grid 16, and the freeze that fixed
   it there fixes it here too — at a sixth of the cost of simply training longer.

## Limitations

Seed 1 only; `PV2_0104` runs seed 2 with the freeze to complete the mean. The frozen
graph comes from seed 0's checkpoint, so all seeds share one graph — the same structure
the published three seeds have at grid 16, and disclosed the same way. Validation only;
contract v2's test split is unread and no Slot baseline exists on it.
