# PV2_0060 — the dial at three seeds; k=14 clears 0.75 on validation

**completed.** Parent `PV2_0059`. Evaluation only, three graph-frozen checkpoints
(`FZG_s0`, `BIM6_s1`, `FZG_s2`), validation 300, contract v1.

## Result

| k | fg_ari | std | fg_iou | obj_iou | pred_fg | >0.75 | beats Slot on all three |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.7381 | 0.0045 | 0.4413 | 0.6138 | 0.428 | no | yes |
| **14** | **0.7569** | **0.0006** | 0.3449 | 0.6480 | 0.542 | **yes** | **yes** |
| 16 | 0.7686 | 0.0030 | 0.2615 | 0.6664 | 0.653 | yes | yes |
| 20 | 0.7718 | 0.0017 | 0.1680 | 0.6913 | 0.808 | yes | yes |

Per seed at k=14: **0.7573 / 0.7562 / 0.7573**, a spread of 0.0011.

## Why k=14 was settled

It is the **first** point clearing 0.75, so it gives up the least foreground IoU, and
its seed spread is the tightest of the four. Larger k buys more fg_ari only by calling
still more patches foreground: at k=20 predicted foreground is 0.808 against a target
of 0.118, which is Slot Attention's own regime (~0.85), and there is no reason to walk
further into it.

## What it costs, stated with every number

Against our connected-component configuration (0.7250 / 0.7188 / 0.4953):

| metric | change |
| --- | --- |
| `patch_fg_ari` | **+0.0319** |
| `patch_matched_object_iou` | **+0.1527** |
| `patch_foreground_iou` | **-0.3739** |

**Two points on a dial, not a strict improvement.** Both configurations beat Slot
Attention on all three metrics; only k=14 clears 0.75 on fg_ari, and only connected
components keeps foreground IoU above 0.7.

k was selected on validation to clear fg_ari 0.75. The contract permits that — it
fixes thresholds on validation — but it is a selection and is recorded as one.
