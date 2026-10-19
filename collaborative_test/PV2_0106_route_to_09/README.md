# PV2_0106 — a route to 0.9 that keeps the core, measured rather than argued

**Status:** completed. **ORACLE DIAGNOSTIC — uses ground truth, never a score, reported
separately per the contract.** **Parent:** `PV2_0104`, `PV2_0105`.
**This is the first quantitative route above 0.9 this project has had.**

## The question

`PV2_0105` showed that perfect merging reaches 0.8242 and stops there, so reaching 0.9
needs something besides merging. `piece_quality` then measured what else is wrong with
the frozen 32x32 model's output:

| quantity | value | reading |
| --- | --- | --- |
| purity | 0.9124 | a predicted group is 91% inside one true object |
| **straddle** | **0.0313** | only 3.1% of groups span two objects — **merging can repair nearly everything** |
| **coverage** | **0.8726** | **12.7% of each true object is already called background** |
| splits | 1.77 | each object is divided into 1.77 groups — what merging repairs |
| lost | 0.0214 | 2.1% of objects are claimed by nothing at all |

So the pieces are clean and the other limit is **figure/ground, not grouping**.

## Result: the two oracles, separately and together

Three frozen seeds, validation 300, contract v2.

| | fg_ari | fg_iou | obj_iou | groups |
| --- | --- | --- | --- | --- |
| as is | 0.7109 | 0.7130 | 0.6588 | 16.50 |
| + merge | 0.8242 | 0.7130 | 0.7217 | 11.94 |
| + coverage | 0.8111 | 0.8108 | 0.7789 | 16.64 |
| **+ both** | **0.9329** | 0.8108 | **0.8390** | 12.09 |

| worth on fg_ari | |
| --- | --- |
| merging alone | +0.1133 |
| coverage alone | +0.1001 |
| **both** | **+0.2220** |

**Superadditive.** 0.1133 + 0.1001 = 0.2134 against a measured 0.2220, so fixing both
does more than fixing each. The residual to `PV2_0082`'s perfect segmenter (0.9756) is
0.0427.

## What this establishes for the goal

**A route above 0.9 exists that leaves the SNN + Kuramoto core untouched.** Both problems
are readout-side, both act on spikes the model already produces, and both have ceilings
above the goal:

1. **A hierarchical merge.** Form groups, then merge groups. Viable because straddling is
   only 3.1% — merging cannot repair a group that spans two objects, and almost none do.
2. **A foreground stage that stops losing 12.7% of each object.**

## This does not contradict `PV2_0091`

`PV2_0091` measured that our readout **dial** cannot reach 0.9: spectral `k` from 6 to 32
tops out at 0.7718 even at Slot Attention's own predicted extent. That swept **thresholds
and kernel widths** — points on one dial. What is proposed here are **stages**: an
operation that did not exist in the readout at all. The distinction is real and
`PV2_0091`'s conclusion stands as written — no setting of the existing readout reaches
0.9.

## Honest limits

- **These are oracle ceilings.** A real merger and a real foreground stage collect only
  part of each. Nothing here says how much.
- **`PV2_0069` is the standing risk.** The graph did not amplify its own estimate — oracle
  one-hot codes gave 0.9671 and the model's own groups gave 0.7199 — and a readout-side
  merger may inherit that fixed point.
- **Coverage is the harder half.** Predicted foreground is already **0.1369 against a
  target of ~0.121**, so the model spends more foreground than it needs and still covers
  only 87.3% of each object: the budget is in the wrong places. The coverage oracle does
  not merely relocate foreground, it adds it — foreground IoU rises 0.7130 to 0.8108 —
  so a real version must find object parts the model currently calls background.
- Validation only; contract v2's test split is unread and no Slot baseline exists on it.

## Next, in order

1. Build the merging stage. It is the cheaper half, the better understood, and its
   viability is supported by the 3.1% straddle rate. Target: a usable fraction of +0.1133.
2. Then the foreground stage, which is where the larger uncertainty is.
3. A contract-v2 Slot baseline with `--preprocess resize` before any external claim.
