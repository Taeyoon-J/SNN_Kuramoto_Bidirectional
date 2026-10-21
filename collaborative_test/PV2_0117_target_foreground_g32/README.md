# PV2_0117 — route A closed: the dial shatters the grouping instead of tightening it

Status: **completed — route A FAILS, on the correct grid, at three seeds**
Parent: `PV2_0116` (which tested the dial at grid 16, where there was nothing to tighten)
Control: `PV2_0104`'s `GC_w0`, `RSC_freeze`, `FRZ_s2`. Contract v2, grid 32, validation 300.

## Why this rerun existed

`PV2_0116` ran `--target-foreground` on `BIM6_s1` at grid 16, whose predicted foreground
is already 0.1290 against a true 0.1152 — almost nothing to tighten. `PV2_0114` measured
the over-prediction at **grid 32**: predicted foreground 0.1400, 22.2% of predicted patches
false, removing them worth **+0.2006** of foreground IoU against **+0.0428** for recovering
every miss. This runs the dial where that fault is.

## Result

| three-seed mean | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | ---: | ---: | ---: | ---: | ---: |
| **no dial** | **0.7110** | **0.7130** | **0.6588** | 16.50 | 0.1369 |
| target 0.13 | 0.4079 | 0.4689 | 0.4327 | 26.92 | 0.1411 |
| target 0.1152 | 0.4352 | 0.4935 | 0.4526 | 25.33 | 0.1371 |
| target 0.10 | 0.4717 | 0.5266 | 0.4796 | 23.36 | 0.1332 |

Every setting, on **every one of the three seeds**, collapses all three metrics. The best
of them is −0.2393 / −0.1864 / −0.1792 against the control.

The no-dial row reproduces `PV2_0104`'s 0.7110 / 0.7130 / 0.6588 exactly, so the harness is
sound and the failure is the dial's.

## What it actually does

`groups` goes from **16.50 to 23–27** while `pred_fg` barely moves — 0.1369 to 0.1332 at the
tightest setting. **The dial does not tighten the foreground; it shatters the partition.**
Calibrating the synchrony threshold per image to hit a fixed fraction changes which pairs
count as linked, and the grouping fragments long before the foreground shrinks.

That is a different failure from the one the route anticipated. `ROUTE_MAP_0.9_v3` asked
whether tightening would be *selective* — whether the patches it dropped would be
disproportionately false positives. The answer is that it never gets as far as dropping
patches: it destroys the grouping first.

**Route A is closed.**

## What remains open in `ROUTE_MAP_0.9_v3`

| route | state |
| --- | --- |
| `PV2_0115` γ-contrast loss | mechanism check **passed** at weight 5.0 (AUC 0.503 → 0.585); three metrics above control; a **candidate**, 8 epochs and one seed |
| A `--target-foreground` | **closed here** |
| B data scale | **closed** (`PV2_0116`) |
| C grid 64 | untested, ~96 GPU-hours |
| D `PV2_0091`'s owed sweep | untested |

Every readout-side lever this project has tried is now closed by measurement. The one
intervention that moved all three metrics in the right direction, and the only one that
reduced predicted foreground, is the γ-contrast **loss** — a change to training, not to the
readout, which is what `PV2_0114`'s arithmetic pointed at and what the goal's second clause
allows.
