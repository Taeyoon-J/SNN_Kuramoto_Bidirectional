# PV2_0116 — routes A and B, both answered, both negative

Status: **completed**. Evaluation only, no training. Grid 16, contract v1, validation 300.
Parent: `ROUTE_MAP_0.9_v3.md` routes A and B.

## Route A — `--target-foreground`, and a setup error of mine

The dial calibrates the synchrony threshold per image to a fixed fraction, selected on
validation. It had never appeared in a command in this repository.

| `BIM6_s1`, grid 16 | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | ---: | ---: | ---: | ---: | ---: |
| **no dial** | **0.7239** | **0.7251** | **0.4951** | 5.46 | 0.1290 |
| target 0.16 | 0.5693 | 0.5549 | 0.4400 | 6.30 | 0.1516 |
| target 0.14 | 0.5952 | 0.6030 | 0.4506 | 6.13 | 0.1399 |
| target 0.121 | 0.6203 | 0.6394 | 0.4588 | 5.94 | 0.1309 |
| target 0.11 | 0.6243 | 0.6506 | 0.4614 | 5.88 | 0.1284 |
| target 0.10 | 0.6362 | 0.6645 | 0.4645 | 5.81 | 0.1252 |

Every setting is worse than the baseline on all three. **But this tests the dial against a
checkpoint that does not have the fault the dial is for.** `PV2_0114` measured the
over-prediction at **grid 32** on `GC_w0`: predicted foreground 0.1400 against a true
0.1152, 22.2% of predicted patches false. `BIM6_s1` at grid 16 already sits at **0.1290**,
so there was almost nothing to tighten, and what the dial changes here is the *mechanism* —
a per-image calibration replacing a fixed threshold — not the amount of foreground.

The first row confirms the harness: the no-dial baseline reproduces the published
`0.7239 / 0.7251 / 0.4951` at `pred_fg` 0.1290 exactly.

`PV2_0117` runs the dial at grid 32 on all three of `PV2_0104`'s seeds, where the
over-prediction actually is. **Route A is not closed by this test.**

## Route B — six times the training data does not convert

`PV2_0098` trained both arms and never scored them. Scored here:

| | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | ---: | ---: | ---: | ---: | ---: |
| DS_1000 | 0.3374 | 0.3246 | 0.2383 | 3.35 | 0.1272 |
| **DS_6000** | **0.3142** | 0.4359 | **0.1880** | 2.70 | 0.1293 |

Six times the data **lowers** FG-ARI and matched-object IoU. It raises foreground IoU only.

`PV2_0098` recorded the training side: loss 16.673826 → **14.429044** and
`spike_plv_collapse` 2.154499 → **1.954825**, both clearly better with more data. They did
not convert. **That is the sixth time in this project that a better loss has failed to
reach the mask**, and it is the reason the calibration gate exists.

Both arms also sit far below the published 0.7239, so this training line differs from
`BIM6`'s in more than the data volume; the comparison that holds is DS_6000 against
DS_1000, not either against the published model.

**Route B is closed.**

## Where this leaves `ROUTE_MAP_0.9_v3`

| route | state after this test |
| --- | --- |
| `PV2_0115` γ-contrast loss | mechanism check **passed at weight 5.0**; three-metric gate not evaluable at 8 epochs |
| A `--target-foreground` | **not closed** — tested on the wrong grid; `PV2_0117` retests at grid 32 |
| B data scale | **closed** — more data lowers two of three |
| C grid 64 | untested, ~96 GPU-hours |
| D `PV2_0091`'s owed sweep | untested |
