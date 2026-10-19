# PV2_0098 — training images, 1000 against 6000

Status: **interrupted** — both arms trained, neither could be scored
Parent: the published 16x16 line (`PV2_0061` / `PV2_0076`), contract v1, split v1
Comparison: published reference 0.7239 / 0.7251 / 0.4951 at pred_fg 0.1290

## Hypothesis

Every result in the published line was trained on `--train-limit 1000`. If the model is
data-starved rather than capacity-starved, six times the images should move the three patch
metrics on its own, with no change to the architecture or the losses. One variable: the
`--train-limit`, 1000 against 6000, same 40-epoch budget, same seed.

## What came back

Both arms trained to completion. Neither was scored: the evaluation step calls
`evaluate_model.py` on `/export_home`, which faulted mid-session (see `STATUS.md`), so both
rows read `FAILED` and **no patch metric exists for this test**.

| arm | wall | loss, first → last | final `spike_plv_collapse` |
| --- | --- | --- | --- |
| DS_1000 | 154 min | 23.457814 → 16.673826 | 2.154499 |
| DS_6000 | 557 min | 22.035025 → **14.429044** | **1.954825** |

Six times the data gives a lower final loss (−2.24) and a lower phase-collapse term
(−0.200, where lower is less collapsed). Both are training-side diagnostics. **Neither is
one of the three metrics and neither decides anything** — the whole point of the calibration
gate is that a better loss has repeatedly failed to convert into a better mask here.

## Why it is not yet an answer

The checkpoints are safe on `/work/USERS/tkim1/runs/DS_1000/` and `DS_6000/`. The test
resumes by scoring them, nothing more — no retraining is needed. Until then the hypothesis
is untested, not supported: a lower loss in this project has five times already come with a
flat or worse `patch_matched_object_iou`.

## Next

Score both checkpoints the moment `/export_home` returns, with the regression row on
`BIM6_s1` passing first. If 6000 images move all three metrics, retrain at seeds 1 and 2;
if it moves only `fg_ari`, it joins the list of mechanism-level gains that did not convert.
