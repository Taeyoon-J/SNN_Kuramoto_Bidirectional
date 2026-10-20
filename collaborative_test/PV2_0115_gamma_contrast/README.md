# PV2_0115 — a figure/ground term on the spikes

Status: **completed** (8-epoch screen). Mechanism check **passed at weight 5.0**; the
three-metric gate cannot be evaluated in this regime (see Result).
Parent: `PV2_0114`. Contract v1, grid 16, three arms: control / weight 1.0 / weight 5.0.
Checkpoints `GCON_ctrl`, `GCON_w1`, `GCON_w5` — **not** `GC_*`, which belong to `PV2_0097`.

## Why

Every PLV term in `loss_function.py` shapes **partition structure** — bimodality, balance,
collapse, participation ratio — and none of them says which block is foreground. `PV2_0114`
measured the consequence: internal spike coherence separates a real group from a spurious
one at **AUC 0.500**, mean activity at **0.494**.

The arithmetic says those terms cannot say it. `plv_group_balance_loss` targets the mean
off-diagonal PLV, a mean over **pairs**, so a coherent group of n nodes counts **n²**. With
~900 background nodes of 1024 the background alone supplies **0.7726** of the 0.867 target
and a 3-patch object is worth **1/90,019** of it.

The cue exists in the input: a patch's γ deviation from its own 5×5 neighbourhood separates
true foreground at **AUC 0.947**, label-free, over 200 images. It is destroyed on the way to
the spikes the readout reads.

And it points at the larger half. `PV2_0114`: removing the false positives is worth
**+0.2006** of foreground IoU, recovering every missed patch **+0.0428**.

## What was added

`spike_gamma_contrast_loss` — the negative per-image Pearson correlation between mean spike
activity after settle and that γ deviation. Scale-free, so it constrains the ordering rather
than the level. Reads no ground truth: γ is the model's own input. Nothing in the
SNN + Kuramoto bidirectional core changes; this is a loss, which `CLAUDE.md` places first.

## Gate and falsifier, registered before the arms trained

- **Mechanism first, and cheaper:** activity-vs-foreground AUC must rise from 0.494
  (`auc_check.py`). If it does not, the term did not do what it is documented to do and **no
  patch metric from this test is interpreted**.
- **Then:** all three metrics above control at predicted foreground ≤ 0.20, with
  `patch_matched_object_iou` specifically up.
- **Falsifier:** AUC rises and the three metrics do not → figure/ground in the spikes was not
  what limited the readout, and this line closes with the other four.

## Verification done before any result

| check | outcome |
| --- | --- |
| behaviour | activity aligned with the deviation −0.6936, random −0.0179, anti-aligned +0.6936 |
| gradient | finite and non-zero, both the tuple and the int grid form |
| both γ layouts | `[B,C,N]` and `[B,N,C]` agree to 1e-6 |
| end-to-end training, CPU | runs and saves a checkpoint |
| correlation → AUC | 0.3102 in the log implies ≈ 0.658, against 0.494 now and a 0.947 ceiling (`corr_to_auc.py`) |

## Three faults found and fixed before they cost a result

1. **Arm names collided with `PV2_0097`.** `GC_w0`, `GC_w5`, `GC_w20` already existed, and
   `GC_w0` is `PV2_0104`'s seed-0 checkpoint referenced by four diagnostics and by
   `PV2_0112`'s sweep. Stopped at epoch 4 of the control, before `GC_w5` was overwritten;
   it is intact at its original timestamp. Renamed to `GCON_*`.
2. **`core.spike_spatial_grid_size` is an int while training and a pair while evaluating.**
   A unit test that passed `(G, G)` directly never met the int, and the first run lost **both
   treatment arms** to a `TypeError` while the control trained through.
3. **A dead attribute lookup.** The wiring tried `core.last_spikes` first; that attribute
   does not exist, so the term only ever worked through its fallback. Removed.

## A zero that is not a bug

An end-to-end CPU run at 64 oscillators reported `gamma_contrast=0.000000` with activity
std exactly 0, while a standalone call on the same input gave 0.1386. The cause is in the
record already: `spike_plv_collapse` read **9.210340**, which `PV2_0094` identifies as
log(10⁴), the dead spike path. `--membrane-vth 0.06` is an **absolute** threshold tuned for
256 oscillators and does not transfer to 64. Adding `--membrane-threshold-mode population`,
which `PV2_0094` found keeps the path alive, took activity std from **0.0 to 0.19**.

So the term is sound and the zero was a degenerate configuration. It is worth recording
because the same zero would be indistinguishable from an inert loss to anyone reading only
the log.

## Limitations

- The **8-epoch screen** is forced by compute: user `ziquanw` occupies all four GPUs and the
  identical config fell from 40 epochs in 23 minutes to 6 in 39. The screen can answer the
  mechanism question only. Its control already breaches the calibration gate at `pred_fg`
  0.3699 against a published 0.1290, so **no absolute number from it may be compared to the
  published line** — relative to control only.
- Seed 0 only. A promotion needs seeds 1 and 2 at full length.

## Result

| 8-epoch screen, seed 0, grid 16 | fg_ari | fg_iou | obj_iou | pred_fg | `gamma_contrast` |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (weight 0) | 0.6945 | 0.3652 | 0.4666 | 0.3699 | — |
| weight 1.0 | 0.6580 | 0.3605 | 0.4567 | 0.3530 | −0.3194 |
| **weight 5.0** | **0.7015** | **0.4544** | **0.4851** | **0.3122** | −0.5013 |

**Mechanism check, which the gate put first.** Activity-vs-foreground AUC over 60
validation images:

| | AUC | mean activity, FG | BG | separation |
| --- | ---: | ---: | ---: | ---: |
| control | 0.503 | 0.4960 | 0.4989 | **−0.0029** |
| weight 1.0 | 0.485 | 0.4974 | 0.5028 | −0.0055 |
| **weight 5.0** | **0.585** | 0.5037 | 0.4986 | **+0.0051** |

**Weight 5.0 passes; weight 1.0 fails** and its scores are therefore not interpreted, as
registered. In the control the foreground is *less* active than the background — the sign
is backwards — and the term flips it.

Weight 5.0 moves all three metrics above control and takes predicted foreground **down**,
0.3699 → 0.3122. That direction matters: `PV2_0114` measured that removing false
foreground is worth **+0.2006** of foreground IoU against **+0.0428** for recovering every
miss, and every stage either branch had built until now pushed foreground *up*. This is
the first intervention here that pushes the other way, and foreground IoU is where it
gains most (+0.0892, 24% relative).

## Why this is a candidate and not a result

1. **The calibration gate cannot be evaluated.** At eight epochs the control itself sits at
   `pred_fg` 0.3699 against a published 0.1290, so every arm breaches the 0.20 gate and
   only the comparison to control is readable. This was recorded before the run.
2. **Weight 1.0 is worse than control on all three** and fails the mechanism check. Zero →
   1 → 5 is not monotone, and two points cannot separate a dose-response from noise.
3. **Seed 0 only, eight epochs.** A promotion needs seeds 1 and 2 at forty.

## A reference that was wrong, and how far it threw the prediction

The run registered "γ deviation separates foreground at AUC 0.947" as its reference. That
0.947 was measured at **grid 32**; these arms are **grid 16**, where cells are 15×20 px
against 7.5×10 and the local deviation is blurred. So 0.947 was never the ceiling for
these arms.

`corr_to_auc.py` built on it and predicted AUC ≈ 0.751 at the correlation weight 5.0
reached (0.501). The measured value is **0.585**. The model was not wrong in shape — more
correlation did buy more AUC — but it was anchored to a cue measured on a different grid,
and overshot by 0.166. The grid-16 cue should be measured before that prediction is reused.
