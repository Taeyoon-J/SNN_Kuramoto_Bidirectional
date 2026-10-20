# PV2_0115 — a figure/ground term on the spikes

Status: **running** (8-epoch screen). No result yet. Registered here before results.
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
