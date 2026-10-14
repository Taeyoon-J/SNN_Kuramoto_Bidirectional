# PV2_0089 — the published configuration at a 32x32 grid, and why it died

**Status:** completed (failed). **Parent:** `PV2_0061`, enabled by `PV2_0088`.
**Contract:** v2 (32x32 cells on the native 240x320 masks; the rule reproduces contract v1
bit for bit at grid 16).
**Hypothesis:** the published configuration transfers to a finer grid with only the grid
changed.
**It does not.** The spike path dies during training.

## What was run

Arm A, the direct port: `--num-regions 1024 --spike-spatial-grid-size 32 --k 1024`,
`--geodesic-chunk 32`, everything else as the published configuration, seed 0, 40 epochs,
`--train-limit 1000` on `wm10k_full_grid32.pt`. The contract-v1 regression row passed
first: 0.7239 / 0.7251 / 0.4951.

A second arm (geodesic steps 4, spatial decay 0.175) was queued and deliberately stopped
before it started, to free the GPU for `PV2_0092`. See the correction below — that arm was
the control for one of the two hypotheses here, and stopping it cost a day.

## The failure

```
Epoch 01  loss=58.99   spike_plv_collapse=4.26
Epoch 05  loss=43.97                              <- still descending
Epoch 09  loss=76.34   spike_plv_collapse=7.29    <- collapsing
Epoch 17  loss=86.27   spike_plv_collapse=9.210340
Epoch 40  loss=86.04   spike_plv_collapse=9.210340
```

`9.210340 = log(10^4)` is the dead-membrane signature `PV2_0054` identified. The loss did
not converge at 86.04; the spike terms saturated. Grid 16 ends at 1.90.

## Diagnosis on the trained checkpoint

Validation, 100 images, contract v2.

| setting | fg_ari | fg_iou | obj_iou | pred_fg | spike_rate | plv_mean |
| --- | --- | --- | --- | --- | --- | --- |
| as trained | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0009** | **0.7101** |
| population threshold k=1.0 | -0.0008 | 0.0095 | 0.0103 | 0.0478 | 0.0743 | 0.7101 |
| population threshold k=0.5 | 0.0588 | 0.0508 | 0.0550 | 0.0828 | 0.1580 | 0.7101 |
| eval decay 0.175 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 0.7101 |
| decay 0.175 + population k=1.0 | -0.0008 | 0.0095 | 0.0103 | 0.0478 | 0.0743 | 0.7101 |
| spectral k=14 + population k=1.0 | 0.0485 | 0.1086 | 0.0271 | 0.4835 | 0.0743 | 0.7101 |

**`plv_mean` is 0.7101 in every row**, against grid 16's 0.753. The graph, the Kuramoto
coupling and the phase structure are healthy at 1024 oscillators. What fails is the
**theta -> membrane -> spike transduction** — exactly the link `CLAUDE.md` asks to be
verified, and the first time in this project it has broken while the phase side stayed
intact.

Lowering the threshold at evaluation revives spikes (0.0009 -> 0.1580) **without reviving
information** (fg_ari 0.0000 -> 0.0588, still chance). It cannot: once the membrane stopped
crossing threshold around epoch 17, the spike-PLV term had no gradient, so 23 of the 40
epochs trained nothing in the spike path. This is training damage, not a readout setting.

## A correction to this record's own diagnosis

The "eval decay 0.175" rows were **void, not evidence**. `spatial_rate` is an
`nn.Parameter` stored in the checkpoint, so `--graph-spatial-decay` at evaluation sets an
initial value that `load_state_dict` immediately overwrites. The flag was a no-op, which is
why those rows are **bit-identical** to the 0.35 rows rather than merely close. Hypothesis
B — that `--graph-spatial-decay` is measured in grid units and 0.35 is therefore twice the
intended decay at a grid where one step covers half the image — is **untested**, not
refuted. It can only be set at training time, which is what `PV2_0093` does.

## What this establishes

1. **The grid is not the obstacle to the dynamics.** 1024 oscillators form phase structure
   at `plv_mean` 0.7101, and `PV2_0088` showed the memory cost is 4.72 GB rather than
   68.7 GB once the relaxation is chunked.
2. **The transduction does not transfer.** Hyperparameters tuned for 256 oscillators —
   `--membrane-vth 0.06` absolute, `--graph-spatial-decay 0.35` in grid units — do not
   carry to 1024, and the failure is silent in the loss until epoch 17.
3. **No 32x32 number can be compared with anything yet.** There is no contract-v2 Slot
   baseline either, and per `PV2_0090` any such baseline must use `--preprocess resize`.

## Next

`PV2_0093` runs three 8-epoch arms, one variable each — the control, a population
threshold, and the spatial decay set at training time — and reads the collapse trajectory
rather than a score, since 8 epochs is enough to tell a surviving arm from a dying one
(the original was at 4.26 by epoch 1 and 7.29 by epoch 9).

## Artifacts

`/work/USERS/tkim1/runs/G32_A/core.pt` (12.6 MB, N=1024), `G32_A.log`,
`runs/pv2_0089_results.txt`, `runs/pv2_0089_diag.txt`.
Contract v2 assets: `targets_v2_grid32.pt` (8000 x 32 x 32),
`wm10k_full_grid32.pt` (10000 x 8 x 1024).
