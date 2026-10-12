# PV2_0061 — test split, read once: fg_ari 0.7533, and Slot Attention beaten on all three

**completed. The 0.75 target is met on the test split at three seeds, and the
standing goal — beat Slot Attention on all three patch metrics as 3-seed means — is
met with it.** Parent `PV2_0060`. Full test split, IDs 7000-7999, **all 1000
images**, contract v1, split manifest v1.

## Headline

| metric | ours, spectral k=14 | matched Slot Attention | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7533** +- 0.0015 | 0.6663 +- 0.1108 | **+0.0870** | **+13.1%** |
| `patch_foreground_iou` | **0.3358** +- 0.0215 | 0.1177 +- 0.0101 | **+0.2180** | **+185.2%** |
| `patch_matched_object_iou` | **0.6549** +- 0.0042 | 0.0871 +- 0.0040 | **+0.5678** | **+651.6%** |

Per seed, ours: fg_ari **0.7547 / 0.7536 / 0.7517**; fg_iou 0.3478 / 0.3109 / 0.3486;
obj_iou 0.6517 / 0.6597 / 0.6534.
Per seed, Slot: fg_ari 0.7265 / 0.7340 / 0.5385; fg_iou 0.1126 / 0.1112 / 0.1294;
obj_iou 0.0915 / 0.0837 / 0.0861.

**fg_ari 0.7533 > 0.75.** Previous test figure (`PV2_0008`, connected components):
0.6075 / 0.6755 / 0.4298, so fg_ari is **+0.1458** on the last test read.

## The other configuration, on the same test images

| metric | ours, connected components | vs k=14 |
| --- | --- | --- |
| `patch_fg_ari` | 0.7258 +- 0.0011 | -0.0275 |
| `patch_foreground_iou` | **0.7124** +- 0.0075 | **+0.3767** |
| `patch_matched_object_iou` | 0.5029 +- 0.0014 | -0.1520 |

**Both configurations beat Slot Attention on all three metrics.** Only k=14 clears
0.75 on fg_ari; only connected components keeps foreground IoU above 0.7. They are two
points on one dial and both are reported, because quoting only the one that clears the
target would hide what it costs.

## The score comes from spikes

Required by the contract, and checked rather than assumed. The spectral readout
clusters `_spike_affinity(spikes, components)` — the affinity is built from the spike
trains, not from theta or PLV, and `--readout` is left at its default `spikes`.
`PV2_0043` measured that **binary spike timing alone carries that affinity at
r = 0.9822**, and 0.9808 / 0.9807 on these graph-frozen checkpoints, with always-on
units contributing <= 0.5%.

`--spectral-k 14` is a fixed group count. It is **not** the true object count:
`--spectral-oracle-k` exists for that and is not used here. No mask, object count or
ground-truth label enters training or prediction.

## Configuration

Checkpoints `FZG_s0`, `BIM6_s1`, `FZG_s2` (the `PV2_0044` graph freeze: seed 1's
trained graph frozen into seeds 0 and 2, 7 graph tensors bitwise equal).

```
evaluate_model.py --checkpoint <ck> \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --targets .../targets_v1.pt --manifest .../split_manifest.json \
  --split test --limit 1000 --spike-per-component \
  --geodesic-steps 3 --geodesic-contrast 2.0 --graph-spatial-decay 0.35 \
  --num-time-steps 1024 --settle 512 --spatial-sigma 1.0 \
  --synchrony-threshold 0.05 --spectral-k 14 --spectral-background-count 1
```

Slot Attention: official architecture from `baselines/slot_attention/model.py`,
official object-discovery recipe, trained on our train split (images 0-5999),
60,000 steps at batch 32, seeded inference (`--seed 10`), background slot chosen
per image, same `evaluate_patch_masks`.

Regression gate passed before the test split was touched: `BIM6_s1` on validation 300
reproduced 0.7239 / 0.7251 / 0.4951.

## Limitations

- **k=14 trades foreground IoU.** 0.7124 -> 0.3358 against our own other
  configuration. Predicted foreground is 0.556 against a target of ~0.118, so this
  sits part-way into the over-prediction regime that Slot Attention occupies (0.721
  here). fg_ari is computed over true-foreground patches only, which is why a wider
  predicted foreground helps it.
- **k was selected on validation** to clear 0.75. The contract permits it; it is still
  a selection, and `PV2_0060` records the whole dial rather than this point alone.
- **Our three seeds share one graph** through `PV2_0044`'s freeze, so the variance the
  graph contributes is removed by construction and these are not three fully
  independent training runs.
- **The Slot baseline is a lower bound.** 60,000 steps at batch 32 = 320 epochs
  against the official 457. Its seed 2 is weak (0.5385) and drags the mean; seeds 0
  and 1 reach 0.7265 and 0.7340, so a better-trained Slot could close the fg_ari gap
  while leaving both IoU gaps intact.
- Single evaluation draw per checkpoint for ours (deterministic); Slot uses one seeded
  draw per checkpoint, with draw noise measured at ~0.005 in `PV2_0049`.

## What is not claimed

The third part of the active goal — **a demonstrated route to 0.9** — is **not** met.
`PV2_0034` shows the architecture reaches 0.9754 with oracle features, and ten feature
routes are now closed. Nothing here improves the features; it changes how groups are
read out of the same model.
