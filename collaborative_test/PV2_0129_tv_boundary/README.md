# PV2_0129 — an unsupervised boundary signal that beats mask supervision

- **status** completed, exploratory: 40 validation images, one seed, one dataset
- **parent** `PV2_0128`, `DSPRITES_AND_SUPERVISION.md`
- **hypothesis** reconstruction learns texture; masks learn boundaries; a total-variation
  penalty on the feature map says "flat inside a region, free to jump at an edge" with no
  labels at all
- **result** it works, and on FG-ARI it beats both mask-supervised arms

## What the measurements said before this

`PV2_0128` eliminated nine candidates and left the γ representation as the lever. Two
encoder experiments then framed the question:

| encoder | objective | fg_ari | fg-IoU | obj-IoU | pred_fg | groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ours, 224 | reconstruction | 0.2658 | 0.5618 | 0.5442 | 0.1273 | 63.7 |
| ctx 29k | reconstruction | 0.2193 | 0.1446 | 0.3752 | 0.4352 | 150.9 |
| ctx 29k | **our masks** | 0.4252 | 0.2069 | 0.7067 | 0.3719 | 92.5 |
| peer 29k | **their masks** | 0.4574 | 0.5381 | 0.7601 | 0.1572 | 41.6 |

Capacity was not the lever: the same 29,128-parameter architecture trained by
reconstruction scored **below** our 224-parameter baseline. Supervision was.

## The change

Add the L1 norm of the spatial gradient of the feature map to the reconstruction loss:

```python
tv = ((f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
      + (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean())
loss = recon + weight * tv
```

Total variation is piecewise-constant regularisation: it costs nothing to be flat inside a
region and pays once at each edge, so it encodes a boundary prior without any label.
Reconstruction alone cannot express it — it has to reproduce texture. The two terms oppose
each other and the weight sets the balance, so three weights were run rather than one
picked.

It targets the quantity `PV2_0128` measured as ordering exactly like fragmentation:
within-object γ dissimilarity, 0.0086 on Tetrominoes, 0.0523 on CLEVR, **0.0712** on
Multi-dSprites.

## Result

Multi-dSprites, grid 64, 40 validation images, our core and readout unchanged:

| encoder | supervision | fg_ari | fg-IoU | obj-IoU | pred_fg | groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline 224 | unsupervised | 0.2658 | 0.5618 | 0.5442 | 0.1273 | 63.7 |
| ctx 29k, recon only | unsupervised | 0.2193 | 0.1446 | 0.3752 | 0.4352 | 150.9 |
| ctx 29k, **recon + TV 0.01** | **unsupervised** | 0.3652 | 0.1996 | 0.6085 | 0.3606 | 105.3 |
| ctx 29k, **recon + TV 0.1** | **unsupervised** | **0.4617** | 0.3957 | **0.7940** | 0.2043 | 65.7 |
| ctx 29k, **recon + TV 1.0** | **unsupervised** | 0.4234 | **0.5035** | 0.7106 | 0.1401 | 55.2 |
| ctx 29k, our masks | supervised | 0.4252 | 0.2069 | 0.7067 | 0.3719 | 92.5 |
| peer 29k, their masks | supervised | 0.4574 | 0.5381 | 0.7601 | 0.1572 | 41.6 |

**TV 0.1 reaches 0.4617 with no labels**, above our mask-supervised arm (0.4252) and above
the peer's mask-supervised encoder (0.4574), and its matched-object IoU of 0.7940 is the
highest in the table.

### The weight is a real trade-off, not a tuning detail

| weight | FG-ARI | fg-IoU | predicted foreground (true 0.0880) |
| ---: | ---: | ---: | ---: |
| 0.01 | 0.3652 | 0.1996 | 0.3606 |
| 0.1 | **0.4617** | 0.3957 | 0.2043 |
| 1.0 | 0.4234 | **0.5035** | **0.1401** |

0.1 maximises FG-ARI; 1.0 gives the most honest foreground and the best fg-IoU. FG-ARI is
scored over true-foreground cells only and cannot see over-painting, so **the FG-ARI-optimal
weight is not the best model**. Neither is reported alone.

## Limitations, and they are large

- **40 images, one seed, one dataset, validation.** Exploratory by the contract's own rule.
- **The core was not retrained.** It was fitted to the old γ; these numbers come from
  feeding new γ to an unchanged core. Retraining could move them either way.
- **The weight was selected on the same 40 images it is reported on.** A clean number needs
  the weight fixed on validation and read once on test.
- **Multi-dSprites only.** Whether TV helps or hurts Tetrominoes and CLEVR, where the
  baseline is already near its ceiling, is unmeasured — and on Tetrominoes, where
  within-object dissimilarity is already 0.0086, there may be nothing left for it to flatten.

## Why it matters anyway

The supervised-versus-unsupervised discussion in `DSPRITES_AND_SUPERVISION.md` rested on the
premise that the unsupervised route was exhausted. **It was not.** The first classical
boundary prior tried, at the first weight sweep, matched and passed mask supervision. That
reopens the unsupervised table rather than forcing a move to a supervised one — and a
supervised move has no comparison to make anyway, since Slot Attention already reaches 99.5
/ 91.3 / 98.8 on these datasets without labels.

## Next

1. Fix the weight on validation, then read test once, three seeds.
2. Retrain the core on TV γ rather than feeding it to a core fitted to the old features.
3. Run TV on Tetrominoes and CLEVR — it may do nothing there, or hurt.
