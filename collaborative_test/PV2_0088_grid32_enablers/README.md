# PV2_0088 — what had to be true before a 32x32 grid could run

**Status:** completed. **Parent:** `PV2_0061`. **Contract:** v1 for every regression
check here; defines **contract v2** for the 32x32 grid.
**Hypothesis:** none. Three blocking items, each with its own check.
**Success criterion:** contract v1 must still score 0.7239 / 0.7251 / 0.4951 after every
change, and the new grid rule must reproduce `targets_v1.pt` exactly at grid 16.

## 1. The target rule on a grid that does not divide

Contract v1 can use `clevr_mask_patch` because 240/16 = 15 and 320/16 = 20 divide. A
32x32 grid does not: 240/32 = 7.5. The peer's own 32x32 contract avoids this by working
on 128x128 images with 4x4 cells, but that is **a different image geometry and a
different split** (their validation is IDs 1320-1639 of a 70,000-ID pool; ours is
6000-6299 of 8,000), so it cannot be adopted without abandoning our fixed manifest.

`make_targets_grid.py` keeps the modal rule and places cell edges at `round(i*H/g)`, so
the cells partition the image exactly and alternate between 7 and 8 rows. Ties still go
to the smallest instance ID; background is still 0; nothing is resized or expanded to
pixels.

**The check that makes it trustworthy:** at g = 16 the edges are `round(15i) = 15i`, so
the rule must reduce to contract v1's own cells. Run against all 8,000 images:

```
verify against targets_v1.pt: labels IDENTICAL
```

Bit for bit. So this is a generalisation of contract v1's rule, not a new rule.

| | grid 16 (v1) | grid 32 (v2) |
| --- | --- | --- |
| cells | 256 of 15x20 px | 1024 of 7-or-8 x 10 px |
| foreground | 11.5% | 11.9% |
| objects per image | 6.20 | 6.49 |
| cells with a tie | 0.07% | 0.21% |
| mean purity of the chosen ID | 0.957 | **0.978** |
| ground-truth same-group pair fraction | 0.7900 | 0.7841 |

Purity rises and more objects survive, which is what a finer grid should do.

## 2. The memory wall

`PV2_0087` counted the relaxation's peak intermediate as `B·N³` floats. Measured, not
predicted — at N=1024 batch 8 the unchunked path asks the allocator for exactly
34,359,738,368 bytes and dies.

`_relax` now forms the same `[B,N,N,N]` sum a block of rows at a time and reduces it
over the same axis. Under autograd every block would still be kept, which would defeat
the point, so each block is wrapped in `torch.utils.checkpoint` and recomputed in the
backward pass.

| | chunk 0 | chunk 32 | chunk 64 |
| --- | --- | --- | --- |
| max \|delta\| vs unchunked, N=256 | reference | **0.000e+00** | **0.000e+00** |
| peak memory, N=1024 batch 8, with backward | **OOM** | 4.72 GB | 9.02 GB |
| input gradient norm | 3.343127e-06 | 3.343127e-06 | 3.343127e-06 |

Bit-identical, gradients included — every parameter's gradient norm matches to all
printed digits (`geodesic_contrast` 3.3580e-07, `log_coupling_gain` 4.0960e+03,
`log_temperature` 2.0176e-04, `spatial_rate` 8.9540e-07, `projection.weight` 6.6485e-06).

**Default is 0**, so every existing command produces the bytes it did before. Contract v1
re-scored with `--geodesic-chunk 64` gives 0.7239 / 0.7251 / 0.4951 at predicted
foreground 0.1290 — unchanged.

Changed: `graph_generator.py` (`geodesic_chunk` argument, `_relax` extracted from
`_geodesic_distance`), `hyperparameter.py` (`geodesic_chunk: int = 0`), `s2net_cls.py`
(passes it through), `train_s2net_core.py` and `evaluate_model.py` (`--geodesic-chunk`).

## 3. The gamma recipe, reconstructed

No script in the tree produces `wm10k_full_grid16.pt`; only `configs/best_16x16.sh`
mentions it. A 32x32 grid needs the same encoding at a different grid, so the recipe had
to be recovered, and three wrong guesses are worth recording because each was ruled out
by measurement rather than by reading:

- **"the pooling is different"** — no. Correlation against the reference is 0.9999 for
  `size=128, no crop, pad=0`, flat across bilinear/bicubic/nearest and across [0,1] vs
  [-1,1] scaling. The feature extraction was right from the start.
- **"the channels are permuted"** — no, though `gamma_ordering.order_gammas` exists and
  permutes exactly this axis per image. Fitting a permutation plus an affine per image
  returns the **identity** permutation on every image tested.
- **"it is per-channel or per-image standardisation"** — no. The fit returns one global
  affine, scale 6.7499 and offset -0.1094, identical to four decimal places on every
  image, which is global standardisation with the dataset's own statistics.

So: resize to 128x128, one valid 3x3 convolution with the trained 8 kernels,
`adaptive_avg_pool2d` to the grid, then standardise globally.

**Where it does not close.** Mean residual against the reference is 1.18e-4 and every
image's best match is its own row (cosine 0.9996-1.0000), but **8.6% of images have a
few cells off by up to 1.1**, and no resize backend tried — PIL bilinear/bicubic/nearest,
torch bilinear/bicubic/area/nearest, antialias on and off — removes them. The reference
is matched in the mean and not exactly. This matters little for contract v2, whose model
is trained from scratch on the grid-32 tensor and never sees the grid-16 one, but it is
**not** a reproduction and is not claimed as one.

`wm10k_full_grid32.pt` is (10000, 8, 1024), mean -0.000000, std 1.000000.

## Files

| file | what |
| --- | --- |
| `collaborative_test/make_targets_grid.py` | contract v2 targets, with `--verify-against` |
| `collaborative_test/build_gamma_grid.py` | gamma at any grid, with `--compare` |
| `collaborative_test/visualize_patch_masks.py` | renders the npz `--save-predictions` wrote |
| `/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt` | 8000 x 32 x 32, contract 2 |
| `/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt` | 10000 x 8 x 1024 |

## Interpretation and limitations

None of this is a result; it is the cost of asking a question. The one finding with
teeth is that **the cubic term was never the obstacle it appeared to be** — the wall was
memory layout, and 68.7 GB became 4.72 GB with the arithmetic untouched.

The gamma residual above is the real limitation: contract v2's input is built by a
recipe that matches the contract-v1 artifact in the mean but not exactly, so a
cross-contract comparison of absolute numbers carries that caveat. Within contract v2
everything is self-consistent.

## Next

`PV2_0089` trains the published configuration on this grid.
