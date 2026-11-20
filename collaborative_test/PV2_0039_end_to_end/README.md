# PV2_0039-0040 — end-to-end training, and why the objective alone destroys the features

**completed. The route to 0.9 is now working end to end.** Parent `PV2_0038`.
Seed 0, validation 300, window 1024, kernel sigma 1.0.

## Two things that had been missed

**`--image-dir` was already implemented.** It builds a `GammaGenerator`, puts the
encoder in the training loop, and is mutually exclusive with `--gamma-seq-path`.
Every run on this branch used `--gamma-seq-path`, so **the features were frozen
throughout** -- which is why `PV2_0033`'s three encoder learning rates produced
byte-identical checkpoints. The earlier conclusion that "the image-to-gamma path
has to be built" was wrong; it existed and was simply never used.

**The encoder is one linear convolution.** `CNNFeatureEncoder` is 8 kernels of
3x3 over RGB: `kernels` plus `bias`, **224 parameters** of the model's 200,114, no
nonlinearity and no depth. That is why per-channel IoU against ground truth is
0.079 with two channels at exactly 0.000.

`--encoder-depth` now adds padded layers after it, so the output's spatial size is
exactly what depth 1 gives and nothing downstream changes -- verified identical at
238x318 for depths 1, 2 and 3, with 224, 4,872 and 14,120 parameters.

RGB for all 8000 images was extracted from the tfrecord with the existing script.
Note the `masks8k.pt` it also writes reports 14.7% foreground and 6.48 objects
where `targets_v1.pt` has 11.5% and 6.20 -- different patch-label rules -- so
scoring stays on `targets_v1.pt` and only the images are taken from the
extraction.

## PLV alone collapses the features

| | fg_ari | foreground_iou | matched_object_iou | groups/image |
| --- | --- | --- | --- | --- |
| frozen features (bar) | **0.6639** | 0.6255 | 0.4806 | 6.18 |
| end-to-end, depth 1 | 0.4328 | 0.2888 | 0.2830 | 4.26 |
| end-to-end, depth 3 | **0.2630** | 0.4758 | 0.2056 | **1.85** |

More capacity made it worse, and depth 3 finds 1.85 objects per image against a
true 6.20 -- nearly everything in one group.

**The objective does not supervise the features.** The PLV losses are statements
about phase statistics, not about objects, so an encoder given freedom can satisfy
them by making every patch alike: synchrony becomes trivial and object
information disappears. Larger capacity reaches that degenerate solution faster,
which is exactly the ordering observed.

It also explains why the frozen gamma was better. It came from a reconstruction
autoencoder, so it was forced to stay informative about the image. Reading that
comment as "the encoder has no capacity" was half right -- the capacity is tiny,
and that smallness was also preventing collapse.

## Reconstruction makes it work

`--slot-reconstruction-weight` requires the grouping to explain the image, with
the target built by `patch_pool_rgb` from the batch:

| | fg_ari | foreground_iou | matched_object_iou | groups/image |
| --- | --- | --- | --- | --- |
| frozen features | 0.6639 | 0.6255 | 0.4806 | 6.18 |
| PLV only | 0.4328 | 0.2888 | 0.2830 | 4.26 |
| **reconstruction 1.0** | **0.6779** | **0.6651** | 0.4692 | 6.27 |
| reconstruction 5.0 | 0.2658 | 0.1346 | 0.1991 | 6.93 |

At weight 1.0 end-to-end training **beats the frozen baseline**: fg_ari +0.014 and
foreground IoU +0.040, with matched-object IoU 0.011 lower. At 5.0 reconstruction
dominates and collapses it the other way, predicting 0.48 of patches as
foreground.

So the mechanism is: **PLV shapes the synchrony, reconstruction holds the features
to the image.** Neither alone is enough -- without reconstruction the features
degenerate, with too much of it the grouping does.

## Why this matters for the 0.9 target

`PV2_0034` measured the ceiling at fg_ari 0.9754 with features that separate
objects, so the headroom is 0.27 and it is all in the features. Until now there
was no working way to change them: `PV2_0037` closed DINO (worse, and already
measured here), label-free clustering (worse) and encoder training (a no-op), and
`PV2_0038` closed self-bootstrapping (+0.006, circular).

This is the first route that works. And capacity, which collapsed the model when
nothing held the features in place, can now be tried on top of a constraint that
does -- which is `PV2_0041`.

One caveat carried forward: the generated gamma spans -0.309..0.152 where the
frozen tensor spans -2.93..3.00, about ten times smaller. The core trained on that
same scale so the pairing is consistent and `gamma_phase_mode` standardises, but
it is a possible confound rather than something to assume away.
