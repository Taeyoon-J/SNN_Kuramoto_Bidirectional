# SW0061 - label-free RGB patch feature pilot

The peer oracle feature experiment shows that the existing SNN/Kuramoto path
can reach FG-ARI `.9754` when patches from the same object receive identical
features. SW0061 tests the first deployable feature intervention without labels
or retraining: replace the reconstruction-autoencoder gamma with eight local
appearance channels computed from the same validation images.

Each 16x16 patch receives mean R, G, B, luminance, chroma, horizontal gradient,
vertical gradient, and gradient magnitude. Generation reads images only; masks,
object count, and validation labels are never opened. The existing SW0053 seed0
checkpoint, dynamics, readout, IDs1320-1351, T256/settle64, and threshold .35
remain fixed. This isolates whether cleaner object-local input features transfer
through the already trained core before any encoder-training experiment.

The 32-image pilot advances only if all three fixed metrics improve over the
validated SW0054 raw reference.

```bash
bash collaborative_test/SW_0061_rgb_patch_features/evaluate.sh 0
```

The pilot completed on CPU because the already scheduled SW0055 training took
GPUs 0/1 between smoke and launch. Device choice does not change this
deterministic inference contract. RGB patch gamma scored
`.308029/.383089/.190871`, versus `.668520/.692026/.489879` for the checkpoint's
native autoencoder gamma. Object-count MAE also rose to `3.40625` with strong
under-counting. Direct feature substitution is rejected. The result is
consistent with a core whose learned projection is tied to its training feature
basis; a new encoder must be trained jointly with the core or followed by core
adaptation. Reports and the label-free generation manifest are in `results/`.
