# SW_0016: RGB-edge membrane separation loss

Status: seed-0 40-epoch training running on server GPU 2. Validation not yet complete; no performance claim.

## Why this test

SW_0013/SW_0014 showed that classifier-only adaptive slots reduce but do not
eliminate overfragmentation or improve all three metrics. The user suggested
`patch_sw`'s `edge_membrane_separation_loss`, which encourages adjacent patches
across strong image RGB boundaries to have dissimilar membrane rhythms. This
targets the spatial/object-boundary signal directly while leaving the
Kuramoto and SNN core computation unchanged. SW_0015 found strong empirical
gamma/HDF5 index alignment, which is necessary before pairing RGB with gamma.

## Controlled comparison

SW_0011 40-epoch, seed-0 setup is reused exactly: first 1000 training gamma
examples, 8 channels, 16×16 patches, batch 16, Adam 1e-3, static 64-step
drive, phase PLV, graph teacher weight 0.1, `sigmoid_membrane`, membrane and
spike settings, all other loss weights unchanged. A fresh model is initialized
from seed 0. **Only** an RGB-boundary membrane loss with weight 0.1 and margin
0.3 is added. The first 1000 HDF5 RGB images are paired by index with gamma;
no GT masks are loaded during training. The loss is off by default (weight 0),
so other training runs retain their previous behavior.

An eight-image validation diagnostic on the old checkpoint found raw edge
loss 0.478038. Gradients reached `dendric_layer.oscillator_dense.weight`
(norm 0.008427), `oscillator_dense.bias` (0.003484), and
`membrane_layer.tau_m` (0.000571). Thus this loss can train the SNN path,
unlike a phase-only objective. These are unweighted gradients, not a guarantee
of improved segmentation. Details: `diagnostic.json`.

## Exact code changes

- `snn_kuramoto_bidirectional/loss_function.py`: add standalone
  `edge_membrane_separation_loss`, adapted from `patch_sw`, with centered
  cosine membrane similarity on neighboring patches and direct RGB boundary
  difference weights. Existing loss formulas are unchanged.
- `snn_kuramoto_bidirectional/training/train_s2net_core.py`: add optional
  `--edge-image-hdf5`, `--edge-membrane-weight`, and
  `--edge-membrane-margin`. When weight is nonzero, form paired gamma+RGB
  batches and add weighted edge loss; otherwise retain gamma-only batches.
- `run.sh`: reproducible training command; `evaluate.sh`: after the checkpoint
  is complete, compare the same peer spike readout and signal-flow diagnostic
  on fixed validation IDs 1320–1639/first 16 respectively.

Server training output:
`/Data0/kevinswk/patch_v2_sw/trained_models/SW_0016_rgb_edge_loss_seed0/`.
The first GPU-1 attempt was stopped after one epoch when another process
started sharing GPU 1. No checkpoint was written. The same run was restarted
from seed 0 on then-idle GPU 2 with log `training_gpu2.log`.
