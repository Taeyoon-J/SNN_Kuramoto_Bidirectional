# SW_0016: RGB-edge membrane separation loss

Status: seed-0 40-epoch training and fixed-split validation complete. Mixed result; not adopted.

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

A synthetic-input comparison against the original `patch_sw` implementation
found exactly the same scalar loss and membrane gradient (maximum absolute
difference 0.0 for both). See `verify_equivalence.py` and `equivalence.json`.

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

## Result and decision

The GPU-2 run completed 40 epochs. The fixed validation set was CLEVR HDF5
IDs 1320-1639 (320 images). Under the *same* component-product spike
classifier threshold 0.50, the controlled comparison is:

| Metric (higher is better) | SW_0011 no RGB-edge loss | SW_0016 RGB-edge loss 0.1 |
|---|---:|---:|
| Patch FG-ARI | 0.179760 | 0.069476 |
| Patch foreground IoU | 0.271468 | 0.331876 |
| Patch matched-object IoU | 0.295250 | 0.164980 |
| Predicted groups/image | 77.03 | 11.98 |

Thus foreground detection and the number of groups improved, but separating
individual objects became much worse. Sweeping the spike classifier threshold
did not recover an all-metric improvement: the best FG-ARI row (threshold
0.80) scored 0.071724 / 0.332549 / 0.168455; the best matched-object-IoU
row (threshold 0.95) scored 0.069077 / 0.332043 / 0.172222. These are
seed-0 validation scores, not a three-seed result. Do **not** adopt this
weight-0.1 RGB-edge objective as a default. Full results are in
`validation_results.json`.

On the first 16 validation images, phase-to-spike affinity correlation was
0.7287 (SW_0011: 0.7265). Same- versus different-object spike synchrony was
0.4608 versus 0.3607, a contrast of 0.1001 (SW_0011: 0.4296 versus 0.2756,
contrast 0.1540). Same/different membrane synchrony was 0.4607/0.2802.
These diagnostics suggest that better membrane boundary separation did not
translate into better spike object grouping; this is an inference rather
than a demonstrated causal mechanism. See `signal_flow.json`.
