# Default training and evaluation settings

This file records the supported `patch_v2_sw` baseline. Paths remain explicit
because they differ between machines; the model and loss values below are now
the defaults of `training/train_s2net_core.py`.

## Training baseline

| Setting | Value |
|---|---:|
| feature maps | 8 |
| regions / oscillators | 256 (16 x 16) |
| epochs | 40 |
| batch size | 16 |
| learning rate | 1e-3 |
| seed | 0 |
| oscillator dimension | 4 |
| gamma drive | static |
| recurrent steps | 64 |
| PLV settle steps | 32 |
| theta initialization | gamma |
| gamma phase mapping | standardize_tanh |
| frequency gain | 2.0 |
| graph | learned, top-k 32 |
| graph spatial decay | 0.55 |
| Kuramoto K | 256 |
| membrane threshold | 0.06 |
| dendritic tau initialization | U(-4, 0) |
| membrane tau initialization | U(-4, 0) |
| gate | raw |
| PLV source / component combination | phase / mean |
| loss signal | sigmoid_membrane |
| spike-rate / smooth / diversity weights | 0 / 0 / 0 |
| structural weight | 0 |
| PLV collapse weight | 1.0 |
| PLV bimodality weight | 1.0 |
| PLV balance weight | 10.0 |
| PLV target density | 0.867 |
| PLV coherence weight | 0.5 |

Minimal phase-readout training command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path /path/to/gamma_seq_k8_grid16.pt \
  --save-path /path/to/runs/BEST/core_phase.pt \
  --device cuda --verbose
```

For the component-wise spike readout, train a separate checkpoint by adding:

```bash
  --spike-per-component
```

The phase dynamics and phase objective are unchanged by that flag, but the
dendritic input shape differs, so the phase and component-spike checkpoints
must not overwrite each other.

## Evaluation baseline

Evaluation uses held-out samples and a longer rollout:

| Setting | Value |
|---|---:|
| held-out start (`--skip`) | 1000 or later |
| number of images | 300 recommended |
| recurrent steps | 256 |
| settle steps | 64 |
| grid / regions | 16 x 16 / 256 |
| graph top-k / spatial decay | 32 / 0.55 |
| gate / oscillator dimension | raw / 4 |
| fixed cluster counts | 3, 4, 6, 8 |

```bash
CUDA_VISIBLE_DEVICES=0 python -m snn_kuramoto_bidirectional.training.evaluate_binding \
  --checkpoint /path/to/runs/BEST/core_phase.pt \
  --gamma-seq-path /path/to/gamma_seq_k8_grid16.pt \
  --patch-labels /path/to/clevr_patch_labels.pt \
  --skip 1000 --num-images 300 \
  --num-regions 256 --grid 16 \
  --num-time-steps 256 --settle 64 \
  --graph-top-k 32 --graph-spatial-decay 0.55 \
  --gate-mode raw --osc-dim 4 \
  --fixed-k 3 4 6 8 --device cuda
```

When evaluating the separately trained component-spike checkpoint, add
`--spike-per-component`. The flag used for evaluation must match training.
