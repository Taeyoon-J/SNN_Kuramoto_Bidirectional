# SNN Kuramoto Bidirectional

This repository contains an experimental unsupervised object-discovery pipeline
that combines a learned image feature layer, patch-based oscillator inputs,
Kuramoto dynamics, and a recurrent spiking classifier.

The current research direction treats image feature-map patches as oscillators.
This is different from the earlier version, where each gamma dimension was a
learned latent "brain region." In the patch-based version, oscillator identity
is spatial:

```text
gamma index 0  -> patch row 0, col 0
gamma index 1  -> patch row 0, col 1
...
gamma index 63 -> patch row 7, col 7
```

Because of that fixed spatial meaning, gamma ordering should not be applied in
the patch-based flow. Reordering gamma dimensions would destroy the mapping
needed to reconstruct masks back into image space.

## High-Level Pipeline

The model is trained in stages:

```text
RGB image
  -> CNNFeatureEncoder
  -> feature maps [B, T, H', W']
  -> patch gamma initializer
  -> gamma sequence [B, T, N]
  -> structural connectivity SC [N, N]
  -> S2NetCore
  -> oscillator activity / spike history [B, N, T']
  -> spatial connected components
  -> object-like patch masks
```

Default experimental values used in the current CLEVR runs:

```text
image_size       = 128
num_feature_maps = 8
kernel_size      = 3
feature map size = 126 x 126
patch grid       = 8 x 8
num oscillators  = 64
gamma_seq shape  = [1000, 8, 64]
```

## Core Files

### `input_layer_generator.py`

Defines the image-to-feature-map encoder:

- `CNNFeatureEncoder`
- `CNNFeatureDecoder`
- `CNNAutoEncoder`

The input layer is pretrained as an image autoencoder:

```text
image -> CNNFeatureEncoder -> feature maps -> CNNFeatureDecoder -> image reconstruction
```

The trained encoder is reused for later stages.

### `gamma_initializer.py`

Contains the patch-based gamma-generation path:

```text
feature map -> FeaturePatchGammaInitializer -> patch gamma vector
```

`FeaturePatchGammaInitializer` uses either adaptive grid pooling or fixed patch
pooling. For the current `8 x 8` experiment, each feature map is pooled into 64
spatial patch values. Those 64 values become oscillator gamma inputs. There is
no learned gamma autoencoder in this branch.

### `sc_generator.py`

Builds a structural connectivity matrix from gamma samples using absolute
Pearson correlation:

```text
gamma samples [num_samples * T, N] -> SC [N, N]
```

For the patch flow, `N` is the number of patch oscillators.

### `s2net_cls.py`

Defines the main model modules:

- `GammaGenerator`
- `S2NetCore`
- `S2NetClassifier`

`S2NetCore` is the current training target. It consumes precomputed
`gamma_seq [B, T, N]` and a fixed SC matrix. Internally it runs:

```text
gamma_seq
  -> graphVectorKuramoto
  -> sinusoidal_gating
  -> DendricLayer
  -> MembraneLayer
  -> spikes and membrane activity
```

`S2NetCore.forward(..., return_core_out=True)` returns:

```text
object_groups, spikes, core_out
```

where `core_out` is the membrane history before hard spike thresholding.

### `spike_classifier.py`

Contains several post-processing options:

- `spike_rhythm`
- `spike_interval`
- `spike_spatial_components`

The current patch-based object extraction uses `spike_spatial_components`.
It converts oscillator activity into a spatial patch grid:

```text
activity [B, N, T]
  -> reshape N into [grid_h, grid_w]
  -> temporal aggregate, usually mean
  -> threshold
  -> connected components
  -> object candidate groups
```

This avoids treating each time interval as an object. That was a problem with
`spike_interval` when `T = 8` and `interval_size = 1`, because each time step
became its own object-like active set.

### `loss_function.py`

Defines the unsupervised losses used for S2NetCore training.

Base losses:

- `spike_rate_loss`
- `spike_temporal_smoothness_loss`
- `spike_diversity_loss`
- `structural_consistency_loss`
- `object_overlap_loss`

Patch-flow diagnostic losses:

- `sample_activity_diversity_loss`
- `spatial_compactness_loss`
- `temporal_activity_balance_loss`

These were added after diagnostics showed:

```text
different images -> nearly identical masks
active patches -> scattered point components
activity over time -> monotonic collapse or saturation
```

The current loss-fix training uses these extra terms to directly discourage
fixed spatial templates and unstable temporal dynamics.

## Gamma Ordering

Gamma ordering is useful only when gamma dimensions are learned latent slots
whose order has no physical meaning.

In the patch-based model, gamma dimensions already have spatial identity:

```text
dimension i = fixed patch location i
```

Therefore gamma ordering should be skipped for patch-gamma experiments.
Otherwise the model can no longer map oscillator indices back to image patches
for reconstruction or mask visualization.

## Training Flow

### 1. Train the Input Layer

The input layer learns feature maps by reconstructing images.

Example server command:

```bash
python training/train_input_layer_generator.py \
  --image-dir /work/USERS/tkim1/clevr/CLEVR_1k/CLEVR_v1.0/images/train \
  --save-path /work/USERS/tkim1/checkpoints/input_encoder_k8_ks3_img128.pt \
  --num-kernels 8 \
  --kernel-size 3 \
  --image-size 128 \
  --max-images 1000 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda
```

### 2. Generate Patch Gamma Sequences

The patch gamma path does not train a gamma autoencoder. It converts feature
maps directly into patch oscillator gamma values.

```bash
python training/train_gamma_initializer.py \
  --image-dir /work/USERS/tkim1/clevr/CLEVR_1k/CLEVR_v1.0/images/train \
  --input-encoder-path /work/USERS/tkim1/checkpoints/input_encoder_k8_ks3_img128.pt \
  --save-path /work/USERS/tkim1/checkpoints/patch_gamma_config_k8_grid8_img128.pt \
  --gamma-seq-save-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_gamma_seq_k8_grid8.pt \
  --preprocess-save-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_feature_preprocess_k8_grid8.pt \
  --num-kernels 8 \
  --kernel-size 3 \
  --image-size 128 \
  --max-images 1000 \
  --patch-grid-size 8 \
  --feature-normalize standardize \
  --feature-clip 3.0 \
  --batch-size 32 \
  --device cuda
```

Expected output:

```text
gamma_seq: [1000, 8, 64]
```

### 3. Train S2NetCore

Current loss-fix training command:

```bash
python training/train_s2net_core.py \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_gamma_seq_k8_grid8.pt \
  --sc-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_sc_k8_grid8.pt \
  --save-path /work/USERS/tkim1/checkpoints/s2net_core_clevr1k_patch_k8_grid8_spatial_lossfix.pt \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda \
  --spike-classify-method spatial_components \
  --spike-spatial-grid-size 8 \
  --spike-spatial-activity-source sigmoid_membrane \
  --spike-spatial-time-aggregate mean \
  --spike-spatial-threshold 0.45 \
  --spike-spatial-min-group-size 2 \
  --loss-signal sigmoid_membrane \
  --spike-rate-weight 0.2 \
  --spike-target-rate 0.25 \
  --structural-weight 0.03 \
  --spike-diversity-weight 0.05 \
  --sample-diversity-weight 0.2 \
  --spatial-compactness-weight 0.1 \
  --temporal-balance-weight 0.5 \
  --loss-patch-grid-size 8 \
  --object-overlap-weight 0 \
  --grad-clip-norm 1.0 \
  --verbose
```

Important training diagnostics to watch:

```text
spike_rate
sample_diversity
spatial_compactness
temporal_balance
structural
```

If `spike_rate` collapses to zero, the model is suppressing activity. If
`sample_diversity` stays high and masks are identical across images, the model
is producing a fixed spatial template. If `temporal_balance` is high, activity
is collapsing or saturating over time.

## Visualization and Diagnostics

Use compact visualization for quick inspection:

```bash
python training/visualize_s2net_objects.py \
  --image-dir /work/USERS/tkim1/clevr/CLEVR_1k/CLEVR_v1.0/images/train \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_gamma_seq_k8_grid8.pt \
  --sc-path /work/USERS/tkim1/gamma_sequences/clevr1k_patch_sc_k8_grid8.pt \
  --checkpoint-path /work/USERS/tkim1/checkpoints/s2net_core_clevr1k_patch_k8_grid8_spatial_lossfix.pt \
  --output-dir /work/USERS/tkim1/visualizations/s2net_spatial_lossfix_2samples \
  --num-samples 2 \
  --figure-mode compact \
  --patch-grid-size 8 \
  --activity-source sigmoid_membrane \
  --activity-threshold 0.45 \
  --time-aggregate mean \
  --device cuda \
  --spike-classify-method spatial_components \
  --spike-spatial-grid-size 8
```

Compact mode creates:

```text
s2net_sample_0000.png
s2net_sample_0001.png
s2net_object_summary.json
diagnostics_summary.json
sample_diagnostics.csv
```

The PNG panels show:

```text
1. original image
2. final spatial mask
3. image x mask
time 0 ... time 7 activity masks
object candidate connected components
```

The numeric diagnostics are often more useful than the image alone.

Key fields:

- `unique_binary_masks`: how many distinct masks appear across samples.
- `mean_pairwise_mask_iou`: whether different images produce the same mask.
- `mean_mask_density`: fraction of active patches.
- `mean_num_components`: number of connected object candidates.
- `mean_largest_component_size`: size of the largest candidate.
- `active_patches_by_time`: whether activity collapses or saturates over time.
- `spike_rate`: whether binary spikes are active at all.

Typical failure modes:

```text
unique_binary_masks = 1
mean_pairwise_mask_iou = 1.0
```

The model is producing a fixed mask template.

```text
mean_mask_density = 0.0
spike_rate = 0.0
activity_max < threshold
```

The activity threshold is too high or the model has suppressed activity.

```text
active_patches_by_time = [45, 22, 15, 11, 8, 7, 3, 2]
```

Activity is collapsing over time instead of representing stable object
evidence.

## Current Research Status

The patch-gamma architecture is implemented and runs end to end. Diagnostics
showed that the original interval classifier produced time-slice masks rather
than true object masks. The spatial connected-component classifier fixes that
post-processing mismatch, but the core model still needs loss pressure to avoid
fixed spatial templates and temporal collapse.

Current active direction:

```text
patch gamma flow
  + spatial_components classifier
  + sample diversity loss
  + spatial compactness loss
  + temporal balance loss
```

The goal is to make oscillator activity image-specific, spatially compact, and
temporally stable enough to reconstruct object-like regions.

