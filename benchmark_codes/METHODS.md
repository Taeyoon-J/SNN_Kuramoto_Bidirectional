# Exact current method

## Initialization and forward

`vendor/full_benchmark/model.py:FreshModel` constructs each model from random
seed255000+seed. No pretrained `torch.load` is involved. The shared dendritic
projection bias starts at zero for every variant. The supervised variants have a
contextual8-channel CNN;253 uses its base CNN. Each image is uint8 RGB128×128.
Own-INITIAL encoder statistics are fitted over exactly the chosen3500 TRAIN
images without opening masks (per-channel for240/252/254, global scalar for253).
Features are normalized, clipped to[-3,3] and pooled into patch γ.

Each patch has4 oscillator components. The learned image-conditioned graph uses
fixed8 squared-geodesic distance, spatial spacing16/grid, three geodesic stages,
and top-k512 at64-grid or128 at32-grid. The Kuramoto/sinusoidal/dendritic/membrane
and spike trace has1024forward steps; correlation/readout discards the first512.
Training backpropagates through the last64 time steps, including live graph
construction. It is **not1024-step full BPTT**. The factorized Kuramoto backend,
phase integration wrapper and all active groups are those in the frozen source.

No16-grid checkpoint is expanded in this experiment: each native grid is freshly
constructed. Fixed architectural constants/buffers are not learned. In
particular unused `geodesic_contrast` is explicitly excluded from trainable groups.

## Losses and backward

Every variant uses phase-PLV loss +5×positive-component-spike-Q loss, with
bimodality6, balance10, coherence0.5, collapse1 and target density0.867. Other
UnsupervisedS2NetLoss weights are zero as set in `full_benchmark/loss.py`.

240/252/254 also use mask-defined frequency denoising + γ anchor. Their frozen
teacher is a copy of **their own random INITIAL encoder/projection/frequency**,
not a learned CLEVR teacher. It makes per-truth-region frequency means and
class-balanced weights. Both denoising and anchor coefficients are1.

252/254 use the identical NeutralRegionHead: phase/spike/membrane activity yields
144features/patch; a LayerNorm/MLP embeds them, combines inference-derived region
proposals with12 deterministic farthest-point proposals, and predicts region
logits. All truth regions, including background ID0, are Hungarian-matched to
exchangeable predictions for class-balanced CE + Dice. Neither training nor
inference fixes a dedicated background slot. Hard proposal construction,
Hungarian matching and output argmax remain discrete.

- 252 detaches activity features before final mask-head loss; that loss trains
  the head. Other body losses still train the body.
- 254 keeps activity features live so final mask loss can reach the body/encoder.
- 253 receives no masks and no teacher; the engine asserts no TRAIN labels were
  opened. Masks are loaded only after frozen inference for evaluation.
- 240 has no final NeutralRegionHead; it uses the original largest-component
  foreground support and signed-energy partition readout.

Adam body LR3e-5; encoder LR3e-6; mask-head LR0.003 with separate Adam. Body/head
gradients are separately clipped to norm1. Each image objective is divided by the
actual logical-batch length. All active tensors must accumulate finite nonzero
gradient coverage, each active group must update, and optimizer restoration is
verified before evaluation. `training_report.json` records the actual evidence.

## Data order and budget

The immutable parent TRAIN pool is35000 per dataset. A sorted
RandomState2553500 choice selects3500 unique rows without replacement, shared
across variants/seeds. Each epoch shuffles these with
RandomState(249000+seed+10000×epoch). Inference/training initialization uses
255000+image_ID+100000×epoch. Ten epochs yield35000exposures and2190Adam updates.
The last12 images are retained. γ/graph/activation are computed anew per image.

`initial.pt` stores own INITIAL state/statistics. `resume.pt` atomically saves
model, optimizers, gradient coverage, cursor and RNG state after the first update,
every8updates and each epoch end. Existing pretrained weights are never reused.

## Fixed evaluation

Every image is predicted without GT. Native integer masks are frozen and hashed
before GT is opened for scoring, using the unchanged protected
`vendor/benchmark/evaluation.py` and `make_targets_grid.py`.

| Dataset | Fixed evaluation | Pixel scoring |
|---|---|---|
|Our CLEVR HDF5|IDs1320–1639,320 reserved images|native128pixels; historical64-grid patch scores also reported|
|Multi-dSprites colored_on_colored|TEST7000–7999,1000images|original64×64pixels|
|Tetrominoes|TEST7000–7999,1000images|original35×35pixels;64-grid output first modal32 then protected lifting|
|Objects Room|source36000–36999,1000images|original64pixels; in-distribution heldout, **not official OOD test**|
|CLEVRTex Original|official TEST source0–999,1000image subset|official center192crop and nearest128pixel mask|

Tetrominoes is35×35, not32×32. Native32 model253 predictions are lifted with the
same protected cell boundaries. Multi-dSprites uses the corrected `multi_dsprites`
artifacts; the old malformed `mdsprites` variants are never substituted.

FG-ARI uses only GT foreground pixels and excludes undefined one-object cases
according to the protected evaluator; it does not insert artificial zeros.
Foreground IoU and Hungarian-matched Object IoU use that same protected evaluator.
For252/254 all predicted region IDs are positive, including whatever region
captures GT background. Therefore raw Foreground IoU has this specific semantic
limitation; no post-hoc largest-region deletion is applied to improve that metric.

Each condition report contains all three seeds, arithmetic mean and population
standard deviation. The independent audit recomputes metrics from frozen masks
and verifies all epoch image orders, all35000exposures,2190optimizer steps,
checkpoint identities and saved prediction hashes.

The small-budget result is not a35000-unique/70000-unique result, not a
reproduction of Slot Attention's500000-step training, and not by itself a
controlled proof of each gating's causal contribution.
