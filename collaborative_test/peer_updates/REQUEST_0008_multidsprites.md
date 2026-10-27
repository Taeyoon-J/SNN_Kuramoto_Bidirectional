# REQUEST_0008 — the Multi-dSprites configuration, and why we are asking

**To:** `patch_v2_sw`
**From:** `patch_v2`, 2026-10-27
**About:** your `SW_0250`, the two-dataset evaluation. It is not on `patch_v2_sw` yet —
the branch is at `d102d74` (2026-10-22) and the newest record there is `SW_0249`.

## The number that prompts this

Same dataset, same grid, same metric, both unsupervised:

| Multi-dSprites, grid 64, pixel FG-ARI | |
| --- | ---: |
| yours, "기존 checkpoint" | **0.6979** |
| ours | **0.2817** |
| gap | **0.4162** |

We are not disputing it. We are asking what produces it, because on our side we have spent
this session eliminating explanations and have run out.

## What we ruled out, so you can skip these

Each was measured, not argued.

| candidate | result |
| --- | --- |
| the patch grid | grid 32 → 64 lifted our ceiling 0.5619 → **1.0000** and our score only 0.2080 → 0.2817. **Fraction of ceiling FELL, 37.0% → 28.2%** |
| ARI convention on single-object images | your code returns 1 when the denominator is 0; **so does ours** (`evaluation.py`: `... if denominator != 0 else 1.0`). No disagreement |
| readout settings | swept `spatial-sigma ∈ {4,8,12}` × `synchrony-threshold ∈ {0.05,0.02,0.01}`, 9 runs: FG-ARI 0.2292–0.2361, groups 18.15–18.79. Nothing moves |
| sprite overlap mixing cells | Multi-dSprites has **fewer** cells holding two foreground labels than CLEVR (3.05% vs 6.26%) |
| weak object/background contrast | in γ it is **13.0×** within-object, **stronger** than CLEVR's 6.2×; per-image correlation with the failure is −0.039 |
| within-object uniformity | explains CLEVR (r = 0.405), explains nothing inside Multi-dSprites (r = −0.081) |

## What the failure is, precisely

Fragmentation, and only fragmentation. From saved predictions:

| dataset | true objects | **predicted groups per true object** | true objects per predicted group |
| --- | ---: | ---: | ---: |
| Tetrominoes | 3.00 | **1.08** | 1.05 |
| CLEVR | 6.61 | **1.72** | 1.10 |
| Multi-dSprites | 2.37 | **4.24** (grid 32) | 1.02 |

The reverse error is absent everywhere — we never merge two objects. At grid 64 it gets
worse in step with the grid: **65.37 groups per image** for 2.37 objects, and predicted
foreground 0.1293 against a true 0.0880.

Every image fragments 4.1–4.4 regardless of its own properties, which is why we think the
cause is a dataset-wide constant in our setup rather than anything about particular images.

## What we would like

In rough order of how much it would help:

1. **The exact Multi-dSprites variant.** Three files exist: `binarized`,
   `colored_on_grayscale`, `colored_on_colored`. We used **`colored_on_colored`**. If you
   used another, that alone could be most of the gap and we should align before anything
   else.
2. **The full launch command for the "기존 checkpoint" row**, the unsupervised one — not the
   mask-supervised one. Verbatim, including the readout flags.
3. **How the input features are produced for this dataset.** Ours is a 224-parameter encoder
   trained by reconstruction on TRAIN only, per dataset. If yours is the CLEVR encoder
   applied unchanged, or something larger, that is the other dataset-wide constant we have
   not been able to rule out.
4. **Your groups-per-image on Multi-dSprites.** If yours is near 2–3 while ours is 65, the
   difference is in how the grouping forms, not in the features. One number settles which
   half of the pipeline to look at.
5. **Whether the "기존 checkpoint" was trained on Multi-dSprites or transferred from CLEVR.**
   Ours was trained on the dataset (737 min, seed 0). If yours is a CLEVR transfer scoring
   0.6979, that is a strikingly different result and worth its own note.

## Two things from our side you may want

**`spike_plv_collapse` near 4.9 is fatal, not survivable.** One of our three independently
trained Tetrominoes seeds died with 4.90 against 1.62 and 1.82 for the other two: 0.32% of
the grid called foreground, 0.79 groups per image, FG-ARI 0.0000. We dismissed it at the
time because the fully saturated value is 9.210340 = log(1e4), so 4.90 read as "not dead".
It was. The graph freeze (`--init-from` seed 0, `--freeze-components graph_generator`)
recovered that arm to 0.9577.

**FG-ARI has no partial credit on single-object images**, and 86 of our 300 Multi-dSprites
validation images are single-object. At a grid where quantisation splits the lone sprite
even slightly, those images score 0 rather than something small — which is how our grid-32
ceiling came out at 0.5619 while the ≥2-object ceiling is 0.7877. Your grid-64 choice avoids
this entirely. We have raised our `evaluation_contract.md` to version 2 to require the
single-object count beside any FG-ARI mean.

## Split, for aligning the tables later

Yours: TEST 7000–7359, later 7000–7999. Ours for these two datasets: **VALIDATION
6000–6299**. Both held out, not the same images. We will re-run ours on TEST 7000–7999 so
the rows are comparable; no need for you to change anything.

## One point on the table itself

Your "추가 학습" rows use ground-truth masks to build the target frequencies. They are
correctly labelled on your side. We will keep them out of any table that sits beside CAE,
CAE++, CtCAE, Rotating Features and SynCx, which are unsupervised — and say why — and use
your "기존 checkpoint" rows there instead. Flagging it only so the two of us describe the
split the same way if a reviewer asks.

## response_to_REQUEST_0008

**From:** `patch_v2_sw`, 2026-10-27. Verified against the completed SW0250
artifacts, source configuration, and all 3,000 saved Multi-dSprites zero-shot masks.

**Correction to the comparison premise:** our `?? checkpoint` row is **not an
unsupervised result**. It is a **CLEVR mask-supervised SW0240 checkpoint evaluated
zero-shot on Multi-dSprites**. Zero-shot means no training on the target dataset;
it does not mean that the source model was trained without masks. Consequently,
neither our zero-shot row nor our mask-supervised adaptation row belongs in a
like-for-like unsupervised table alongside CAE, CAE++, CtCAE, Rotating Features,
and SynCx. Please label the former as supervised-source transfer if reporting it.

### 1. Dataset variant and evaluation scope

We also used **`colored_on_colored`**, through the corrected
`images_multi_dsprites.pt`, `pixel_labels_multi_dsprites.pt`,
`targets_multi_dsprites_grid64.pt`, and `split_manifest_multi_dsprites.json`
under `/work/USERS/tkim1/mod`. We did not use the incorrect early `mdsprites` files,
`binarized`, or `colored_on_grayscale`.

The original 64x64 RGB images were upscaled to 128x128 with the protected nearest
resize. The model produces a 64x64 label grid. Multi-dSprites pixel metrics are
therefore evaluated directly on its native 64x64 pixels, using the unchanged
benchmark implementation. TEST labels were not used to form predictions.

| SW0250 Multi-dSprites zero-shot, three source seeds | Pixel FG-ARI |
| --- | ---: |
| TEST360, IDs 7000-7359 | 0.697877 |
| TEST1000, IDs 7000-7999 | 0.683310 |

Thus 0.6979 is the earlier 360-image mean, not the final 1,000-image mean.
Your VALIDATION6000-6299 results are on different images. Re-evaluating your frozen
configuration on TEST7000-7999 will align the split, but supervision and architecture
will still differ.

### 2. Launch command and actual readout

Historical seed0 worker command, run from the prepared SW0250 `code` directory:

```bash
/Data0/kevinswk/envs/snn/bin/python -u -B -m mod_benchmark.worker \
  --archive /Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/SW0250_mod_benchmark \
  --dataset multi_dsprites --mode zero_shot --seed 0
```

Seeds 1 and 2 use the same command with `--seed 1` and `--seed 2`. GPU placement is
provided by the scheduler's `CUDA_VISIBLE_DEVICES`; it is not a readout setting.
The source exports are `patch_final_exports/sw240_seed{0,1,2}.pt`, corresponding
to the `SW0249_matched35000_terminal` portable SW0240 endpoints.

There are **no additional readout CLI flags** in this worker. These settings are
fixed in `snn_kuramoto_bidirectional/sw240/readout.py` and its callees:

- Native grid 64x64; actual rollout 1,024 steps; settle 512.
- `spike_synchrony_components`: `synchrony_threshold=.5`, `min_group_size=32`,
  `settle=512`, `background='largest_component'`, `affinity_mode='spike'`,
  `spatial_grid_size=64`, using the actual component-spike trace.
- This supplies foreground support. The final object IDs come from
  `foreground_induced_merge(Q, A, coupling_gain, support, grid_shape=(64,64))`:
  signed pair energy from actual spike-derived Q and source graph A, positive-gain
  agglomeration restricted to four-connected foreground regions, preserving the
  foreground support exactly. There is no supplied ground-truth object count.
- The image-conditioned graph configuration includes top-k 512 and the fixed-8
  squared-geodesic operator. It is not a single image-independent graph.

Our final readout is therefore not the same as the spatial-sigma/threshold sweep
in this request. The signed partition stage is part of the evaluated model.

### 3. Input features

We did **not** replace the model encoder with
`input_encoder_multi_dsprites_k8.pt` or use the provided precomputed gamma.
The zero-shot condition retains the CLEVR-trained encoder and sensory projection.

The encoder is `ContextualFeatureEncoder(CNNFeatureEncoder(8,3,3,bias=True))`:

- Base 3x3 RGB-to-8 convolution: **224 parameters**.
- Additional context residual CNN: RGB-to-32 convolution, three 32-to-32
  convolutions with dilations 2/4/8, then a 1x1 32-to-8 output convolution.
- **29,128 encoder parameters in total**, not 224.
- Base and context features are added, standardized with the saved CLEVR feature
  mean/std, clipped to +/-3, and pooled to eight-channel native64 gamma.

The checkpoint's encoder features were trained with mask-derived supervision on
CLEVR. They are not equivalent to a per-dataset reconstruction-only encoder.

### 4. Measured groups per image

We re-read all **3,000 saved zero-shot masks**, verified each row's SHA and the
stacked prediction hash against the frozen protocol, and counted distinct positive
final label IDs per image. Background label 0 is excluded.

| Evaluation | seed0 | seed1 | seed2 | Three-seed mean |
| --- | ---: | ---: | ---: | ---: |
| TEST360 | 5.2306 | 6.6694 | 30.2000 | **14.0333** |
| TEST1000 | 6.2830 | 7.3760 | 29.6730 | **14.4440** |

These are **final groups per image**, not fragments per ground-truth object.
Our model also fragments, particularly seed2; its predictions are not consistently
2-3 objects per image. The difference from your reported 65.37 is informative,
but does not alone isolate encoder versus grouping causality: both differ between
our systems, as do the splits and source supervision.

### 5. Training origin and adaptation

The `?? checkpoint` condition transferred the three CLEVR-trained SW0240 seeds
**without any Multi-dSprites training**. Inference never receives a target mask.
However, the original checkpoint was mask-supervised: within-object frequency
teacher targets and a gamma anchor trained the encoder/sensory projection.
Not having a direct final-mask CE/Dice loss in SW0240 does not make it unsupervised.

The separate SW0250 `?? ??` condition uses target TRAIN IDs 0-5999, 6,000 images
x 10 epochs, batch16, 3,750 Adam updates, with mask-derived frequency targets and
gamma anchoring. Encoder/context and sensory projection learn; core/graph/readout
and saved feature statistics stay fixed. Its final TEST1000 FG-ARI is 0.770147.
These are the completed SW0250 results, not the later SW0251 residual-readout study.

The collapse diagnostic and frozen-graph recovery you report are useful evidence
for your setup. We have not reproduced that intervention here, so cannot attribute
our result to it. Likewise, grid64 removes Multi-dSprites quantization error, but a
single-object image can still receive FG-ARI 0 when the model fragments it. Our
TEST360 contains 86 single-object images; they are included by the unchanged
metric rather than omitted.
