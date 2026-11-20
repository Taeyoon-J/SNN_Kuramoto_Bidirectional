# PV2_0134 — what the head is worth, and what a new dataset costs

**Status** completed
**Parent** PV2_0133 (changes A and B)
**Compares against** 253's own partition on the same images, and Slot Attention's published
checkpoint on the same images through the same evaluation function
**Evaluation** peer patch contract, `evaluate_patch_masks`, grid 64, ceiling 1.0
**Seeds** one. Every number here is seed 0 and is a candidate, not a result.

> ## CORRECTION, see PV2_0135
>
> **Change C's result below is withdrawn.** The second-seed cache it relied on contains no
> second seed: `features_b` and `gamma_b` are bitwise identical to the first pass, `regions_b`
> is `regions_all` rather than a second-seed 253 partition, and the 253 rollout turns out to be
> fully deterministic in `inference_seed` — three seeds give bitwise identical gamma, q,
> adjacency, spikes and component spikes. C's objective therefore compared the head against the
> same tensor, so the symmetric KL was identically zero and measured nothing. Seed invariance is
> not an available signal for this core; whether it would help a stochastic rollout is untested.
>
> Everything else in this test stands, because nothing else read that cache. In particular the
> Tetrominoes **−0.1141 Foreground IoU** observation that motivated rule B′ came from C's
> **epoch 0** — the head before any update, with no second-seed tensor involved — so B′'s
> motivation and B′ itself are unaffected. So are change A's isolation, the transfer result, the
> recipe result, the Slot Attention rule sweep and the three-dataset table.

## Hypothesis

PV2_0133 opened the head direction: with change A (background as an ordinary candidate) and
change B (arms selected on validation FG-ARI), nine of nine metric cells improved across
three datasets. This test asks the three questions that leaves.

1. **C.** Does cross-seed agreement give the head an unsupervised objective that beats the
   reconstruction losses? *(The premise was false — see the correction above and PV2_0135.)*
2. **Transfer.** Does one head, fitted on CLEVR, carry to the other datasets unchanged?
3. **Cost.** If it does not, what does a new dataset actually cost — and is any of that cost
   labels?

## What happened

### Change C — withdrawn, but it exposed a real defect in change B

> Everything in this subsection about C is superseded by PV2_0135: the two "seeds" were the same
> rollout, so the objective measured nothing and the collapse described below is drift under a
> term with no content. The B-rule defect it surfaced is independent of that and is real.

The idea: the rollout is stochastic in its inference seed, so one image under two seeds gives
two activity traces of the same scene. A grouping that tracks real objects should survive the
seed change; one that tracks rollout noise should not. No labels, and no colour-reconstruction
shortcut.

It has a trivial minimum. Assign every node to one candidate and both seeds agree perfectly,
so the symmetric KL reaches zero while the partition carries nothing. The first sweep showed
exactly that — every arm reached the **identical** validation FG-ARI whatever its rgb and tv
weights:

| dataset | 253 validation FG-ARI | every C arm |
|---|---|---|
| CLEVR | 0.8501 | 0.4954 |
| Tetrominoes | 0.8745 | 0.6817 |

Two collapse guards were added — maximise the entropy of the marginal over candidates, or
hold that marginal near the 253 partition's own size distribution — together with epoch 0 as
a selection candidate, so the rule can decline to move at all. With the guards, **no arm on
any dataset beat its own epoch 0.** Cross-seed agreement does not help this head.

What it did find is a defect in change B. Selecting on validation FG-ARI alone has no opinion
about how much of the image is called foreground, and on Tetrominoes it chose a head at

```
253        0.8580   0.7885   0.8173
head C     0.8590   0.6744   0.8214      +0.0010  -0.1141  +0.0041
```

**−0.1141 Foreground IoU bought with +0.0010 FG-ARI.** FG-ARI may rise at some cost in
Foreground IoU; a tenth of it is not that cost.

### B′ — the corrected selection rule

B′ takes the highest validation FG-ARI among the (arm, epoch) pairs whose validation
Foreground IoU is within 0.01 of 253's, with epoch 0 among the candidates. The floor is a
stated tolerance applied on validation, not a quantity tuned on the evaluation split, and
both rules' choices are recorded so the correction's cost stays visible.

It costs nothing where it is not needed and does real work where it is:

| dataset | B alone would take | B′ takes |
|---|---|---|
| CLEVR | epoch 11, val 0.8605 / 0.8011 | epoch 11, val 0.8605 / 0.8011 — same |
| Tetrominoes (`local_rgb`) | epoch 0, val 0.8754 / **0.6717** | epoch 14, val 0.7732 / 0.7858 |
| Multi-dSprites (`local_rgb`) | epoch 0, val 0.6965 / 0.8404 | epoch 0 — same, 6/21 pass the floor |

The epoch-0 rows also settle an honesty question worth asking of PV2_0133: the **untrained**
head gets *higher* FG-ARI than the trained one and far worse Foreground IoU (−0.1141 on
Tetrominoes, −0.0015 on CLEVR). So the A+B gain is genuinely produced by training, not by the
head's architecture alone.

### The head does not transfer, the core does

One head fitted on CLEVR, loaded unchanged — no refitting, no per-dataset arm or epoch:

| dataset | | fg_ari | fg_iou | obj_iou | changed |
|---|---|---|---|---|---|
| Tetrominoes | 253 | 0.8580 | 0.7885 | 0.8173 | |
| | CLEVR head | 0.8259 | 0.6817 | 0.7824 | 8.39% |
| | | **−0.0321** | **−0.1068** | **−0.0349** | |
| Multi-dSprites | 253 | 0.6974 | 0.8177 | 0.7628 | |
| | CLEVR head | 0.6597 | 0.8053 | 0.7531 | 1.06% |
| | | **−0.0377** | **−0.0124** | **−0.0097** | |

It hurts all three metrics on both datasets. The head is a per-dataset component. This is not
a problem for the generalisation claim — it is the architecture the claim already assumed:
the core is trained once on CLEVR and reused, and the head is the cheap part that gets
swapped. But it has to be stated the right way round. **Generalisation lives in the core.**

### What a new dataset costs

Refitting the head per dataset is cheap in compute — about a minute per arm over 256 cached
images — but B′ picks the arm and epoch on that dataset's validation FG-ARI, and FG-ARI needs
labels. The training losses are unsupervised; the selection is not, and calling the result
label-free would be wrong.

So the two were separated. The **recipe** — which arm, how many epochs — was fixed once on
CLEVR (`local_rgb_tv`, 11 epochs) and reused verbatim, with nothing from the new dataset's
labels entering at any point:

| dataset | method | fg_ari | fg_iou | obj_iou | changed |
|---|---|---|---|---|---|
| CLEVR | 253 | 0.8421 | 0.7545 | 0.6410 | |
| | head, B′ refit | **0.8507** | **0.7830** | **0.6709** | 2.06% |
| | head, CLEVR recipe, no labels | **0.8507** | **0.7831** | **0.6710** | 2.07% |
| Tetrominoes | 253 | 0.8580 | 0.7885 | 0.8173 | |
| | head, B′ refit | **0.8586** | **0.7895** | **0.8187** | 0.57% |
| | head, CLEVR recipe, no labels | 0.8015 | **0.7924** | 0.7962 | 1.80% |
| Multi-dSprites | 253 | 0.6974 | 0.8177 | 0.7628 | |
| | head, B′ refit | **0.7041** | **0.8213** | **0.7660** | 0.33% |
| | head, CLEVR recipe, no labels | **0.6986** | **0.8224** | **0.7657** | 0.11% |

The label-free recipe reproduces the refit result on CLEVR to four decimals and improves all
three metrics on Multi-dSprites. It fails on Tetrominoes, losing 0.0565 FG-ARI, and the reason
is visible in the curves: Tetrominoes selects `local_rgb_trust` at **epoch 2**, and eleven
epochs of edge-weighted total variation over-smooths sharp blocks on a black ground. The
failure is the epoch count, not the losses.

So the honest cost of a new dataset is: 256 unlabelled images and about a minute, **plus a
labelled validation split to stop at the right epoch** — unless a label-free stopping rule can
be found. That is the next experiment, and it is well posed: the arm choice barely matters
(four of five arms land within 0.001 of each other on Multi-dSprites validation), while the
epoch choice is what decides Tetrominoes.

### Slot Attention: its score is not a reading artifact

Slot Attention at grid 64 calls 85.1% of the image foreground where the truth is 21.2%, which
is most of why its Foreground IoU reads 0.2022. Slot ids are arbitrary, so which slot counts
as background is a free choice, and a wrong choice would be our error, not the model's.

Four rules were fixed before looking at any number and chosen on the **validation** split:
`global_largest`, `image_largest`, `border_largest`, `border_union` (the last merges every
slot with over half its patches on the frame border, the only one that can repair a split
background).

**All four give the identical FG-ARI** — 0.4061 on validation, 0.3747 on fixed320. The rules
only relabel or merge slots lying on the border, and those patches are background in the
truth, which FG-ARI excludes by definition. The tie therefore breaks on Foreground IoU, the
choice most generous to the baseline, selecting `border_largest`:

| rule | fixed320 fg_ari | fg_iou | obj_iou | predicted foreground |
|---|---|---|---|---|
| global_largest (what we had used) | 0.3747 | 0.2051 | 0.1101 | 0.8447 |
| image_largest | 0.3747 | 0.2029 | 0.1122 | 0.6903 |
| **border_largest** (selected) | **0.3747** | **0.2219** | **0.1165** | 0.7339 |
| border_union | 0.3747 | 0.2034 | 0.1125 | 0.6957 |

The background rule is worth at most +0.017 Foreground IoU and nothing at all on FG-ARI.

The diagnostic that explains it: this checkpoint's slot share, over all patches, is
`0.156 0.147 0.141 0.148 0.133 0.131 0.144` with **6.99 of 7 slots used per image**. It is
not discovering objects on these frames, it is tiling. Crucially, the **old** grid-16 run with
the published centre-crop preprocessing shows the same thing — `0.151 0.136 0.142 0.142 0.138
0.149 0.143`, 6.93 slots per image — so this is not caused by our frame choice. Its own
recorded `metrics.json` reads fg_ari 0.6195, fg_iou 0.1241, obj_iou 0.0920 on 300 images at
grid 16, nowhere near the 0.8102 that has been quoted from an older split.

## Side by side on the same 320 images

Same images (fixed320, ids 1320–1639), same `evaluate_patch_masks`, same targets, grid 64.

| | fg_ari | fg_iou | obj_iou |
|---|---|---|---|
| Slot Attention (single published checkpoint, best of four background rules) | 0.3747 | 0.2219 | 0.1165 |
| 253 core alone | 0.8421 | 0.7545 | 0.6410 |
| 253 core + head (B′) | **0.8507** | **0.7830** | **0.6709** |

## Limitations

- **One seed.** Everything above is seed 0. Nothing here is a result until seeds 1 and 2 exist
  for the 253 core at grid 64, which is the first item in `BENCHMARK_PLAN_v3.md` and needs
  roughly 18.5 h of rented 4090 time per seed.
- **Slot Attention is one published checkpoint**, never a 3-seed training mean, and it was
  trained on the original CLEVR render while this split is `clevr_with_masks` — a different
  render the authors themselves warn about. The 6.99-slots-per-image tiling is consistent with
  a failed transfer. **This is the largest fairness gap in the benchmark**: beating a model
  that is failing to transfer is not the same as beating the model. Retraining Slot Attention
  on `clevr_with_masks` should precede any headline claim.
- The head's gain on Tetrominoes is +0.0006 / +0.0010 / +0.0015 — real in sign on all three,
  but far inside what one seed can establish.
- The 0.01 Foreground IoU floor in B′ is a stated tolerance. It was not tuned, but it was also
  not derived; a different tolerance would select differently on Tetrominoes' `local_rgb`.

## Next

1. A label-free stopping rule for the head, so a new dataset costs no labels at all. The arm
   choice is nearly free; only the epoch matters. A convergence criterion on the unlabelled
   images — stop when the predicted partition stops moving between epochs — is the obvious
   candidate and is testable against the three curves already saved here.
2. Seeds 1 and 2 of the 253 core at grid 64 on CLEVR. Nothing above is a result without them.
3. Retrain Slot Attention on `clevr_with_masks` so the reference column is a transfer-fair
   comparison.
