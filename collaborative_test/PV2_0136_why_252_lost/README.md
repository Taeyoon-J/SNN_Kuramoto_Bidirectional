# PV2_0136 — why 252 lost to 253 on the same data

**Status** completed
**Type** analysis of existing runs; no new training
**Sources** both runs' `training_report.json` and `training.jsonl` on frontier, `curve252.json`,
and `vendor/full_benchmark/loss.py`. Every number below is read from those files.

Three questions were asked of this comparison. They have one answer between them.

## 1. The budgets were identical, so the data is not the explanation

From each run's own `training_report.json`:

| | 252 | 253 |
|---|---|---|
| updates | 6,250 | 6,250 |
| images seen | 100,000 | 100,000 |
| unique TRAIN images | 10,000 | 10,000 |
| epochs | 10 | 10 |

Same count, same split, same schedule. Whatever separates them is not how much data they saw.

## 2. 253's core learned and 252's did not

Both runs optimise the same unsupervised core objective. Comparing the mean of the first 50
updates against the mean of the last 50, from `training.jsonl`:

| component | 252 | 253 |
|---|---|---|
| `phase_primary` | 7.2389 → 7.0511 (**−2.6%**) | 4.6355 → 2.4556 (**−47.0%**) |
| `positive_spike` | 15.6905 → 10.4814 (**−33.2%**) | 15.3876 → 2.5839 (**−83.2%**) |
| `gradient_norm` | 47.96 → 247.27 (**+416%**) | 6.08 → 6.27 (+3.2%) |

253 drove its phase objective down by nearly half and its spike objective by five sixths, with
a gradient norm that barely moved — a converging run. 252 moved both barely, while its gradient
norm more than quintupled — a run pulling against itself.

The parameters say the same thing. Mean relative change over the 17 core tensors:

| | 252 | 253 |
|---|---|---|
| core | **0.0700** | **0.1124** |
| encoder | 0.0066 (12 tensors) | 0.0083 (2 tensors) |
| head | **1.7740** (max 5.6567) | no head |

And the gradient each part received, summing `active_tensor_gradient_coverage`:

| | 252 | 253 | ratio |
|---|---|---|---|
| core | 3.408e+05 | 9.140e+04 | 252 got **3.7×** more |
| encoder | 1.862e+07 | 2.884e+05 | 252 got **65×** more |
| head | 1.426e+05 | 0 | — |

**252's core received 3.7 times the gradient and moved 0.62 as far.** Its encoder received 65
times the gradient and moved *less* in relative terms. More gradient producing less motion is
cancellation: the terms are pulling in opposite directions and the sum is small. The rising
gradient norm is the same fact seen from the other side.

Meanwhile the one thing that did move is the head — 177% mean relative change, one tensor by
566% — while the core it sits on moved 7%.

## 3. Why supervision made it worse, not better

The mechanism is in `vendor/full_benchmark/loss.py`. Supervision enters 252 by exactly two
routes, and neither one can improve the thing that is scored.

**Route A — the mask teacher, which is detached** (`loss.py:63-67`):

```python
features = activity_features(output['trace'])[0]
if model.variant == 252: features = features.detach()
regions  = regions_from_output(model, output)
head_loss = mask_objective(model.head(features, regions), labels[0])
```

`features.detach()` severs the gradient between the head and the trace. The mask labels train
the head and **cannot reach the core at all**. That term improved by 52.9% (`final_mask`
9.1696 → 4.3186), which is precisely why the head moved 177%: all of it went there. At
evaluation the score comes from the core's spikes, so this improvement is spent on a component
whose gradient never touched what is measured.

**Route B — the gamma teacher, which got worse** (`loss.py:58-61`):

```python
supervised, detail = teacher_objective(output['gamma'],
    effective_frequency(model.wrapped.core, output['gamma']), teacher.targets(model, rgb, labels))
base = base + supervised
```

This one *does* reach the core. It went the wrong way:

| | first 50 | last 50 | change |
|---|---|---|---|
| `gamma_anchor` | 0.00016 | 0.01591 | **+9,907%** |
| `frequency_denoising` | 0.11854 | 0.13893 | +17.2% |

The term carrying label information into gamma ended **62× worse** than it started. It did not
fail to help; it actively degraded while consuming gradient the core objective needed.

So the answer to "supervised yet lower than unsupervised" is structural, not a tuning accident:
**the labels reach only a readout that is discarded at scoring time, and the gradient they do
inject into the shared trunk fights the objective that actually shapes the spikes.** 252 bought
a better head on a worse core, and the score is read from the core.

## 4. The symptom in the evaluation curve

`curve252.json`, five checkpoints across the run:

| checkpoint | updates | fg_ari | **fg_iou** | obj_iou |
|---|---|---|---|---|
| u3000 | 3,000 | 0.4454 | **0.2021225** | 0.4662 |
| u4000 | 4,000 | 0.4675 | **0.2021225** | 0.4787 |
| u5000 | 5,000 | 0.5010 | **0.2021225** | 0.4953 |
| u6000 | 6,000 | 0.5166 | **0.2021225** | 0.5101 |
| u6250 | 6,250 | 0.5239 | **0.2021225** | 0.5172 |

Foreground IoU is **bitwise identical at every checkpoint**. That is not a plateau, it is a
constant: the readout designated no background, so `prediction != 0` covered the whole image
and the metric collapsed to the truth's foreground fraction on those images. It would read the
same for an untrained model, or for a model that output a single label everywhere.

This is the same degeneracy change A was introduced to fix in the head (PV2_0133/PV2_0134):
with the all-ones partition as the candidate set, Foreground IoU is pinned at 0.2115 and
training moves it `+0.0000`; with a real background candidate it reaches 0.7830.

So FG-ARI rising from 0.4454 to 0.5239 while Foreground IoU never moves at all is not partial
progress. Two of the three metrics were improving inside a segmentation that never separated
figure from ground.

## Caveat on the headline comparison

252's curve was scored on **160** fixed320 images under the `historical_fixed64_patch` contract
(`curve252.json:_note`), while 253's 0.8421 / 0.7545 / 0.6410 is **320** images under the patch
contract at grid 64 with `min_group_size` 8. Those two numbers should not be subtracted from
each other. The diagnostics above are the comparable evidence: same code, same budget, same
logging, read from both runs' own files.

## The universal label-free head recipe, found in the same round

Separately from the 252 analysis, this round answered the cost question PV2_0134 left open.
B′ independently chose **arm `local_rgb_trust` at epoch 2** on both Tetrominoes and
Multi-dSprites validation. Fixing that single recipe and refitting on each dataset's
**unlabelled** images only improves all nine metric cells — though `PV2_0141` shows only four
of the nine clear the measurable wobble, so what is *established* is CLEVR's three metrics and
Multi-dSprites' FG-ARI, not all nine:

| dataset | 253 | + head, no labels from this dataset | delta |
|---|---|---|---|
| CLEVR fixed320 | 0.8421 / 0.7545 / 0.6410 | **0.8468 / 0.7574 / 0.6456** | +0.0047 / +0.0029 / +0.0046 |
| Tetrominoes test | 0.8580 / 0.7885 / 0.8173 | 0.8586 / 0.7895 / 0.8187 | **withdrawn (PV2_0146)** — head seed 0 only; over 5 head seeds −0.0020 / +0.0003 / −0.0008 |
| Multi-dSprites test | 0.6974 / 0.8177 / 0.7628 | **0.7041 / 0.8213 / 0.7660** | +0.0067 / +0.0036 / +0.0032 |

The recipe was chosen on the three **validation** splits, which the contract permits; no
evaluation split took part. The per-dataset B′ refit still scores higher on CLEVR (0.8507 vs
0.8468) but needs CLEVR validation labels, so it is an upper bound with a labelled cost, not the
headline. This is the configuration `BENCHMARK_PLAN_v4.md` asks the benchmark to report.

Two label-free rules for choosing the epoch *automatically* were also tried and both failed.
Cross-seed agreement is unavailable (PV2_0135). **Stability** — the fraction of patches whose
predicted label changes between epochs, on unlabelled training images — settles to ~0.003 by
epoch 2 on CLEVR while B′'s epoch there is 11, and the curve is not monotone, so no threshold
recovers it. Fixing the recipe is what works; deriving the epoch per dataset does not.

## What this implies for the design

1. **A supervised term that reaches only a detached head is not supervision of the model.** If
   the score is read from spikes, a loss must have a gradient path to the state updates that
   produce spikes, or it is training something else.
2. **Watch the gradient norm's trend, not its value.** 253 held +3.2% over 6,250 updates; 252
   rose 416%. That divergence was visible long before the final score.
3. **Parameter movement per unit of gradient is the cancellation detector.** 3.7× the gradient
   and 0.62× the motion identified the conflict without needing the evaluation at all.
4. **A constant metric is a bug report.** Foreground IoU identical to seven decimals across
   five checkpoints should stop a run, not be read as a plateau.
