# PV2_0145 — what "beating Slot Attention" can and cannot mean

**Status** completed. Analysis of published tables already in this repository plus the
PV2_0138 numbers. No GPU.
**Why** the standing goal says Slot Attention must be beaten. That sentence has **three**
different meanings and we satisfy only one of them. Keeping them apart is the difference
between a defensible claim and one that gets rejected on sight.

## The problem, already documented

`BENCHMARK_PLAN.md` §0 states it:

> There are two incompatible conventions both called "FG-ARI", and we are using the rarer one.
> Slot Attention reports ARI-FG **98.8** on CLEVR6; SynCx reports ARI-FG **0.59** on CLEVR.
> Both are "foreground ARI". They differ because of the data variant, the model class and —
> decisively — because the field scores **per pixel**, while `evaluation_contract.md` scores
> **per patch** and forbids expanding a predicted mask to pixels.

So "we beat Slot Attention" needs saying in one of three precise forms.

## Sense 1 — Slot Attention's published numbers. **We lose, and it is not a fair comparison.**

| Slot Attention, arXiv 2006.15055 | ARI-FG |
|---|---|
| Tetrominoes | **99.5 ± 0.2** |
| Multi-dSprites | **91.3 ± 0.3** |
| CLEVR**6** | **98.8 ± 0.3** |

Our best patch-protocol numbers are 0.8580 / 0.6974 / 0.8421. We do not come close, and
`BENCHMARK_PLAN.md:54` already concluded that **"Slot Attention's 98.8 on CLEVR6 is out of
reach."** Those rows use per-pixel scoring, decoder-based reconstruction, and easier variants
(CLEVR**6** means at most six objects). A table placing our number beside 98.8 would be wrong,
and the plan says so.

**This sense should never be claimed.**

## Sense 2 — the published checkpoint on our images, our protocol. **We win decisively.**

Same 320 CLEVR images, same `evaluate_patch_masks`, same targets, its background rule chosen on
validation out of four fixed in advance:

| | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|
| Slot Attention, published checkpoint | 0.3747 | 0.2219 | 0.1165 |
| **ours** (253 core + label-free head) | **0.8468** | **0.7574** | **0.6456** |

Honest, but weak: that checkpoint was trained on the original CLEVR render and **tiles rather
than segments on ours** — 6.99 of 7 slots on every frame with near-equal shares. We are beating
a failed transfer. Seed 0 only.

## Sense 3 — Slot Attention retrained on our render. **Not done.** `PV2_0139` holds the code.

This is the only version that would support the goal's sentence, and it is the open item.

## Where we actually stand against the field that uses a comparable protocol

The synchrony family scores per pixel on the same small suite, and for **Multi-dSprites our
grid-64 number *is* the pixel number** — `Dataset.labels` and `Dataset.pixels` are the identical
64×64 array, verified, so the ceiling of 1.0 is real and the comparison is legitimate.

| model (unsupervised) | Tetrominoes | **Multi-dSprites** | CLEVR |
|---|---|---|---|
| CAE | 0.00 | 0.05 | 0.04 |
| CAE++ | 0.78 | 0.51 | 0.27 |
| CtCAE | 0.84 | 0.56 | 0.54 |
| Rotating Features | 0.42 | **0.84** | 0.65 |
| SynCx | **0.89** | 0.82 | 0.59 |
| **ours, CLEVR-trained core, zero-shot** | 0.9280¹ | **0.6974** | 0.6668² |

¹ one seed, validation, grid 32. ² three seeds ± 0.0506, grid 32.

**Multi-dSprites is the update.** `BENCHMARK_EVALUATION.md` had recorded it as *"loses, at a grid
where winning was impossible"* (0.2080 at grid 32, whose ceiling is 0.5619). At grid 64 the
CLEVR-trained core reaches **0.6974**, which places us **above CAE, CAE++ and CtCAE and below
Rotating Features and SynCx** — fourth of six, where before we were last. And it is **zero-shot**:
every other row trained on Multi-dSprites.

That last point is the one worth pressing. Our 0.6974 also lands on the peer's **CLEVR
mask-supervised** zero-shot numbers (0.6833 / 0.6979, `PV2_0128`), without any supervision.

## What the benchmark should therefore claim

1. **Never** put our number beside Slot Attention's published 98.8 / 99.5 / 91.3. Different
   protocol, different data variants; `BENCHMARK_PLAN.md` already rules it out.
2. Against the **synchrony family** on Multi-dSprites, where our protocol coincides with theirs,
   state the position plainly: 0.6974 zero-shot, fourth of six, above CtCAE, below SynCx and
   Rotating Features — and note that it is the only zero-shot row in the table.
3. Against **Slot Attention**, state sense 2 exactly as it is: we beat the published checkpoint
   transferred to our render, and that checkpoint is failing to transfer. Then either produce
   sense 3 or say it is missing.
4. Everything above is **seed 0** except CLEVR's 0.6668 ± 0.0506.

## Limitation

The Tetrominoes 0.9280 and CLEVR 0.6668 rows are grid-32 numbers from earlier records and are
**not** on the patch-64 protocol the recent tests use; they are carried here because they are
what the published comparison was made against. Only the Multi-dSprites row is a clean
protocol match, and that is precisely why it is the one to lead with.
