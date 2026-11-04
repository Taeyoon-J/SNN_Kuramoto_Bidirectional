# Benchmark plan v3 — what to run, what to claim, and what to say about Slot Attention

2026-11-04. Supersedes `BENCHMARK_PLAN_v2.md`. Every number here was measured today; the
measurements are in `PV2_0133_readout_and_benchmark/`.

## The decision this plan turns on

Slot Attention publishes **pixel-level** FG-ARI: Tetrominoes 99.5, Multi-dSprites 91.3,
CLEVR6 98.8. We predict one label per grid cell. There are two ways to put those side by
side, and they lead to opposite conclusions.

**Pixel protocol** — expand our 64×64 prediction back to 128×128 and score against the
original pixel truth. Then the grid itself caps us, and the cap is below their numbers:

| dataset | native pixels | our ceiling | their published |
| --- | --- | ---: | ---: |
| CLEVR | 128×128 | 0.8988 (grid 64) | 0.988 |
| Tetrominoes | 35×35 | 0.9352 (grid 32; grid 64 impossible) | 0.995 |
| Multi-dSprites | 64×64 | **1.0000** | 0.913 |

A perfect model loses on two of three. Grid 128 would fix CLEVR and is not available:
`direction_learner` would be 16,384² = 268M parameters, sixteen times grid 64's memory.

**Patch protocol** — quantise the truth to the same grid and score there. **The ceiling is
1.0 on every dataset by construction**, because the target *is* the quantised truth. The
grid stops being a handicap on either side, and the comparison becomes about the models.

**Use the patch protocol, and run Slot Attention through it.** This is what the evaluation
contract already requires: *"Slot Attention masks go through the same patch conversion and
the same evaluation function."* It is not a convenience — it is the only form in which the
two systems are measured on the same thing.

## What that comparison says

CLEVR, fixed320 (ids 1320–1639), 320 images, grid 64, `evaluate_patch_masks`, identical
targets. One seed on our side; one published checkpoint on theirs.

| | fg_ari | fg_iou | obj_iou | predicted foreground |
| --- | ---: | ---: | ---: | ---: |
| **253 + learned head** | **0.8503** | **0.7837** | **0.6706** | — |
| 253 | 0.8421 | 0.7545 | 0.6410 | 0.2001 |
| 252, supervised | 0.5213 | 0.2115 | 0.5184 | — |
| Slot Attention | 0.3808 | 0.2022 | 0.1053 | **0.8510** |
| truth foreground | | | | 0.2115 |

Slot Attention's number is not a grid artefact. Holding the split fixed and varying only
the grid gives 0.4153 / 0.3912 / 0.3808 at 16 / 32 / 64 — **0.035 across a 4× change**. At
every grid it calls 85% of the image foreground against a 21% truth, and the slot its
background rule picks covers 14.9% of patches. Its background is split across several
slots, so "the largest slot is background" cannot find it.

### What must be written next to that table, every time

- The split is `clevr_with_masks`, **not the render Slot Attention trained on**, and it is
  CLEVR10 rather than the CLEVR6 the 98.8 was measured on. Their README says the two
  renders are not comparable.
- `ckpt-500` is **one published checkpoint**, never a 3-seed mean.
- **We trained on this split.** The defensible sentence is *"on identical data and
  identical scoring we lead"*, not *"we beat Slot Attention"*.
- Aligned preprocessing (`--preprocess resize`) is used, which **favours** Slot Attention:
  `PV2_0090` measured the published centre crop as costing it 0.0513.

## The three-dataset row, and why generality is the claim

Core trained on **CLEVR only** and frozen. Head retrained per dataset on 256 unlabelled
images of that dataset, six minutes each, selected on that dataset's validation FG-ARI.

| dataset | 253 | 253 + head | delta |
| --- | ---: | ---: | ---: |
| CLEVR | 0.8421 / 0.7545 / 0.6410 | **0.8503 / 0.7837 / 0.6706** | +0.0082 / +0.0292 / +0.0296 |
| Tetrominoes | 0.8580 / 0.7885 / 0.8173 | **0.8586 / 0.7895 / 0.8187** | +0.0006 / +0.0010 / +0.0015 |
| Multi-dSprites | 0.6974 / 0.8177 / 0.7628 | **0.7041 / 0.8213 / 0.7660** | +0.0067 / +0.0036 / +0.0032 |

Nine of nine cells improve, changing 0.33–2.04% of patches. **The core never saw
Tetrominoes or Multi-dSprites.** That is the claim worth making: one core, trained once,
with a readout that adapts to a new dataset in minutes and without labels.

Multi-dSprites deserves its own sentence. Our records had it as the standing weak point —
0.2817 in our own pipeline, 0.2789 ± 0.0813 over three seeds in the peer engine at grid 32
*trained on Multi-dSprites*. A CLEVR-only grid-64 core reaches **0.6893 on the same pixel
protocol and the same 1000 TEST images**, never having seen the dataset. Whether the gain
is the grid or the training budget is not yet separated.

## What to run next, in order

1. **Seeds 1 and 2 for 253 at grid 64 on CLEVR.** Nothing above is a result at one seed,
   and the standing rule judges on 3-seed means. ~18.5 h each on a rented RTX 4090, ~$8;
   two in parallel is one wall-clock day.
2. **Slot Attention's background rule.** Our runner picks the slot covering the most
   patches; the peer's Slot64 contract picks the slot covering most of the one-pixel
   border. With a fragmented background these differ, and that is a live candidate for part
   of the 0.8102 → 0.4153 gap. Costs nothing: the predictions are saved.
3. **Train the core on Multi-dSprites at grid 64.** Zero-shot already reaches 0.6893 where
   the grid-32 trained model got 0.2789. This separates grid from training data, and
   Multi-dSprites is the one dataset whose pixel ceiling is 1.0000 — the only place a claim
   against their *published* number is possible.
4. **Tetrominoes and Multi-dSprites heads at 3 seeds**, once 1 and 3 land.

## What not to do

- **Do not run SW0261 from scratch.** Its core objective is identical to 253's
  (`L_phase + 5·L_positive_component_spike_Q`, same coefficients), so training it
  reproduces our core; and its head branch, tested four ways on our own better core, is
  what sections 6–7 of `PV2_0133` measure. The useful part of 261 — the context readout —
  is already ported and is what "253 + head" means above.
- **Do not put a patch number next to a published pixel number.** That pairing produced
  every confused comparison this project has made.
- **Do not report 252 as a supervised ceiling.** Its core never learned (`phase_primary`
  moved −0.1% over 6,250 updates against 253's −46%) because the teacher term reaches gamma
  directly while the core objective must cross a 1,024-step rollout with only 64
  differentiable steps. It is a measurement of that defect, not of supervision.
