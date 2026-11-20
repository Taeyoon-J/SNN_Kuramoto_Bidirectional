# PV2_0121 — Tier 1, first number: the architecture works outside CLEVR

Status: **Tetrominoes completed, Multi-dSprites training.** Seed 0, grid 32, 100 validation
images, each dataset's own patch contract. No gate was registered: this establishes that
the model trains and scores on new data at all.

## Result

| Tetrominoes, grid 32, seed 0 | |
| --- | ---: |
| **FG-ARI** | **0.9445** |
| foreground IoU | 0.6814 |
| matched-object IoU | **0.7925** |
| groups per image | 10.86 |
| predicted foreground | 0.2984 |

The spike path survived the move to a new dataset, which was the open question —
`PV2_0094` showed an absolute membrane threshold does not transfer across oscillator
counts and nothing said it transfers across datasets. `--membrane-threshold-mode
population` was carried for that reason and it held.

## Three reasons this number is not what it looks like

**1. It is not comparable to our CLEVR 0.7603.** Tetrominoes is 35×35 with three objects
and 75.5% background; CLEVR is 240×320 with up to ten objects and 87.9% background. Nor is
it comparable to the published synchrony family — CAE 0.04, CAE++ 0.27, CtCAE 0.54,
SynCx 0.59, Rotating Features 0.65 — because those are **CLEVR**, and where SynCx reports
0.89 on Tetrominoes it is under the **pixel** protocol while this is the patch contract.

**2. Predicted foreground is 0.2984 against a true 0.2363 — 26% over.** `PV2_0119`
measured exactly what that buys: FG-ARI rises for free when foreground is over-predicted,
and foreground IoU pays for it. At 0.6814 it is paying.

**3. One seed, 100 images.** Not a promotion.

## What does survive the caveats

**Matched-object IoU is 0.7925.** That column cannot be bought by over-predicting — it
divides by the true object count — and it is above our best CLEVR figure of 0.7017. The
model is finding the objects, not just painting foreground.

## Next

Multi-dSprites, then the conversion to the pixel protocol, which is the only form in which
these can sit beside CAE, Rotating Features, SynCx and AKOrN. `BENCHMARK_PLAN.md` already
settled that the paper table goes pixel-protocol at grid 32 or finer, and measured why
these datasets are where that costs us almost nothing: a grid-32 cell is 1.09 px on
Tetrominoes and 2.00 px on Multi-dSprites against 7.50 × 10.00 on CLEVR.

## Four evaluation failures, and the check that should have come first

The evaluation failed four times before producing this number, each time on a field my
benchmark artefacts did not carry: `--expect-contract`, then `names`, then `patch_size`,
then the manifest's `splits`. Each was found by running the evaluator and reading the
traceback.

They were one question — does what I built match the schema the evaluator expects — and
`preflight.py` now asks it before anything runs. Both datasets pass 8 of 8.

The `splits` one was not mere convenience: the evaluator compares the targets' `names` to
`manifest["splits"][split]`, which is how the γ tensor and the ground truth are verified to
describe the same images in the same order. My manifests carried only `ranges`, so that
check could not run at all.
