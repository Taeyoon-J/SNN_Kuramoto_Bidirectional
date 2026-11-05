# PV2_0141 — how much of "all nine cells improve" is actually established

**Status** completed (analysis of data already in this repository; no GPU, no server)
**Effect on earlier results** narrows what PV2_0136 and `BENCHMARK_PLAN_v4.md` may claim.

## The question

Every number in PV2_0133–PV2_0138 is seed 0, so none has an error bar, and some of the gains
are tiny — Tetrominoes improved by **+0.0006** FG-ARI. Without a spread that figure is
unreadable. I have been reporting "all nine metric cells improve" as the generalisation result;
this asks how many of those nine are actually established.

## The measurement

Seed spread needs seeds we do not have. But the `fit5` runs recorded the **full per-epoch
validation curve for every arm**, which gives a different and still useful quantity: how much
the score moves when nothing about the model changes except where training happened to stop, or
which loss mixture was picked.

- **late-epoch spread** — std of validation FG-ARI over epochs 10–20 of the selected arm, after
  the curve has settled. Movement with no modelling content.
- **arm spread** — std across the five arms of their selected validation score.

A delta smaller than this is certainly not established. A delta larger than it still might not
be, because this is a **lower bound**: genuine seed variance adds the random encoder draw on
top, and we know that draw matters — the three CLEVR seeds' normalization gives feature std
**0.2478 / 0.1337 / 0.3533**.

## Result

| dataset | metric | reported Δ | late-epoch std | arm std | verdict |
|---|---|---|---|---|---|
| CLEVR | fg_ari | +0.0086 | 0.0012 | 0.0009 | **clears it** |
| | foreground_iou | +0.0285 | 0.0013 | 0.0141 | **clears it** |
| | matched_object_iou | +0.0298 | 0.0022 | 0.0056 | **clears it** |
| Tetrominoes | fg_ari | +0.0006 | 0.0058 | 0.0453 | inside the wobble |
| | foreground_iou | +0.0010 | 0.0005 | 0.0027 | inside the wobble |
| | matched_object_iou | +0.0015 | 0.0031 | 0.0258 | inside the wobble |
| Multi-dSprites | fg_ari | +0.0067 | 0.0013 | 0.0022 | **clears it** |
| | foreground_iou | +0.0036 | 0.0017 | 0.0049 | inside the wobble |
| | matched_object_iou | +0.0032 | 0.0014 | 0.0117 | inside the wobble |

**4 of 9 clear twice the wobble.** Tetrominoes clears none of it: its arm spread on FG-ARI is
0.0453, seventy-five times the +0.0006 we reported.

## What this changes

**"All nine metric cells improve" is true in sign and misleading as a claim.** What is
supportable from seed 0 is narrower:

- **CLEVR: established on all three metrics.** +0.0086 / +0.0285 / +0.0298 against wobble of
  order 0.001–0.014.
- **Multi-dSprites: FG-ARI established** (+0.0067 against 0.0022); the two IoU gains are not.
- **Tetrominoes: nothing established.** The head neither helps nor hurts there as far as this
  data can tell.

That is still a real generalisation result — a CLEVR-trained core plus a label-free head
improves grouping on a dataset it never trained on — but it rests on **CLEVR and
Multi-dSprites FG-ARI**, not on nine cells.

It also reframes what the seeds are for. They are not a formality to satisfy the contract:
**five of our nine headline cells are currently indistinguishable from noise**, and seeds are
the only thing that can settle them.

## Caveat on the method

Late-epoch and arm spread are not seed variance. They measure sensitivity to the selection,
which shares the same trained core; seed variance would also redraw the encoder and the whole
trajectory. So this understates the true noise, which makes the "inside the wobble" verdicts
safe and the "clears it" verdicts provisional.

## Reproduce

```bash
python3 collaborative_test/PV2_0141_how_much_is_noise/noise.py
```

Reads `PV2_0134_head_selection_and_transfer/diagnostics/fit5/*.json`, already in this
repository. Writes `noise.json`.
