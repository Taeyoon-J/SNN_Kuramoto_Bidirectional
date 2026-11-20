# PV2_0083 — the error is between patches, not inside them

**completed. Closes the input-resolution hypothesis and locates the error precisely.**
Parent `PV2_0082`. Diagnostic on the published configuration's own predictions; no
training, no new model.

## The hypothesis

`PV2_0082` left a fact I had not followed: **17.3% of foreground patches contain two or
more objects.** A perfect pixel segmenter resolves those because it sees sub-patch
structure, and our encoder pools to 16x16 before anything else happens, so sub-patch
detail is destroyed at the input and the dynamics never have it.

If the error concentrated there, the limit would be **input resolution** — which none of
the 27 closed routes touched. Every feature route changed *what* the eight channels
contain at this resolution, never whether they carry sub-patch structure. The fix would
be a finer encoder pooled into the same 256 oscillators, keeping SNN + Kuramoto intact.

## Result: it does not

Per-patch agreement with the target, after mapping each predicted group to the target
label it most often coincides with (predicted ids are arbitrary):

| foreground patches | agreement | share | share of total error |
| --- | --- | --- | --- |
| all | 0.7636 | 9,032 | — |
| **holding one object** | **0.7979** | 82.7% | **70.7%** |
| **holding two or more** | **0.5999** | 17.3% | 29.3% |

Straddling patches are genuinely harder — **+0.198** — but they carry only **29.3%** of
the error. **70.7% of it is on patches containing exactly one object**, where there is no
sub-patch ambiguity at all and the model still puts the patch in the wrong group.

So input resolution is not the limit, and a finer encoder would address at most a third
of the error and only partly. **The merges are between patches, not inside them.**

## What it confirms

Even on completely unambiguous patches the model mis-groups **20%**. That is a grouping
failure, not a resolution failure, and it matches the account in `ROUTE_MAP_0.9.md`
exactly: whole objects are merged (`PV2_0053`: 38.8% of images hold a pair whose centroids
are closer than the objects are internally spread), so every patch of a merged object
lands in the wrong group regardless of how cleanly that patch is resolved.

Together with `PV2_0082` this removes the last two explanations that were not the model
itself:

- the **metric** is not the obstacle — a perfect segmenter scores 0.9756-1.0000 on
  contract v1 under any defensible rule;
- the **input resolution** is not the obstacle — 70.7% of the error is on patches a
  finer encoder would not help with.

The gap from 0.7603 to 0.9 is the grouping, and the grouping fails by merging whole
objects that look alike and sit near each other.
