# PV2_0082 — contract v1 admits 0.9; the gap is the model's, not the measurement's

**completed. This refutes the argument I was building, which is why it matters.** Parent
`PV2_0076`. Diagnostic: no model is involved and nothing here is a score.

## The hypothesis I was testing

After twenty-seven closed routes I was moving toward the claim that **0.9 may not be
reachable on contract v1 by any method**, and three facts made it look reasonable. The
published Slot Attention checkpoint scores **0.6195** here; the same architecture trained
on our own split scores **0.6663** (`PV2_0049`); we score **0.7603** (`PV2_0076`). If the
field's standard object-discovery method, properly trained, tops out at 0.67 on this
contract, then a 0.9 target might belong to a different measurement rather than to a
better architecture — the peer's 32x32 contract puts their model at 0.7819 with Slot at
0.7817, which is where 0.9-scale numbers would live.

`clevr_mask_patch` labels a 15x20 patch by majority vote with ties to the smallest ID, so
a patch straddling two objects gets one of them by **convention**. A perfect pixel
segmenter still has to choose, and under any other defensible rule it disagrees with
`targets_v1` on exactly the ambiguous patches. Scoring those rules bounds what any method
can be expected to reach.

## Result: the convention is not the obstacle

17.3% of foreground patches contain **two or more** objects, and 0.96% contain three or
more. So the ambiguity is real. But a perfect pixel segmenter barely pays for it:

| rule, applied to the TRUE pixel masks | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| majority, ties to low ID — contract v1 itself | 1.0000 | 1.0000 | 1.0000 |
| majority, ties to **high** ID | **0.9996** | 0.9956 | 0.9966 |
| any object present beats background | 1.0000 | 0.5099 | 0.4853 |
| an object must hold **half** the patch | **0.9756** | 0.9731 | 0.9626 |

Flipping the tie-break costs **0.0004**. Requiring an object to hold half of its patch —
a much stronger change than a tie-break — still leaves **0.9756**. The first row
reproducing 1.0000 is the check that the pipeline and the rule agree.

**So contract v1 admits 0.9 comfortably, and my hypothesis is refuted.** The 0.14 between
0.7603 and 0.9 is **model error, not measurement error**, and moving to a finer grid would
not earn it.

## Why this is worth recording as a negative

It removes the most attractive remaining explanation for part 3's failure. I had put the
32x32 contract on the list of options partly on this reasoning, and the measurement says
it would change the number without addressing the cause. Future work should not spend
effort on the contract.

It also sharpens what 27 routes mean. The target is legitimate and reachable in
principle; we are short because the model groups worse than a perfect segmenter, for the
reasons `ROUTE_MAP_0.9.md` records — the pipeline repairs incoherent damage while its own
errors are coherent object merges, the objective does not rank solutions the way the
metric does, and only 1,325 parameters carry the score. **Not because the metric is
unfair.**

## Limitation

Every rule here reads the true masks, so this bounds the *labelling convention*, not what
is inferable from an image. A perfect pixel segmenter is not available without labels;
the measurement says only that if one existed, the contract would score it near 1.0.
