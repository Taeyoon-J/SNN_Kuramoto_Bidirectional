# The route to 0.9, rewritten after `PV2_0090`

`PROPOSAL_0.9.md` named one remaining route — a dataset with independent object motion —
because thirty closed routes had established that the missing information is "two objects
that look alike and sit near each other are different objects", and that a static render
does not contain it. That proposal is now partly answered (`PV2_0086`, below) and partly
superseded, because `PV2_0090` changed what is known.

## What `PV2_0090` changed

Until it, no model had been measured near 0.8 on this data under this contract, so every
statement about reachability rested on our own ceiling arguments. Now one has:

| metric | ours, best | aligned Slot Attention |
| --- | --- | --- |
| `patch_fg_ari` | 0.7603 | **0.8102** |
| `patch_foreground_iou` | **0.7132** | 0.1151 |
| `patch_matched_object_iou` | **0.6598** | 0.1025 |

This is a **complementarity, not a ranking**. We know where the objects are — foreground
IoU 6.2x theirs, per-object IoU 6.4x theirs. They know which of the patches they call
foreground belong together. Neither model is strictly better, and the thing we are worse
at is the thing the goal's first condition measures.

That reframes part 3. The question is no longer "is 0.9 reachable on this data" — a
reconstruction-trained slot model gets to 0.81 while being almost blind to figure/ground,
so the information is clearly there. The question is **what, architecturally, lets slot
competition partition foreground better than Kuramoto synchrony does**, and whether that
can be added without giving up the core.

## The architectural difference, identified

Slot Attention's slots **compete**: each pixel's attention is a softmax **across slots**,
so a pixel claimed by one slot is thereby not claimed by another. The partition is a
constraint, not an outcome.

Our grouping is pairwise phase synchrony on a **strictly non-negative adjacency**.
`graph_generator.py:152-155` takes the top-k logits, softmaxes them, scales by a positive
gain and scatters them into a zero matrix. Every coupling is attractive. **Nothing
anywhere pushes two patches apart.** Groups exist only because the graph is sparse
(top_k 32 of 256) and spatially decayed.

Measured on a trained checkpoint, with the oscillators held in two phase groups and the
mean coupling drive on each group read off:

| coupling | drive on group A | drive on group B |
| --- | --- | --- |
| as it stands (non-negative) | +0.00904 | -0.00904 |
| rows balanced to sum zero | **-1.27025** | **+1.27025** |

The present graph is **nearly neutral** about two groups coexisting — a drive of 0.009 —
and what little it does is not separation. With signed coupling the drive is 140x larger
and points the other way: each group is pushed off the other.

## Why this is the right suspect

Three recorded findings line up behind it, and none of them was explained before:

- **`PV2_0069`: the graph is a fixed point.** Oracle one-hot codes in give 0.9671; the
  model's own 0.7239-accurate groups fed back give 0.7199. No iterative scheme climbs.
  A pairwise attractive rule has no mechanism to *correct* a merge, because a merge is
  consistent with every pairwise constraint it contains: if A~B and B~C are each
  plausible, A, B and C all lock, and nothing is violated.
- **`PV2_0075`: the corridor is 6x cheaper than a within-object step.** The geodesic
  makes crossing between objects expensive, but expensive is not forbidden, and with
  only attraction available the cheapest route still eventually wins.
- **The error mode is coherent object merges** (`ROUTE_MAP_0.9.md`, finding 2): the
  pipeline repairs spatially incoherent damage (+0.152) but is a fixed point at 0.727 on
  its own errors, which are merges.

Repulsion is the missing half of the mechanism. Mixed-sign coupling is also the classical
route to **cluster synchronisation** in Kuramoto systems — several phase-locked groups
holding fixed offsets — whereas purely positive coupling drives toward global coherence.
The architecture has never been in that regime.

## The change, and why it preserves the core

One scalar, `coupling_balance`, subtracting that fraction of each row's uniform share from
its top-k softmax:

```python
weights = values.softmax(dim=-1)
if self.coupling_balance != 0.0:
    weights = weights - self.coupling_balance * weights.mean(dim=-1, keepdim=True)
weights = weights * self.log_coupling_gain.exp()
```

At 0.0 this is the previous adjacency exactly. At 1.0 each row sums to zero: a node's most
similar admitted neighbours attract it, its least similar repel it, total drive conserved.

Nothing else moves. The oscillators are still Kuramoto, the graph is still image-conditioned
and learned, the gating, dendrite, membrane, spikes and classifier readout are untouched,
and the score still comes from classifier masks on spikes. This is a change to the sign
structure of an existing matrix, not a new module.

Verified before use: default 0.0 reproduces a strictly non-negative adjacency with row sum
8.0 and 0.0% negative entries; balance 1.0 gives row sum 0.0 and 14.0% negative entries;
gradients are healthy at 0.0, 0.5 and 1.0 under a non-degenerate loss, reaching the
projection weight, `spatial_rate`, `geodesic_contrast` and `log_temperature` alike.

## What would have to be true for this to work

Stated in advance so the result cannot be read generously after the fact.

1. **It must raise fg_ari inside the calibrated regime.** `ROUTE_MAP_0.9.md`'s gate:
   predicted foreground at or below ~0.20 against a target of 0.118. Three automated
   verdicts in this project were already wrong for ignoring it.
2. **It must not buy fg_ari with foreground IoU.** Our 0.7132 is the result that does not
   depend on a tie-break; a change that trades it away for fg_ari has swapped which half of
   the complementarity we hold, not fixed anything.
3. **`patch_matched_object_iou` must move too.** It is not helped by over-prediction, so it
   is the honest witness.
4. If it clears those at seed 0, seeds 1 and 2, then validation selection, then one test read.

## What this does not claim

It is not established that the current architecture cannot reach 0.9 — `PV2_0091` is
tracing our whole extent dial against Slot's operating point to put a number on how far
the readout alone can go, and that measurement is not finished. What is established is
that the gap is **not** the metric (`PV2_0082`: a perfect segmenter scores 0.9756-1.0000),
**not** the input resolution (`PV2_0083`: 70.7% of the error is on single-object patches),
and **not** the data (`PV2_0090`: another model reaches 0.8102 on it).

## The motion route, as it stands

`PROPOSAL_0.9.md`'s route was tested twice. `PV2_0085` called it negative on a control
scoring 0.3042, which was too weak to carry a verdict. `PV2_0086` rebuilt the platform —
an 8-channel MOVi encoder, control 0.4723 at predicted foreground 0.1006 against a target
of 0.1061 — and the motion term at weight 10 gives **+0.0103 fg_ari and +0.0204 obj_iou
while staying calibrated**. The first positive in roughly thirty-one routes, one seed,
small, and partly bought by a 0.019 widening of extent; `obj_iou` is the part that cannot
be bought that way.

Both routes stay open. They are not alternatives: motion supplies information the render
lacks, signed coupling supplies a mechanism the dynamics lack, and the second can be tested
on contract v1 today while the first cannot.
