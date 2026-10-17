# PV2_0092 — signed coupling

**Status:** running (seed 0 pilot). **Parent:** `PV2_0061`, with the reframing from
`PV2_0090`. **Contract:** v1, unchanged.
**Hypothesis:** the grouping fails by merging objects because the coupling matrix is
strictly non-negative, so nothing in the dynamics ever pushes two patches apart. Letting
a node's least similar admitted neighbours repel it should break merges that every
pairwise attractive constraint permits.
**Gate, fixed before the run** (`PROPOSAL_0.9_v2.md`): the gain must appear with predicted
foreground at or below 0.20; it must not be bought with foreground IoU; and
`patch_matched_object_iou` must move too, since over-prediction does not help that one.

## The analysis, before the change

`graph_generator.py:152-156` builds the adjacency as `values.softmax(dim=-1) * gain`
scattered into a zero matrix and symmetrised. Every entry is non-negative, so every
coupling is attractive. Groups form only because the graph is sparse (top_k 32 of 256)
and spatially decayed — nothing separates them.

Measured on a trained checkpoint with the oscillators held in two phase groups, reading
the mean coupling drive on each:

| coupling | drive on group A | drive on group B |
| --- | --- | --- |
| as it stands | +0.00904 | -0.00904 |
| rows balanced to sum zero | **-1.27025** | **+1.27025** |

140x larger, and in the separating direction. The existing graph is nearly neutral about
two groups coexisting.

Three recorded findings predict this is the right place to look. `PV2_0069`: the graph is
a fixed point — its own groups fed back give 0.7199 from 0.7239, so no iterative scheme
climbs, which is what a pairwise attractive rule implies, since a merge violates no
pairwise constraint it contains. `PV2_0075`: the corridor between objects is 6x cheaper
than a within-object step, so expensive is not forbidden. `ROUTE_MAP_0.9.md` finding 2:
the error mode is coherent object merges.

Mixed-sign coupling is also the classical route to cluster synchronisation in Kuramoto
systems. The architecture has never been in that regime.

## The change

One scalar, default 0.0 (inert):

```python
weights = values.softmax(dim=-1)
if self.coupling_balance != 0.0:
    weights = weights - self.coupling_balance * weights.mean(dim=-1, keepdim=True)
weights = weights * self.log_coupling_gain.exp()
```

Nothing else moves: Kuramoto oscillators, image-conditioned learned graph, gating,
dendrite, membrane, spikes and the classifier readout are untouched, and the score still
comes from classifier masks on spikes.

Files: `graph_generator.py` (argument, `self.coupling_balance`, the three lines above),
`hyperparameter.py` (`coupling_balance: float = 0.0`), `s2net_cls.py` (pass-through),
`train_s2net_core.py` and `evaluate_model.py` (`--coupling-balance`).

## Checks before running

| check | result |
| --- | --- |
| balance 0.0 adjacency | row sum 8.0000, min 0.000000, **0.0% negative** — the original |
| balance 0.5 | row sum 4.000000, min -0.125000, 13.9% negative |
| balance 1.0 | row sum 0.000000, min -0.250000, 14.0% negative |
| gradients at 0.0 / 0.5 / 1.0 | healthy and identical except `log_coupling_gain`; reach `projection.weight` 7.938e+01, `spatial_rate` 2.351e+00, `geodesic_contrast` 4.582e-01, `log_temperature` 1.900e+01 |

A first gradient check reported zeros at balance 1.0. That was the test's fault, not the
code's: it used `A.sum()` as the loss, which is identically zero once the rows sum to
zero. Repeated against `(A * random).sum()` the gradients are healthy, as above. Recorded
because the zero would have looked exactly like a detached graph.

## Result

Pending. Four arms at seed 0 — balance 0.0, 0.25, 0.5, 1.0 — each read through both the
connected-component and the spectral k=14 readouts, with a contract-v1 regression row
first so nothing below it is read unless 0.7239 / 0.7251 / 0.4951 reproduces.

## Limitations, stated in advance

Seed 0 only; the protocol requires seeds 0, 1, 2 before anything is claimed. The balance
is a single global scalar, so it cannot express "repel across an object boundary, attract
within one" — a learned or per-edge sign would be the next step if the global version
moves anything. Four of this project's routes improved a mechanism-level statistic without
moving the score, and that remains the most likely outcome here.

---

# Result — the gate is not passed, and the control is not the published model

## What came back

Seed 0, contract v1, validation 300, four arms identical but for `coupling_balance`.
The contract-v1 regression row passed first: 0.7239 / 0.7251 / 0.4951.

| arm | training loss | collapse | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- | --- |
| balance 0.0 (control) | 23.52 -> 17.00 | 2.157 | 0.5027 | 0.5112 | 0.3903 | 0.1824 |
| balance 0.25 | 23.66 -> 17.05 | 2.142 | 0.5962 | 0.4708 | 0.4240 | 0.2325 |
| balance 0.5 | 24.04 -> 17.12 | 2.182 | 0.4949 | 0.3370 | 0.4187 | 0.2869 |
| balance 1.0 | 25.22 -> 17.22 | 2.172 | 0.5352 | 0.2520 | 0.4681 | 0.4090 |

Against the gate registered in `PROPOSAL_0.9_v2.md`:

| balance | d fg_ari | d fg_iou | d obj_iou | extent <= 0.20 | iou kept | obj moved |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | +0.0935 | -0.0404 | +0.0337 | **no** (0.2325) | **no** | yes |
| 0.5 | -0.0078 | -0.1742 | +0.0284 | **no** (0.2869) | **no** | yes |
| 1.0 | +0.0325 | -0.2592 | +0.0778 | **no** (0.4090) | **no** | yes |

**Signed coupling does not pass.** Every arm widens predicted foreground and sells
foreground IoU to do it — exactly the failure mode the gate exists to catch.

## The error in this experiment, stated plainly

**The control is not the published model.** Retrained at seed 0 it scores **0.5027**
where the published contract-v1 configuration scores **0.7239** on the same images
through the same readout — a gap of 0.22. The regression row reproduces 0.7239 exactly
from the existing checkpoint, so the evaluation is right and the **training command is
wrong**. The most likely omission is `PV2_0044`'s graph freeze: the published three seeds
are the FZG (frozen-graph) line and these arms trained the graph from scratch.

So this experiment answers a narrower question than it was built to answer:

- **Valid:** the arm-to-arm comparison. All four share one base and differ in one scalar,
  so what signed coupling *does* is measured.
- **Not tested:** whether signed coupling helps the published model. Its platform is 0.22
  below it.

This is the failure `PV2_0085` was corrected for, repeated. The check performed was
`spike_plv_collapse` (2.157, healthy) and the check **not** performed was the control's
own score against the published number. A healthy collapse value is not a healthy
platform.

## The one signal worth keeping

`patch_matched_object_iou` rises in **every** arm — +0.0337, +0.0284, +0.0778 — and grows
with the balance. That metric is not helped by over-prediction, which is what makes it
the honest witness in this project's own gate. So the mechanism does something real to
per-object grouping while damaging figure/ground. The direction is informative even
though the gate failed: repulsion separates objects and simultaneously pushes patches out
of the background cluster.

## Next

Re-run the control with the published configuration — `PV2_0044`'s freeze included —
before any further claim about signed coupling. Until then this stands as a negative on
its own platform and as a note that the mechanism moves `obj_iou` in a way no other
tested route has.
