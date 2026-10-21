# PV2_0122 — the warm start works: grid 64 segments with no fine-tuning at all

Status: **step 1 completed, positive.** Route C is reopened. Replaces `PV2_0120`, which
trained grid 64 from scratch and was abandoned.

## Why `PV2_0120` was the wrong experiment

`PV2_0120` trained grid 64 from scratch, did not finish one epoch in 8.5 hours, and
projects to about **14 days** for 40. The peer reports native64 at roughly **2 hours a
seed** (`SW_0192`: 286.8 s per batch of 4, TRAIN64 with 16 proposals ≈ 1.98 h). `SW_0154`
says why — they never train the fine grid from scratch:

> *"A row-major native16 node maps to its exact 4x4 block of native64 children.
> Node-leading parameters repeat by that mapping; pairwise parameters use it on both axes."*

`REVIEW_0006` recorded this technique as worth taking and then I designed route C without
it. That omission cost `PV2_0120`'s 8.5 hours.

## What was ported

`map_grid.py`, grid 32 → 64, each parent node owning a 2×2 block:

| kind | tensors | treatment |
| --- | --- | --- |
| pairwise | `sc`, `kuramoto.direction_learner` | mapped on both axes → (4096, 4096) |
| node-leading | `kuramoto.omega`, `kappa`, `dendric_layer.tau_n`, `membrane_layer.tau_m` | repeated along the node axis |
| no node axis (10) | encoder projection, graph temperature, coupling gain, spatial rate, geodesic contrast, `freq_gain`, dendrite dense | copied unchanged |
| `graph_generator.grid_distance` | | **rebuilt for the destination grid** |

`grid_distance` is geometric, not learned, so replicating the coarse grid's distances would
hand the finer model the coarse geometry. It is recomputed instead — and the formula was
**checked against the model's own stored values at grid 32, maximum absolute difference 0**,
rather than assumed.

The peer's checkpoint was deliberately **not** taken. `SW0227`'s native64 sits on their
declared mask-supervised leaderboard; borrowing the weights would make the number not ours.
The mapping rule is published in their record and porting it is legitimate.

## Result — 30 validation images, both rows identical conditions

| | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent, grid 32 (`GC_w0`) | **0.7588** | 0.7629 | 0.7179 | 15.27 | 0.1281 |
| **replicated, grid 64, zero fine-tuning** | 0.7066 | **0.7633** | **0.7188** | 19.70 | 0.1257 |

**Foreground IoU and matched-object IoU survive the move intact** — 0.7633 against 0.7629
and 0.7188 against 0.7179, with four times the oscillators and not one gradient step.
Predicted foreground holds at 0.1257 against the parent's 0.1281.

FG-ARI falls 0.7588 → 0.7066, and it falls **with** the group count rising 15.27 → 19.70.
That pairing is the mechanism: replication gives each parent node's 2×2 block four children
that start from identical parameters but receive **different γ**. Where phases diverge
inside a block, one object is split across several groups — which FG-ARI penalises and
neither IoU does. That is precisely what a short fine-tune has to repair, and it is a
plausible reason the peer needs only 16 proposals.

## What this changes

Route C is **reopened** and is no longer a 14-day proposition. `PV2_0120`'s record stands
as the abandoned from-scratch attempt.

## Three failures on the way here, all the same kind

The grid-64 evaluation failed three times before producing a number, and each was me
assuming a runtime condition instead of measuring it:

1. The targets declared `contract_version='64'` because I omitted `--contract` when building
   them. The evaluator's guard refused to score rather than silently mislabelling the
   number. Corrected to `'3'` — grid 16 is v1, 32 is v2.
2. `grid_distance` was dropped rather than rebuilt, and a strict `load_state_dict` needs the
   key present.
3. `evaluate_model.py` **has no batch flag**: `--limit` is the batch, and `_relax` allocates
   `batch × chunk × N × N`, which is 12.5 GiB at limit 100 and chunk 8. Fixed with chunk 1
   and 30 images, applied to **both** rows so the comparison stays like for like.

## Next

Fine-tune from the replicated checkpoint and re-measure. The target is the group count:
if a short fine-tune pulls 19.70 back toward 15 while the two IoUs hold, FG-ARI should
recover and route C has a number worth three seeds.
