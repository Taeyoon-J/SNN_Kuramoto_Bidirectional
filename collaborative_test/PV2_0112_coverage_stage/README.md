# PV2_0112 — a label-free coverage stage, specified and pre-registered

Status: **planned** — implemented and verified offline; cannot run until the server returns
Parent: `PV2_0106` (oracle decomposition), `PV2_0107b` (merge stage, failed), `REVIEW_0007`
Replaces: `PV2_0111`, withdrawn. That number is burned and is not reused.
Contract: v2, grid 32, split manifest v1. Control: `PV2_0104` checkpoints `FRZ_s{0,1,2}`.

## Why this, and why not the linkage sweep

`PV2_0111` was going to sweep complete against mean linkage on the belief that our merge
stage lost `patch_matched_object_iou` because it picked the wrong pairs. `REVIEW_0007` killed
that hypothesis: the peer's `SW_0234` merge is **GT-prescribed** — the pairs are chosen with
the truth — and it still loses the column, .674313 → .667059. Our label-free merger lost
.0285 at three seeds. Pair selection is not the cause, so a better linkage cannot be the fix.

What both decompositions agree is valuable is the *other* stage. On their masks, foreground
recovery alone moves **all three**: +.0248 FG-ARI, +.0359 foreground IoU, **+.0877
matched-object IoU**. On ours, coverage alone was worth +.1001 FG-ARI. It is the only stage
either branch has measured that lifts the honest column.

## The constraint, taken from two of their records before writing any code

| | what it did | FG-ARI | foreground IoU | Object IoU |
| --- | --- | ---: | ---: | ---: |
| `SW_0207` | widened the support of groups that already existed | −.053863 | **+.171275** | down |
| `SW_0234` oracle | created new groups for objects missed entirely | +.0248 | +.0359 | **+.0877** |

`SW_0207` is label-free at inference and was **closed, not adopted**: instances fragment. The
two differ in one respect only, which `SW_0234` states — the oracle *완전히 누락된 물체에는
새 그룹도 만든다* and *이미 합친 그룹은 분리하지 않는다*.

**So the rule this stage encodes:** an existing group's membership is never touched. Only
patches that no group claimed may become groups, and only among themselves. A patch adjacent
to an existing group stays unclaimed rather than joining it.

## What was built

`evaluate_model.py::_coverage_groups`, with `--coverage-quantile` (the one dial),
`--coverage-synchrony` and `--coverage-min-size`. It runs **after** the merge stage, never
before: covering first would let the merger absorb the recovered groups into existing ones,
which is the `SW_0207` failure by a different route.

Per image: take the mean activity over time after `settle`; mark every patch any group
claims; admit unclaimed patches strictly above the `coverage_quantile` of the image's own
activity; link admitted patches that are 4-neighbours on the grid, optionally also requiring
affinity above `coverage_synchrony`; keep components of at least `coverage_min_size` and
append them as new groups. No ground truth, no object count. The dial is selected on
validation.

## Verified offline — `test_coverage.py`, 8 checks, all passing

Run with CPU torch on a synthetic 16x16 grid holding three 3x3 blocks, the readout handed
only two of them:

```
PASS  existing groups untouched
PASS  exactly one new group
PASS  recovered group IS the missed object
PASS  no patch in two groups
PASS  inert at an extreme quantile
PASS  inert when quantile is None
PASS  two separated misses stay two groups
PASS  adjacent patch does not widen an existing group
```

**The test caught a real bug before any GPU time was spent.** The first implementation
admitted patches with `act >= quantile_value`. The background is the majority of an image —
87.5% of patches — so for any quantile below that majority the quantile *value* is the
background value itself and `>=` admits the entire background as one component: 238 of 256
patches came back as a single recovered "object". Changed to strictly `>`. Where activity is
continuous the two forms agree, so the fix is safe in both regimes.

## Pre-registered success criterion

Registered here, before the run, as `PROPOSAL_0.9_v2.md` requires:

1. **All three** metrics above the `PV2_0104` control at the same checkpoints, three seeds.
2. `patch_matched_object_iou` must rise. It is the column `SW_0207` lost and the column the
   merge stage lost; a gain that does not appear here is not a gain.
3. Predicted foreground at or below **0.20** — this stage adds foreground by construction, so
   the calibration gate is the binding constraint, not a formality.

**Falsification, stated now:** if foreground IoU rises while FG-ARI or matched-object IoU
falls, that is the `SW_0207` failure mode reproduced on our side, and the stage closes
without a threshold or learning-rate rescue — as theirs did.

## Commands, ready to run

See `commands.sh`. It opens with the mandated regression row on `BIM6_s1` expecting
`0.7239 0.7251 0.4951`; nothing below it is read unless that reproduces.

## Blocked on

`/export_home` is faulted and the VPN tunnel to the 152.19 subnet is down. Both are recorded
in `STATUS.md`. This test is fully specified, implemented and offline-verified; it needs only
a working server.
