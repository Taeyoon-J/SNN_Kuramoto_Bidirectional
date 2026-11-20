# Where 0.9 stands, after the corrections

`ROUTE_MAP_0.9.md` closed thirty-one routes and concluded that the missing information is
instance identity. Three things measured since then change what that conclusion rests on,
and one of them changes the conclusion itself. This file supersedes its "what a credible
attempt would have to do" section; everything else there still holds.

## 1. The comparison we were measuring against was wrong (`PV2_0090`)

Slot Attention was run through its published centre crop, which on a 240x320 frame is the
centre 240x240, so its prediction grid and the target grid described **different regions**
— column `c` up to 40 px, two target columns, apart. Scored aligned, on test, three seeds
times three seeded draws:

| metric | ours, best | Slot, crop | Slot, aligned |
| --- | --- | --- | --- |
| `patch_fg_ari` | 0.7603 | 0.6656 | **0.8102** |
| `patch_foreground_iou` | **0.7132** | 0.1151 | 0.1151 |
| `patch_matched_object_iou` | **0.6598** | 0.0874 | 0.1025 |

Isolating the geometry alone — the same crop predictions against targets rebuilt on the
same crop — accounts for +0.0513 of the +0.1446; the rest is the model doing better when
it can see the whole scene. **"We beat it on all three" does not survive.** The two IoU
results do, and by more than before.

## 2. The readout cannot close the gap (`PV2_0091`)

fg_ari is computed over true-foreground patches, so predicted extent moves it for free and
only a matched-extent comparison means anything. Tracing our whole dial:

| | fg_ari | predicted foreground |
| --- | --- | --- |
| connected components | 0.7300 | 0.129 |
| spectral k=14 | 0.7603 | 0.556 |
| **spectral k=20** | **0.7718** | **0.808** |
| aligned Slot Attention | 0.8102 | 0.8531 |
| a perfect segmenter (`PV2_0082`) | 0.9756 - 1.0000 | — |

At k=20 we buy almost Slot's width and score 0.0384 below it. **0.7718 is the dial's
maximum anywhere, 0.13 below the bar**, and the curve is flattening — k=14 to k=20 widens
extent 45% for +0.0115. So the gap is in what the spikes encode, not how they are
thresholded, and an architectural change is **necessary**, which is a firmer statement
than thirty-one closed routes gave.

## 3. What is now excluded, and what that leaves

| explanation | status |
| --- | --- |
| the metric or the contract | excluded — a perfect segmenter scores 0.9756 under a far stronger rule change than a tie-break (`PV2_0082`) |
| input resolution | excluded — 70.7% of the error is on patches holding exactly one object (`PV2_0083`) |
| the grid | excluded for the dynamics — 1024 oscillators form phase structure at `plv_mean` 0.7101 (`PV2_0089`), and the memory cost is 4.72 GB once chunked (`PV2_0088`) |
| the data | excluded — another model reaches 0.8102 on these same images (`PV2_0090`) |
| the readout | **excluded** (`PV2_0091`) |
| the dynamics' sign structure | tested and negative on its own platform (`PV2_0092`) |

## 4. The honest position on 0.9

**No demonstrated path to 0.9 exists, and the distance is larger than it looked.**

- The best anything has scored on this data under this contract is **0.8102**.
- Our best anywhere on our dial is **0.7718**.
- 0.9 is above every number anyone has measured here.

The gap from 0.7603 to 0.9 is 0.14. The gap from the best measured model on this data to
0.9 is 0.09. Treating 0.9 as one architectural change away is not supported by anything in
this project's record; treating it as reachable at all requires believing that a
label-free model can get substantially closer to a perfect segmenter than Slot Attention
does, on a dataset where 17.3% of foreground patches straddle two or more objects.

## 5. The two routes with evidence behind them

**Motion (`PV2_0086`).** On an 8-channel MOVi platform whose control reaches 0.4723 at a
predicted foreground of 0.1006 against a target of 0.1061, the motion-affinity term at
weight 10 gives **+0.0103 fg_ari and +0.0204 obj_iou while staying calibrated**. The first
positive in roughly thirty-one routes. One seed, small, and part of the fg_ari is bought by
a 0.019 widening — `obj_iou` is the part that cannot be. Needs seeds 1 and 2 and a
matched-extent re-run.

**Signed coupling (`PV2_0092`), re-run properly.** The gate failed and the control was not
the published model — 0.5027 against 0.7239 — most likely for omitting `PV2_0044`'s graph
freeze, so the question it was built to answer is untested. What the arms do show is that
`patch_matched_object_iou` rises in every one of them, +0.0337, +0.0284, +0.0778, growing
with the balance, in the connected-component readout. That metric is not helped by
over-prediction, which is why the gate uses it as its honest witness. The re-run needs the
published configuration as its control, and that is the first thing to do when compute
returns.

## 6. What a credible attempt still has to do

The four conditions in `ROUTE_MAP_0.9.md` stand, with one added:

5. **Verify the control reproduces the published number before reading any arm.** A
   healthy `spike_plv_collapse` is not a healthy platform. `PV2_0085` failed this way and
   `PV2_0092` repeated it; the regression row protects the evaluation, and nothing was
   protecting the training command.
