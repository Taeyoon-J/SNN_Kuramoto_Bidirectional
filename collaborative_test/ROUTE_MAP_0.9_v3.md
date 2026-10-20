# Route map to 0.9, v3 — written 2026-10-20, after the readout closed

Supersedes `ROUTE_MAP_0.9_v2.md`, whose central proposal — two readout-side stages,
merging and coverage — was measured this session and does not survive. Every number below
was measured here or on the peer branch and is cited to the test that produced it.

## Where the goal stands

| part | state |
| --- | --- |
| exceed 0.75 patch FG-ARI | **met** — 0.7603, test split IDs 7000–7999, 1000 images, three independently trained seeds, std 0.0003 |
| review the peer and apply what holds | **done** — `REVIEW_0007`; `PV2_0111` withdrawn on `SW_0234`'s evidence, `PV2_0112` designed from `SW_0207` against `SW_0234`, implemented, verified 8/8 offline, run at three seeds |
| a route to 0.9 keeping SNN + Kuramoto bidirectional | **open** — this document |

## What v2 proposed, and what killed it

`PV2_0106`'s oracle decomposition said merging and coverage together reach **0.9329**, and
the peer reproduced the shape independently at **0.9115** (`SW_0234`). The ceiling stands.
What closed is every way of *reaching* it from the readout.

| route | evidence | state |
| --- | --- | --- |
| merge groups | `PV2_0107b`, 3 seeds: fg_ari +0.0483, **obj_iou −0.0285**. `SW_0234`'s merge is **GT-prescribed** and still loses the column | **closed** — not a pair-selection failure |
| add unclaimed patches | `PV2_0112`, 3 seeds, gate failed at every quantile; the admitted patches were **0–1% foreground**, because the base rate among unclaimed patches is **0.73%** | **closed** by arithmetic |
| widen existing groups | peer `SW_0207`: fg-IoU **+0.171**, FG-ARI **−0.054**, instances fragment | **closed** by them |
| delete spurious groups | base rate is favourable (22.2%) but **no signal exists**: internal spike coherence AUC **0.500**, mean activity **0.494** | **closed** for want of a signal |
| "make it fire on small objects" | `PV2_0113`: it already does — coverage 88% at 3–5 patches against 96% at 21–40, and **0.5%** of objects claimed by nothing | **refuted**, my own hypothesis |

## The measurement that reframed the problem

`PV2_0114`, from the readout's own predictions, per image:

| | |
| --- | ---: |
| true foreground | 118.0 |
| predicted foreground | 143.4 |
| **false positives** | **31.8 — 22.2% of what we predict** |
| false negatives | 6.4 — 5.4% of what exists |

| foreground IoU | | |
| --- | ---: | ---: |
| as it is | 0.7450 | |
| every miss recovered | 0.7879 | +0.0428 |
| every false positive removed | **0.9456** | **+0.2006** |

**Inventing foreground costs 4.7x what missing it does.** Every stage either branch built —
ours and theirs — pushed foreground *up*, at the 0.0428 half. `PV2_0106` named the second
oracle *coverage*, both branches took the name, and the name pointed at the smaller error.

## The one measurement that opens a route

The figure/ground cue **exists in the input and is destroyed before the spikes**:

| signal | separates true foreground | where |
| --- | ---: | --- |
| γ deviation from a patch's 5×5 neighbourhood | **AUC 0.947** | input, 200 images, label-free |
| spike internal coherence | 0.500 | readout |
| spike mean activity | 0.494 | readout |

And no loss asks for it. Every PLV term in `loss_function.py` shapes **partition structure**
— bimodality, balance, collapse, participation ratio — and none says which block is
foreground, which is why the background block is as coherent as the objects.

The arithmetic says those terms *cannot* say it. `plv_group_balance_loss` targets the mean
off-diagonal PLV, a mean over **pairs**, so a coherent group of n nodes counts **n²**. With
~900 background nodes of 1024 the background alone supplies **0.7726** of the 0.867 target
and a 3-patch object is worth **1/90,019** of it.

## The route, and what would falsify it

**`PV2_0115`: put the cue back, label-free.** `spike_gamma_contrast_loss` adds the negative
per-image Pearson correlation between mean spike activity after settle and that γ deviation.
Scale-free, reads no ground truth, touches no part of the SNN + Kuramoto bidirectional core.
Implemented, gradient-checked on both the int and the pair grid form, training.

If it works, the arithmetic of the gain is already known: correct figure/ground placement
attacks the 31.8 false positives, and removing them is worth **+0.2006** of foreground IoU
against the +0.0428 available from coverage. That is the first lever this project has found
that points at the larger half.

**Gate, registered before the run.** Mechanism first and cheaper: activity-vs-foreground AUC
must rise from 0.494. Then all three metrics above control at predicted foreground ≤ 0.20,
with `patch_matched_object_iou` specifically up.

**Falsifier, also registered.** If the AUC rises and the three metrics do not, then
figure/ground in the spikes was not what limited the readout, and this line closes with the
others. The remaining lever would then be the readout's own foreground rule rather than the
spikes, and the honest position would be that 0.9 is not reachable on this contract without
a change larger than a loss.

## What is not yet established, stated plainly

- `PV2_0115` has **no result**. The 8-epoch screen is an interim measure forced by compute:
  user `ziquanw` occupies all four GPUs and the identical config fell from 40 epochs in 23
  minutes to 6 in 39. The screen can only answer the mechanism question; its control already
  breaches the calibration gate at `pred_fg` 0.3699, so no absolute number from it may be
  compared to the published line.
- `PV2_0106`'s 0.8726 coverage and `PV2_0113`'s 0.9243 are **different measurements** — seed
  count, image count and possibly the coverage definition — and are not interchangeable.
- The 0.9329 and 0.9115 ceilings are **oracle results** and are reported as diagnostics.
