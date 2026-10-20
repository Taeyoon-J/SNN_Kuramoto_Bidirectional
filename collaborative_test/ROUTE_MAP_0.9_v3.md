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

## If `PV2_0115` fails: four levers that are not closed

Presenting one open path was thin. These are what remains, with what each is worth from
`PV2_0114`'s measured confusion (per image: 118.0 true foreground, 143.4 predicted, 31.8
false positives, 6.4 missed) and what each costs.

**A. `--target-foreground`, never used in any command in this repository.** It calibrates
the synchrony threshold per image to a fixed target fraction, and that fraction is a
hyperparameter selected on validation like any other — not an oracle. It is **the only dial
that acts on the 0.2006 half** rather than the 0.0428 half.

Its sign is genuinely unknown, which is why it is worth running rather than arguing. If
tightening is **unselective** it makes things worse, measured on the confusion above:

| excess removed | fg-IoU | pred_fg |
| ---: | ---: | ---: |
| none (now) | 0.7450 | 0.1400 |
| 20% | 0.7240 | 0.1351 |
| 60% | 0.6812 | 0.1252 |
| 80% | 0.6593 | 0.1202 |

If it were perfectly selective it reaches 0.9456. The truth is between, and nothing in the
record says where. Note the distinction from `PV2_0114`'s signal check, which found no
signal **per group**; this dial acts **per patch**, through the synchrony threshold, and
that is a different question. **Cost: evaluation only, no training.**

**B. `PV2_0098`'s `DS_6000` checkpoint is trained and has never been scored.** Six times the
training images took the loss to 14.429044 against 16.673826 and the collapse to 1.954825
against 2.154499. Both are training-side and this project has five times seen a better loss
fail to convert — but the checkpoint sits on `/work` and the test resumes by scoring it.
**Cost: evaluation only, no training.**

**C. Grid 64.** Our own 16 → 32 step bought **+0.1017** of actual pixel-protocol score and
lifted the ceiling by 0.1382. `PV2_0087` priced a 4× finer grid at **18.6×** the compute, so
64 is expensive and the peer is already moving that way (`SW_0154` onward). **Cost: large,
and the machine is currently shared with another user on all four GPUs.**

**D. `PV2_0091`'s clean dial sweep, still owed.** That record concluded "the gap is in what
the spikes encode, not in how they are thresholded" **from values already recorded**, and
says so: its own sweep was never run. The conclusion this route map leans on is therefore
argued, not measured, and A would also settle it.

**Ordering.** A and B need no GPU time at all and both bear on the larger half, so they run
first when the tunnel returns, regardless of what `PV2_0115` says. C is the fallback that
costs real compute. D is cheap and closes an open claim.

## What is not yet established, stated plainly

- `PV2_0115` has **no result**. The 8-epoch screen is an interim measure forced by compute:
  user `ziquanw` occupies all four GPUs and the identical config fell from 40 epochs in 23
  minutes to 6 in 39. The screen can only answer the mechanism question; its control already
  breaches the calibration gate at `pred_fg` 0.3699, so no absolute number from it may be
  compared to the published line.
- `PV2_0106`'s 0.8726 coverage and `PV2_0113`'s 0.9243 are **different measurements** — seed
  count, image count and possibly the coverage definition — and are not interchangeable.
- The 0.9329 and 0.9115 ceilings are **oracle results** and are reported as diagnostics.
