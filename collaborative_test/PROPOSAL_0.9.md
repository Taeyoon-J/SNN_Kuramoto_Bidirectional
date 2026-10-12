# Proposal: the one route left to 0.9, and what it would cost

**Status: needs a decision before it can start.** It introduces data this project does
not currently use. Everything else in `ROUTE_MAP_0.9.md` is closed.

## What 29 routes established

The antecedent of the goal's third part — "if unreachable in the current architecture" —
is now rigorous rather than assumed:

- **The metric is not the obstacle.** A perfect pixel segmenter scores 0.9756-1.0000 on
  contract v1 under any defensible patch rule (`PV2_0082`), so 0.9 is admitted.
- **Input resolution is not the obstacle.** 70.7% of the error is on patches holding
  exactly one object, where nothing is ambiguous (`PV2_0083`).
- **The architecture transcribes faithfully.** Given a grouping, the readout reproduces it
  at 0.97 (`PV2_0066`, `PV2_0072`).
- **The failure is coherent object merges.** 38.8% of images hold an object pair whose
  centroids sit closer together than the objects are internally spread (`PV2_0053`), the
  pipeline is a fixed point on such errors at 0.727 (`PV2_0073`), and it repairs only the
  spatially incoherent kind (+0.152 at heavy random corruption, `PV2_0072`).
- **Every label-free source of object identity is tested.** Reconstruction with slot
  competition (`PV2_0039`, `PV2_0068`), cross-view consistency under appearance nuisance
  (`PV2_0070`), and self-reference (`PV2_0069`, `PV2_0073`).

So the missing information is specific: **two objects that look alike and sit near each
other are different objects.** A single static render does not contain it. Nothing in the
grouping pipeline can invent it.

## The proposal

**Add a dataset where objects move independently, and keep everything else.**

Independent motion supplies exactly the missing fact, without labels: two regions that
move differently are different objects, and two that move together are one. It is the one
signal that a fixed point at the current accuracy cannot satisfy, because agreeing on a
merged pair becomes inconsistent across frames.

### What changes, and what does not

| | |
| --- | --- |
| **unchanged** | the Kuramoto rollout, the gate, the dendrite, the membrane, the spike readout, the graph generator's structure, the patch grid, `evaluate_patch_masks`, the three metrics |
| **added** | a frame pair per sample, and one loss term: the synchrony matrix of frame *t* must agree with that of frame *t+1* after the motion between them |
| **the plumbing already exists** | `--recon-target-path` shows how an external per-patch target threads through training with frozen features; `--view-consistency-weight` already computes a second view and a PLV agreement term (`PV2_0070`); `--phase-gamma-path` and `graph_extra_channels` route a signal to the graph alone |

`PV2_0070`'s term is the same shape. What failed there was the *view*: appearance jitter
preserves the partition, so agreeing on a merged pair was free. Motion does not preserve
it — a merged pair splits between frames — which is the difference that matters.

### Candidate datasets

| dataset | fit | cost |
| --- | --- | --- |
| **MOVi-A** (Kubric) | CLEVR-like shapes and materials, independent object motion, same style of masks | closest to contract v1; would need a new split manifest and patch targets |
| **CATER** | CLEVR objects, camera and object motion | motion is partly camera, which confounds the signal |
| **CLEVRER** | CLEVR objects, collisions | richer than needed |

MOVi-A is the closest match: keeping the render family means the encoder and the operating
point carry over, so the experiment tests the signal rather than a domain shift.

### How it would be judged

Unchanged from this cycle, and the gates matter more than the idea:

1. A **regression row** first — re-score an existing checkpoint, refuse to read anything
   below it unless the recorded value reproduces.
2. Gains counted **only below predicted foreground 0.20**. Three automated verdicts in this
   cycle were wrong because they read fg_ari alone while predicted foreground had moved to
   0.85, 0.65, or an oracle row had been counted as usable.
3. **Do not optimise the loss.** Ten changes reached a lower loss with a worse score
   (`PV2_0080`); lowering it is not evidence of progress.
4. Promising settings go to seeds 0, 1, 2 before any claim. Four single-seed conclusions
   were retracted in earlier cycles.
5. Contract v1 and the fixed split stay as they are, so every number here remains
   comparable. A motion dataset needs its **own** contract and its own Slot baseline; it
   does not replace this one.

### Honest expectation

Motion is the mechanism the measurements point at, and it is the only untested source. It
is **not** a guarantee of 0.9. The nearest evidence is that our readout transcribes a good
grouping at 0.97 (`PV2_0066`), so if motion supplies groupings materially better than
0.76, the pipeline should carry them — but `PV2_0070` showed a consistency term can be
satisfied while the score falls, and that risk applies here too.

### What it is not

Not a contract change. `PV2_0082` measured that moving to a finer grid changes the number
without addressing the cause, and `ROUTE_MAP_0.9.md` records it so nobody spends effort
there.

## Progress (`PV2_0084`)

Started, with the dataset in place and two corrections to the reasoning above.

**Done:** MOVi-A downloaded and prepared (1,000 train, 250 validation scenes, 9 GB), with
statistics close to contract v1 — foreground 0.103 against 0.118, 5.97 objects against
6.32, the same labelling rule. Motion saturates around a 6-12 frame gap at 17-19% of
pixels changing instance label.

**Correction 1.** Motion's margin looked 13x appearance's against *raw* RGB means; against
*standardised* appearance it is **1.4x** (0.717 against 0.501). And MOVi-A's standardised
appearance already has twice contract v1's margin, so **the dataset change confounds the
motion test** — any claim has to be made within MOVi-A.

**Correction 2.** I took a shortcut and concatenated motion as extra gamma channels; it
hit `PV2_0054`'s dead-membrane wall (`spike_plv_collapse` 9.21 = log(10^4), no spikes),
and routing it to the graph alone did not rescue it. **Concatenation was never this
proposal** — the design above specifies motion as the external target of a consistency
term, and the smoke test caught the drift before a run was wasted.

**Next:** a per-patch motion affinity as that target, against the appearance-only arm,
which trains healthily.

## If the answer is no

The session's result stands on its own: **test 0.7603 / 0.3373 / 0.6598**, three seeds,
contract v1, 16x16, beating a matched Slot Attention trained on the same split on all
three metrics by +14.1%, +187% and +658% — with 29 routes, five structural findings and
six withdrawn or corrected claims documented.
