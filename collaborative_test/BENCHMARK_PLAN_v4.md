# Benchmark plan v4

Supersedes v3. Written after PV2_0133–PV2_0136. v3 settled the protocol; v4 settles what we
must still do before the comparison is one we would defend, and in what order.

Our position is unchanged: the claim is that this model finds objects, and Slot Attention is
the bar it has to clear. v4 exists because three things we learned since v3 change how that
claim has to be made.

## 1. The protocol, unchanged from v3

- **Patch contract at grid 64.** Ground truth is quantised to the grid and scored there. The
  ceiling is 1.0 by construction. A predicted mask is never expanded to pixels to be scored.
- **One function, `evaluate_patch_masks`**, for both models, on the same image ids in the same
  order.
- **Three metrics, always reported together and never combined**: `patch_fg_ari`,
  `patch_foreground_iou`, `patch_matched_object_iou`.
- **The score comes from classifier masks on spikes.** `spike_synchrony_components` on
  `trace['spikes']`, then `foreground_induced_merge`, optionally refined by the readout head.
  theta and PLV are diagnostics and are never the score.

## 2. What changed, and what each change requires

### 2.1 The head is per-dataset, so say what it costs

PV2_0134: one head fitted on CLEVR and loaded unchanged loses on all three metrics on both
other datasets (Tetrominoes −0.0321 / −0.1068 / −0.0349). Generalisation is in the core.

PV2_0136 (this round): a **single fixed recipe** — arm `local_rgb_trust`, 2 epochs, the choice
B′ made independently on both Tetrominoes and Multi-dSprites validation — refits the head on a
new dataset using **only its unlabelled images**, and improves all nine metric cells:

| dataset | 253 | + head, label-free | delta |
|---|---|---|---|
| CLEVR | 0.8421 / 0.7545 / 0.6410 | 0.8468 / 0.7574 / 0.6456 | +0.0047 / +0.0029 / +0.0046 |
| Tetrominoes | 0.8580 / 0.7885 / 0.8173 | 0.8586 / 0.7895 / 0.8187 | +0.0006 / +0.0010 / +0.0015 |
| Multi-dSprites | 0.6974 / 0.8177 / 0.7628 | 0.7041 / 0.8213 / 0.7660 | +0.0067 / +0.0036 / +0.0032 |

**This is the configuration the benchmark should report**, because it is the one whose cost we
can state honestly: a frozen CLEVR-trained core, plus 256 unlabelled images and about a minute
per new dataset. The recipe was selected across the three **validation** splits, which the
contract permits; no evaluation split was involved in choosing it.

The per-dataset refit under B′ scores higher on CLEVR (0.8507 vs 0.8468) but needs that
dataset's validation labels. Report it as a separate, clearly-labelled row — an upper bound
with a labelled-validation cost — never as the headline.

**Requirement.** Every reported head row states which split chose its arm and epoch.

### 2.2 Selection needs a Foreground IoU floor

Selecting on validation FG-ARI alone once accepted **+0.0010 FG-ARI for −0.1141 Foreground
IoU**. Rule **B′** — highest validation FG-ARI among (arm, epoch) pairs whose validation
Foreground IoU is within 0.01 of the core's, with epoch 0 a candidate so the rule can decline
to move — is the selection rule for everything from here.

A lower FG-IoU in exchange for a higher FG-ARI is acceptable; a tenth of it is not. The floor
is the written form of that.

### 2.3 Slot Attention is not currently a fair reference

Three facts, all from PV2_0134:

- Its background rule is worth **nothing** on FG-ARI. All four rules tried give the identical
  value (0.4061 validation, 0.3747 fixed320), because they only touch patches the truth calls
  background and FG-ARI excludes those. Its score is not a reading artifact.
- Its published checkpoint uses **6.99 of 7 slots on every frame** with near-equal shares
  (0.156 / 0.147 / 0.141 / 0.148 / 0.133 / 0.131 / 0.144). It is tiling, not segmenting.
- It was trained on the **original CLEVR render**, not `clevr_with_masks`, which the authors
  themselves warn about. The older centre-crop run tiles the same way, so this is not caused
  by our preprocessing.

**Beating a model that is failing to transfer is not beating the model.** Until it is retrained
on our render, the comparison measures its domain shift as much as our segmentation.

**Requirement.** The reference column is reported in two rows, always both:

| row | what it is |
|---|---|
| Slot Attention, published checkpoint | a **transfer** result, single checkpoint, never a 3-seed mean |
| Slot Attention, retrained on `clevr_with_masks` | the **matched** comparison, 3 seeds |

If the retrained row cannot be produced, the headline says "we beat the published checkpoint
transferred to our render", which is a weaker and accurate claim, not "we beat Slot Attention".

### 2.4 Determinism is ours to use

PV2_0135: the 253 rollout is bitwise deterministic in `inference_seed` — three seeds give
identical gamma, q, adjacency and spikes. So our caches, diagnostics and scores reproduce
exactly from checkpoint plus image, with no inference seed to average over.

Slot Attention resamples its slots on every forward pass and needed `--seed` added to be
reproducible at all; re-scoring the same checkpoints once moved seed 0 from 0.7223 to 0.7234.
Its column therefore needs several draws averaged, ours does not.

**Requirement.** Report our inference as deterministic and state it. Average the reference over
at least 3 inference draws per checkpoint and report that spread.

## 3. Seeds — the thing that makes any of it a result

Everything above is **seed 0**. The contract judges success on **3-seed training means** of all
three metrics, per-seed values and standard deviation reported alongside.

What exists, checked against each run's own `training_report.json`:

| core | seeds | budget | what it supports |
|---|---|---|---|
| 253 trained on **CLEVR**, grid 64 | **0 only** | 10,000 unique × 10 epochs, 6,250 updates | every number in PV2_0133–0136, including all three datasets' transfer rows |
| 253 trained on **Tetrominoes** | 0, 1, 2 | 3,500 unique × 10 epochs, 2,190 updates | a *different claim*: a core trained on its own dataset |
| 253 trained on **Multi-dSprites** | 0, 1, 2 | 3,500 unique × 10 epochs, 2,190 updates | likewise |

The three-seed sets are **not** the seeds we need. They are cores trained on each dataset at a
smaller budget; our claim is about **one CLEVR-trained core reused elsewhere**, and that core
exists at seed 0 only. Using the per-dataset cores to fill a 3-seed column would answer a
question nobody asked and would misstate the generalisation result.

**Requirement, and the top of the queue.** Train seeds 1 and 2 of the 253 core on CLEVR at grid
64, 10,000 × 10 to match seed 0. About 18.5 h each on a rented 4090, roughly $8 per seed; two in
parallel is about a day. Until these exist, **every number in this project is seed 0** — on all
three datasets, not only CLEVR, because all three are read from the one CLEVR-trained core.

The per-dataset 3-seed cores are worth a separate, clearly-labelled table if we want to show
that a core trained on each dataset also works. That is a different claim and must not be mixed
into the generalisation column.

Never report one checkpoint evaluated three times as a 3-seed training mean.

## 4. Order of work

1. **Seeds 1 and 2 of the 253 core on CLEVR.** Nothing else converts candidates into results.
   Needs a spending decision.
2. **Retrain Slot Attention on `clevr_with_masks`, 3 seeds.** Without it the headline claim is
   about a failed transfer. No GPU rental needed — it trains on frontier.
3. **Re-score the three datasets under the fixed label-free recipe at all available seeds**,
   producing the per-seed table with means and standard deviations.
4. **Finish change C on an invariance that exists** (mirror, in progress) — an open question
   about whether the head can be improved without labels, not a blocker for the benchmark.
5. **Train the core on Multi-dSprites at grid 64** to separate "grid" from "training data" in
   the cross-dataset story. Lowest priority; the transfer rows already carry the claim.

## 5. The final report, when it exists

Both models, three metrics, per-seed values plus mean and standard deviation, side by side on
the same image ids. With it: the final config, the split manifest version, the evaluation
contract version, the code commit, the reproduction commands, checkpoint identifiers, known
limitations — and the evidence that the score came from classifier masks on spikes, not from
theta or PLV.
