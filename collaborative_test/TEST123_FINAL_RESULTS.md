# Experiments 1, 2 and 3: final verified comparison

All requested training and evaluation deliverables are complete. Each
experiment has seeds0/1/2 and full evaluations at epochs1/3/10. All predictions
were scored against our HDF5 validation IDs1320-1639 (320 images), with the same
16x16 patch metrics. Ground-truth masks were used for scoring, not prediction.
The user requested pausing goal mode after these results are shared; no further
model improvement or training is authorized by this request.

## Experiment definitions

1. **Our model on our data:** 70,000 unique HDF5 training IDs0-999 and
   1640-70639, ten passes, retaining the strongest frozen-encoder/frozen-graph
   recipe. Checkpoints1/3/10 measure learning duration.
2. **Our model on released Slot data:** TFDS CLEVR3.1.0 train, released-code
   skip512 and at-most-six-object filter, crop/resize. This yields 34,766
   training images. Existing encoder, normalization and frozen graph are
   retained. Evaluation uses our mask-bearing HDF5 data, as explicitly requested.
   This is cross-domain downstream training, not the entire model trained
   from scratch exclusively on released Slot data.
3. **Slot Attention on our data:** official architecture from scratch, exactly
   the same 70,000 training IDs as experiment1, ten passes (700,000 image
   exposures per seed), batch32, 11 slots and three iterations. Each pass has
   2,188 updates including the final partial batch. The training images and
   pass counts match experiment1; loss, optimizer schedule, batch size and
   pretraining history differ between architectures.

IDs1000-1639 were excluded from the large training sets. TFDS released images
do not include instance masks, so experiment2 is evaluated in our domain.

## Three-seed means

| Experiment | Epoch | FG-ARI | Foreground IoU | Matched-object IoU |
| --- | --- | --- | --- | --- |
| 1: our model / our 70,000 | 1 | 0.774323 | 0.548028 | 0.603848 |
| 1 | 3 | 0.752242 | 0.548378 | 0.614339 |
| 1 | 10 | 0.677605 | 0.469147 | 0.561948 |
| 2: our model / released Slot data | 1 | 0.769129 | 0.486582 | 0.608472 |
| 2 | 3 | 0.738899 | 0.569473 | 0.596554 |
| 2 | 10 | 0.740027 | 0.566435 | 0.603881 |
| 3: Slot / our 70,000 | 1 | 0.524048 | 0.212999 | 0.158664 |
| 3 | 3 | 0.684707 | 0.215308 | 0.195487 |
| 3 | 10 | 0.774933 | 0.203589 | 0.206937 |

All entries above use three independently trained seeds. The earlier
experiment3 epoch10 mean `.853923/.200172/.212717` used only seeds1/2; its
FG-ARI falls to `.774933` after including seed0 (`.616953`). The completed
individual epoch10 Slot FG-ARIs are `.616953/.836794/.871053`.

The existing official **single pretrained transfer checkpoint**, evaluated on
the same validation IDs, scores `.894641/.223511/.245968`. Its provenance and
mask contract are in
[SW0046 reference](SW_0046_aligned_slot_audit/results/seed0_validation1320_1639/evaluation_summary.json).
It is not a three-seed training mean and does not have matched training budget.

## Interpretation

At ten passes over the same 70,000 training images, experiment3 has higher
FG-ARI than experiment1; experiment1 has higher foreground and object IoU.
Our selected early epoch1 checkpoint averages `.774323/.548028/.603848`:
its FG-ARI is nearly equal to Slot's epoch10 mean (difference -0.000610), while
both IoUs are higher. This small ARI difference has not been established as
statistically significant. Comparing those rows uses different training
durations and the stated differing pretraining histories.

Experiment2 also exceeds the public pretrained Slot checkpoint on both IoUs
but has lower FG-ARI. These results do not show our large-data model beating
Slot on all three metrics. Increasing training duration helps Slot's FG-ARI
substantially here, while our model's FG-ARI falls. These experiments change
image diversity and optimizer updates together and do not isolate a causal
effect of unique-image count alone.

The earlier SW0072 2,500-image mean `.781407/.478038/.616746` remains a separate
small-data reference. It must not be presented as experiment1's 70,000-image
result. These runs do not independently prove a count-classifier milestone,
causal stage-wise activation improvement or a rhythmic-gating ablation result.

## Artifacts and verification

- [Experiment1 results](SW_0090_large_unique_scale/results/full320/summary.json)
  and [three-image visualization](SW_0090_large_unique_scale/results/full320/visualizations/overview_all_three.png).
  Original images, all individual GT/predicted masks, instance maps and grids
  for fixed IDs1320/1321/1322 are present and PNG integrity was checked.
- [Experiment2 results](SW_0092_cross_dataset_training/results/our_on_official/summary.json)
  and [verification](SW_0092_cross_dataset_training/results/our_on_official/verification.json).
- [Experiment3 results](SW_0092_cross_dataset_training/results/slot_our70000/summary.json)
  and [verification](SW_0092_cross_dataset_training/results/slot_our70000/verification.json).
- [Machine-readable comparison](SW_0092_cross_dataset_training/results/test123_final_comparison.json).

Verification covers 27 evaluations: correct checkpoint/epoch and seed, exact
held-out IDs, 320 valid scores for every metric, per-image records, no GT during
prediction, and independently recomputed means. Experiment3's three training
logs each show all ten epochs and 21,880 updates; their protocols match
experiment1's actual training-ID manifest. All requested GPU jobs have exited.
Experiment2 wrapper markers were recovered only after checking completed
trainer logs, finite checkpoint tensors and final/epoch10 tensor equality.
No training was repeated to repair those wrapper markers.
