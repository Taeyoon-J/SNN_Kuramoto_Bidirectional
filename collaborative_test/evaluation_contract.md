# Evaluation contract

**Version 1.** Raise this version on any change below, and re-evaluate both
models when it changes.

Scoring code: `collaborative_test/evaluation.py`, used unmodified by both models.

## What is scored

The three metrics it defines, each reported separately, never combined:

| metric | what it measures |
| --- | --- |
| `patch_fg_ari` | foreground ARI: are two foreground patches of the same object grouped together |
| `patch_foreground_iou` | figure/ground: is the foreground region right at all |
| `patch_matched_object_iou` | Hungarian-matched per-object IoU, divided by the true object count |

Predictions and targets are `[B, grid_h, grid_w]` int64 label maps on one grid.
**Nothing is expanded to pixels.** Both models' masks go through the same patch
conversion and the same functions.

## Grid and ground truth

- Grid: **16 x 16**, so `patch_size = (15, 20)` on the 240 x 320 masks of
  `clevr_with_masks`, which divides exactly.
- Target labels come from `clevr_mask_patch`: each patch takes its most frequent
  instance ID, ties going to the smallest ID, so background wins a tie.
- **Background ID is 0** in both prediction and target, and is entity 0 of the
  dataset's mask stack.
- Low-purity patches are kept. No patch is filtered out of the evaluation.

## Predictions

- Every patch carries exactly one integer ID; overlaps are resolved before
  scoring, without reference to the ground truth.
- Unassigned patches are background (0), which is what
  `spatial_components_to_patch_labels` produces for a patch no group claimed.
- An empty prediction is allowed and scores what the metrics say it scores: 0 on
  `matched_object_iou` against a non-empty target, 0 on `foreground_iou`.
- Predicted background sitting on target foreground stays a cluster in
  `patch_fg_ari`; it is not dropped.

## Undefined scores

Per the implementation: `patch_fg_ari` is NaN with fewer than two foreground
patches, `patch_matched_object_iou` is NaN with no target objects. NaNs are
excluded per metric; `valid_count` is reported beside every mean so the
denominators are visible.

## Aggregation

Image-wise mean over the test split (`nanmean`), then the mean and sample
standard deviation over seeds 0, 1, 2. Report per-seed values, the mean and the
std. Never average the three metrics together.

## The final score

**From masks produced by a classifier on spikes or membrane.** A grouping read
from Kuramoto theta or from a PLV matrix is a diagnostic and never the success
score, however it is clustered.

## Oracle results are diagnostics

Anything that used the true object count or the true masks to choose a cluster
count, a threshold or a graph is reported in a separate diagnostics section and
labelled as an oracle. It is never compared against Slot Attention.

This rules out of the headline, among the results already collected: spectral
clustering at the true cluster count ("oracle k"), and the oracle coupling graph.

## Split

`collaborative_test/data/split_manifest.json`, version 1. Fixed image IDs and
order for train, validation and test, used by every experiment and by both
models. Hyperparameters, losses, classifiers and thresholds are selected on
validation only.

The gamma tensor and the mask tensor are indexed by the same image order; an
experiment verifies that before it trains.

## Slot Attention reference

One published checkpoint, not three seeds, so its column is a single-checkpoint
reference and is labelled as such. It was trained on the original CLEVR render
while the test split here comes from `clevr_with_masks`, which is a different
render; that disadvantages it and is stated wherever the comparison appears.
