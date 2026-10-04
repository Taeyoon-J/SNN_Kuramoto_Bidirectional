# SW_0001 — spike synchrony readout

Status: completed validation probe; no model weights or core computation changed.

## Hypothesis

The starting spatial-components classifier returns one object for every evaluated image because time-mean sigmoid membrane activity is above 0.5 almost everywhere. Spike rhythm may still contain pairwise object information. A fixed-count clustering of centered spike traces could recover more than one object.

## Procedure

Use the existing seed-0 baseline checkpoint. Run 256 time steps, discard the first 64 for similarity, and compute the absolute centered cosine similarity between oscillators' returned spike traces. Apply normalized spectral clustering with a fixed k=8 and designate the largest cluster as background. Other clusters become object IDs. The true object count is never supplied to the classifier. Evaluate on validation IDs 1320–1639.

Candidate sources and k=4, 6, 8 were screened on the first 16 validation images. The aggregate spike affinity with k=8 was then checked on the full 320-image validation split. A component-wise product affinity with k=6 was retained as a contrast. The full-validation scores are below.

| Readout | Patch FG-ARI | Foreground IoU | Matched object IoU | Predicted foreground fraction |
|---|---:|---:|---:|---:|
| Aggregate spike synchrony, k=8 | 0.136286 | 0.271997 | 0.145604 | 0.458057 |
| Product of four component synchronies, k=6 | 0.055590 | 0.282826 | 0.121303 | 0.169727 |

The same-split baseline spatial-components scores are 0.000000 / 0.216821 /
0.014756 on validation IDs 1320–1639. Thus the fixed-k spike-synchrony readout
improves all three validation metrics against the original classifier with
identical model weights and examples. The separate reference test result
(0.000000 / 0.207996 / 0.015036) is not used for this pairwise comparison.
The reproducible evaluation is `../evaluate_fixed_split.py`; full same-split
results are stored under the baseline checkpoint's `validation_fixed_split`.

## Diagnostic

On the first 16 validation images, mean phase PLV was 0.444 within a true object and 0.388 across true objects. Spike synchrony was 0.462 within an object and 0.389 across objects; background-background synchrony was 0.679. This indicates object information survives into spike timing, although separation is weak. The original classifier returned exactly one object per image in this diagnostic set.

Next test: measure SNN gradient connectivity under the phase-only objective, then introduce a narrowly scoped spike-path training signal. Keep this readout fixed while testing the loss.
