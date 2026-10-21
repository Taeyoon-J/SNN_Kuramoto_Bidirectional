# Learning and data saturation tracking

User request, 2026-10-20: future training reports must show when additional
training data or further training stops producing meaningful improvements.

## Required records for future training implementations

Record these separately per seed and checkpoint:

- Number of distinct TRAIN image IDs seen in this stage, and total unique
  images available to the stage. Record inherited foundation training separately.
- Optimizer updates, actual image presentations, complete passes, and active
  training time. Exclude GPU waiting time from active training time.
- Immutable model/optimizer checkpoint identity, evaluation image IDs, native
  patch grid, readout, supervision and all three unchanged evaluation metrics.
- Loss components and gradient health are diagnostics, not substitutes for
  mask FG-ARI, foreground IoU and matched-object IoU.

Future 70k runs should preserve intermediate checkpoints around first-stage
unique-image counts 2,500, 10,000, 35,000 and 70,000, aligned to full optimizer
batch boundaries. After full coverage, preserve checkpoints after passes
2, 4, 6, 8 and 10 when the registered budget reaches those boundaries.
Counts below the training batch size or beyond the registered budget are skipped.
Checkpoint retention and CPU/GPU evaluator hooks must be implemented and
preflighted before claiming that a future run collects a learning curve.

Use a fixed 64-image, evenly spaced subset of the existing 320 validation images
for inexpensive intermediate comparisons, with unchanged inference seeds and
readout. Preserve prediction masks before accessing evaluation GT. Freeze the
milestone schedule before training; record this as repeated validation use.
Evaluate the terminal model and any selected earlier candidate on the full320
board for confirmation. Do not use the untouched final reserve for tracking.

## Reporting saturation

Report score deltas against the preceding milestone and the best previous
milestone, plus paired-image uncertainty. Flag a provisional plateau when
two successive intervals each improve FG-ARI by less than 0.005 absolute.
This is a reporting heuristic, not proof that more data cannot help or an
automatic stopping rule. Report IoU regressions, seed variation, and whether
the apparent plateau survives full320 confirmation. Do not claim statistical
equivalence merely because a confidence interval contains zero.

The within-run first-pass curve confounds additional distinct images with more
optimizer updates. To attribute a benefit specifically to dataset size, use
nested TRAIN subsets from the same foundation initialization, fixed model/loss,
matched optimizer-update budgets, paired seeds and unchanged evaluation data.
Report both dataset size and exposure count; ten passes over70k is700k image
presentations, not700k distinct images.

## Current SW0240 limitation

SW0240 preserves its latest committed boundary, not an archive of every
intermediate model. Existing loss/update history cannot reconstruct historical
evaluation scores. Do not restart it or invent a retroactive learning curve.
Its seed0 terminal score is available; seed1/2 results and the final aggregate
will be tracked separately. Current scheduler-policy changes do not add
intermediate-checkpoint evaluation to this already running experiment.

Report at experiment start and completion, or when the user asks. Do not send
routine epoch/minute updates. This request does not authorize auto-pausing or
changing a running experiment's learning budget based on validation scores.
