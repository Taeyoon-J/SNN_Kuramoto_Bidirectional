# Status

Contract v1, split manifest v1, targets v1. Goal: `patch_foreground_iou` above
0.7, currently 0.412 +/- 0.098.

## Baseline

`PV2_0001`, test split, 300 images, three seeds, scored from spike masks:

| metric | mean | std | Slot Attention | |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.4457 | 0.0706 | 0.6195 | below |
| `patch_foreground_iou` | 0.4115 | 0.0980 | 0.1241 | beats |
| `patch_matched_object_iou` | 0.3438 | 0.0924 | 0.0920 | beats |

## In flight when the session lost the server

- `PV2_0002a` -- firing-rate constraint, `--loss-signal spikes
  --spike-rate-weight 1.0 --spike-target-rate 0.2`. Training finished for seeds
  0, 1, 2 (`RATE_s0/1/2`) and a weight-0.3 probe (`RATE_w03`). The rate is now
  pinned: the rate loss sits at 5e-5 for every seed, against a sevenfold spread
  before. **Not yet scored on test.** On validation it looks worse than the
  baseline on all three metrics, so it may buy variance at the cost of mean.
- `PV2_0002b` -- geodesic graph, launched but the server dropped before the
  first epoch landed. Runs `GEO_s3_c2`, `GEO_s3_c5`, `GEO_s4_c2` may or may not
  exist; check before relaunching.

## What the threshold sweep settled

Tuning the classifier's synchrony threshold **cannot reach the goal**. On
validation, foreground IoU peaks near 0.456 at threshold 0.35 and falls on both
sides; below 0.30 everything collapses. The predicted foreground fraction can be
driven onto the target's 0.126, and foreground IoU still only reaches 0.267 --
so the fraction being right is not enough, the foreground has to be in the right
places. The graph is the remaining lever.

## Why the graph

Training on a ground-truth coupling graph, which is a diagnostic and not a
result, gives foreground IoU 0.610 against 0.421 and lifts the phase readout from
0.598 to 0.723.

What the learned graph gets wrong is specific: its edges come from feature cosine
with a Euclidean distance prior, so two objects of the same colour -- alike to the
features, a few patches apart -- stay linked at 0.418 mean edge weight against
0.749 inside an object. 83% of CLEVR scenes contain a repeated colour.

`geodesic_steps` measures patch distance along the image instead of through it,
by min-plus relaxation over a dissimilarity-weighted step cost. Verified locally:
two identical patches three apart get edge weight 0.123 under the geodesic
against 1.141 under the Euclidean prior, while an immediate neighbour keeps
0.458. Gradients reach both the graph projection and the geodesic contrast
parameter.

## Resume

1. `ssh frontier` to restore the multiplexed connection; it expires and the agent
   cannot reopen it.
2. Score `RATE_s0/1/2` on test, finish `PV2_0002a`.
3. Check whether `GEO_*` runs survived; relaunch
   `/work/USERS/tkim1/runs/pv2_0002b.sh` if not.

Everything lives under `/work/USERS/tkim1`; code in `/export_home/tkim1`.
