# STATUS

Updated 2026-10-04. Branch `patch_v2`, commit `a6740b8`. Contract v1, split v1.

## Where the goal stands

Goal: `patch_foreground_iou` at or above 0.700, from classifier masks on spikes.

Best confirmed: **`PV2_0004`, foreground IoU 0.4967** (test, 3 seeds, std 0.0872),
with `fg_ari` 0.5105 and `matched_object_iou` 0.3689 -- the first setting above
the `PV2_0001` baseline on all three. 0.203 short of the goal.

Against the Slot Attention reference (one published checkpoint, not a 3-seed
training mean): foreground IoU and matched-object IoU beat it, foreground ARI is
short by 0.109.

## What the ceiling measurement changed

`PV2_0005` re-measured the oracle ceiling: a ground-truth coupling graph reaches
foreground IoU **0.561**, and the learned graph already reaches about 0.49.

**A perfect coupling graph is worth about +0.06, so 0.700 is not reachable by
improving the graph.** Under the oracle graph foreground ARI is *worse* (0.329 vs
0.511) -- perfect coupling makes grouping worse. The limit is downstream of the
graph, in how spikes become masks. Graph-side work (geodesic rounds, spatial
prior) is close to spent; `--geodesic-steps 5` actively hurts.

Everything from `PV2_0006` on therefore works on the readout and on the spiking
layers themselves.

## Resume point

Nothing below needs re-running. Checkpoints live under `/work/USERS/tkim1/runs/`.

Running on frontier, detached -- these survive the loss of a driving session:

| what | where | state |
| --- | --- | --- |
| `PV2_0006` -- `--spike-plv-weight` 1 / 5 / 20 on the `d35` graph | `$R/SPLV_w{1,5,20}`, log `$R/pv6.log` | training, 18/40 epochs |
| queue round B -- readout knobs, evaluation only, on `PV2_0004`'s seed-0 checkpoint | `$R/queue_results.txt`, log `$R/queue.log` | waits for training to clear |
| queue round C -- slower dendrite/membrane time constants | `$R/SLOW{D,M,B}` | after round B |

To resume: read `$R/pv6.log` and `$R/queue_results.txt`, then write up whatever
beat its reference. The queue does not commit or push by design -- results are
recorded only after they have been read and analysed.

## Standing cautions

- A session's background waiters die with the session; only the detached remote
  work continues. Nothing re-invokes a session on its own.
- Per-component synchrony is a **product over four components**, so its
  thresholds sit near 0.002-0.030, two orders of magnitude below a single
  correlation's. A threshold carried across readouts has already produced one
  wrong conclusion and one wrong ceiling.
- Keep stderr in sweeps. A suppressed argparse error from a flag that did not
  exist came back as an empty table and was nearly read as a result.
- Single-seed validation numbers are candidates, never findings. Two findings
  have already been retracted for being reported at one seed.
