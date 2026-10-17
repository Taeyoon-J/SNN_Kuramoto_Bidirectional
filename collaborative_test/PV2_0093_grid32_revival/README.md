# PV2_0093 — what keeps the 32x32 spike path alive

**Status:** completed. **Parent:** `PV2_0089`. **Contract:** v2.
**Hypothesis:** `PV2_0089`'s 32x32 spike path died because a hyperparameter tuned for 256
oscillators does not transfer to 1024. Two candidates, one variable per arm.
**Success criterion:** the collapse trajectory, not a score — grid 16 ends at
`spike_plv_collapse` 1.90 and `PV2_0089` died at 9.210340 = log(10^4), so 8 epochs
separate a surviving arm from a dying one.

## Result

Each arm 8 epochs, seed 0, 1000 training images, identical but for the one variable.

| arm | collapse trajectory | verdict |
| --- | --- | --- |
| control, as `PV2_0089` | 4.26 3.75 3.45 3.04 2.85 2.84 3.13 **5.69** | **dying** |
| **A: population threshold k=0.5** | 4.72 4.03 3.76 3.43 3.20 2.88 2.66 **2.58** | **alive, monotone** |
| B: spatial decay 0.175 at training | 4.32 3.70 3.38 2.96 2.87 3.03 3.94 **8.94** | **dead** |

**Hypothesis A is confirmed and B is refuted.** The population threshold is the only arm
whose collapse falls monotonically; the control turns upward at epoch 7 and B overshoots
the control and reaches the dead-membrane value within 8 epochs.

## What this says

The thing that did not transfer is the **transduction**, not the graph.
`--membrane-vth 0.06` is an **absolute** threshold tuned at 256 oscillators; at 1024 the
membrane distribution sits below it, the neurons stop crossing, the spike-PLV term loses
its gradient and the path dies. A population threshold places it relative to the batch's
own statistics and the path survives. This agrees with `PV2_0089`'s diagnosis, where
`plv_mean` was 0.7101 in every row — the phase side was healthy throughout.

**B mattered to test and is now closed.** `--graph-spatial-decay` is measured in grid
units and a grid-32 step covers half the image a grid-16 step does, so 0.35 looked like
twice the intended decay. It is not the cause, and halving it makes the collapse worse
than the control. `PV2_0089`'s eval-time test of this was a no-op — `spatial_rate` is an
`nn.Parameter` restored by `load_state_dict` — so this arm is the first real test of it.

## Cost of the earlier decision

The arm stopped at 02:15 on 2026-10-14 to free the GPU was B's two-variable cousin
(decay 0.175 **and** geodesic steps 4). Stopping it saved ten hours and cost the control;
separating the variables here cost six hours and gave a cleaner answer than that arm
would have, since it would have confounded two changes.

## Next

`PV2_0094` trains the full 40 epochs with the population threshold, then evaluates and
renders the masks.
