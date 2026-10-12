# PV2_0079 — the Kuramoto dynamics are not the limit either

**completed. Twenty-fifth route closed, and it exhausts the axes I can test inside this
architecture.** Parent `PV2_0072`. Trained runs, seed 0, 40 epochs, validation 300,
contract v1.

## Why here

Twenty-four routes had changed the features (eleven), the graph's input, training signal
and parameters (eight), the geodesic cost (three) and the readout (two). **The Kuramoto
dynamics had never been touched**, and condition 3 of the goal explicitly allows
modifying the architecture while keeping SNN + Kuramoto bidirectional.

`PV2_0072` gave the reason to look: the pipeline **repairs** a damaged grouping (+0.152
fg_ari at heavy corruption), and that repair happens in the rollout, not in the features.
If repair capacity is the lever, the parameter bounding it is the phase space the
oscillators work in — `osc_dim` is 4 while validation images hold **6.32 objects**, so
four dimensions must separate about seven groups. `PV2_0034`'s oracle managed that at
osc_dim 4, but with a **perfect** input, where no separation work is needed.

## Result

| setting | fg_ari | fg_iou | obj_iou | pred_fg | final loss |
| --- | --- | --- | --- | --- | --- |
| **bar: osc_dim 4, k 256** | **0.7239** | **0.7251** | **0.4951** | 0.1290 | 32.49 -> **17.86** |
| osc_dim 8 | 0.5670 | 0.5719 | 0.4338 | 0.1810 | 86.67 -> **14.19** |
| osc_dim 16 | 0.0033 | 0.0000 | 0.0000 | 0.0000 | 87.18 -> **86.07** |
| k 64 | 0.6088 | 0.7122 | 0.4398 | 0.1121 | 24.36 -> 17.17 |
| k 1024 | 0.3331 | 0.2384 | 0.2023 | 0.1121 | 35.88 -> **80.78** |

**osc_dim 4 and k 256 are already the best of these**, and the hypothesis that four
dimensions are too few for 6.32 objects is refuted.

## The losses explain it, and repeat a pattern

**osc_dim 8 reached loss 14.19, well below the baseline's 17.86, and scored 0.5670.** The
objective was satisfied *better* while the score fell — the **fifth** time this project
has seen that: `PV2_0055` (position in the graph, 13.91 vs 17.86 for -0.17), `PV2_0058`
(chroma11 at 17.71 for 0.3599), `PV2_0068` (reconstruction), `PV2_0077` (training under
the superlinear cost, 16.71 for 0.4268), and here.

I read this as a wider phase space giving the PLV terms more degenerate ways to be
satisfied. **`PV2_0080` refutes that**: osc_dim 3 also reaches a lower loss (16.21) with a
worse score (0.6783), so reducing capacity does the same thing and the pattern is not
about capacity. The correct statement is that the objective and the metric are **decoupled
around the operating point**, whose loss of 17.86 is higher than both neighbours' while
its score is better than both.

osc_dim 16 never trained at all — loss 86, the dead-membrane regime `PV2_0054` diagnosed,
confirmed by predicted foreground 0.0000. k 1024 diverged (35.88 -> 80.78). k 64
converged cleanly at 17.17 and still lost 0.115, so weaker coupling costs repair
capacity, which is consistent with `PV2_0072`'s picture.

## What is now exhausted

Twenty-eight interventions across twenty-five numbered routes: features 11, graph 8,
geodesic cost 3, readout 2, dynamics 4. One transferred — `PV2_0076`'s superlinear step
cost, +0.0070 on test, and `PV2_0077` showed even that works only outside the objective.

The obstacle is stated as precisely as I can state it:

1. The oracle numbers measure **transcription**, not discovery (`PV2_0072`).
2. The pipeline repairs spatially **incoherent** damage; its own errors are spatially
   **coherent** object merges, and it is a fixed point on those at 0.727 (`PV2_0073`).
3. The geodesic barrier is two thin walls (5.27x) with a corridor six times cheaper than
   a within-object step, and the cost's shape cannot be tuned to fix that without
   splitting objects (`PV2_0075`, `PV2_0078`).
4. Five separate changes reached a **lower loss and a worse score**, so the PLV objective
   admits degenerate solutions that more capacity makes easier to find.

Part 3 of the goal is unmet at test 0.7603. Reaching 0.9 needs either supervision the
contract forbids, or data this dataset does not contain — motion, or several views of one
scene — because in a single static render nothing distinguishes two adjacent
similarly-coloured objects except their separation, which the pipeline already uses.
