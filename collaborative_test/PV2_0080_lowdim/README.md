# PV2_0080 — less capacity fails too, which refutes my own explanation

**completed. Twenty-sixth route closed, and it corrects `PV2_0079`'s account of why five
changes reached a lower loss with a worse score.** Parent `PV2_0079`. Trained runs, seed
0, 40 epochs, validation 300, contract v1.

## The prediction

`PV2_0079` had closed osc_dim 8 and 16 and coupling 64 and 1024, and explained the
recurring "lower loss, worse score" pattern as capacity: bimodality, balance and target
density are statistics of the synchrony matrix, so **more** capacity makes a
right-statistics wrong-groups matrix easier to produce. That explanation predicts the
opposite direction should help, and I had only swept upward — osc_dim below 4 had never
been run.

## Result: it does not, and the explanation was wrong

| setting | fg_ari | fg_iou | obj_iou | pred_fg | final loss |
| --- | --- | --- | --- | --- | --- |
| **bar: osc_dim 4, k 256** | **0.7239** | **0.7251** | **0.4951** | 0.1290 | **17.86** |
| osc_dim 3 | 0.6783 | 0.6892 | 0.4688 | 0.1412 | **16.21** |
| osc_dim 2 | 0.0033 | 0.0000 | 0.0000 | 0.0000 | 86.19 |
| k 128 | 0.4678 | 0.6197 | 0.3761 | 0.1069 | 17.04 |
| k 512 | 0.3694 | 0.1696 | 0.2334 | 0.2835 | 16.71 |

**osc_dim 4 and k 256 are a genuine two-sided optimum.** osc_dim 2 lands in the
dead-membrane regime (loss 86, predicted foreground 0.0000) that `PV2_0054` diagnosed.

And **osc_dim 3 reached loss 16.21 — lower than the bar's 17.86 — for fg_ari 0.6783.**
That is the sixth instance of the pattern, and it is on the *small* side of the optimum.

## So the correct statement is not about capacity

`PV2_0079` attributed the pattern to extra capacity creating degenerate solutions. That
cannot be right, because reducing capacity does the same thing. Both departures from
osc_dim 4 reach a **lower** loss and a **worse** score:

| setting | loss | fg_ari |
| --- | --- | --- |
| osc_dim 3 | 16.21 | 0.6783 |
| **osc_dim 4** | **17.86** | **0.7239** |
| osc_dim 8 | 14.19 | 0.5670 |

**The operating point has a relatively high loss and the best score**, and moving either
way trades loss for score. The honest statement is that **the objective and the metric
are decoupled around the operating point** — not that one regime is degenerate and
another is not. All six instances fit this, including the four outside the dynamics
(`PV2_0055`, `PV2_0058`, `PV2_0068`, `PV2_0077`).

That is a strong constraint on any future work here: **lowering this loss is not
evidence of progress, and tuning toward it is as likely to hurt as to help.** It also
reframes `PV2_0044`'s graph freeze, the one intervention that did raise the score — it
constrains the solution rather than improving the objective's value.

## Where part 3 stands

Unmet at test 0.7603. Twenty-six routes, thirty-two interventions: features 11, graph 8,
geodesic cost 3, readout 2, dynamics 8. One transferred (`PV2_0076`, +0.0070, and only
outside the objective).

What `PV2_0080` adds is that the next step cannot be a search over this objective's
hyperparameters — the operating point is a two-sided optimum and the objective does not
rank solutions the way the metric does. A route to 0.9 would need a different training
signal, and the three label-free candidates are all tested and closed: reconstruction
(`PV2_0039`, `PV2_0068`), cross-view consistency (`PV2_0070`) and self-reference
(`PV2_0069`, `PV2_0073`).
