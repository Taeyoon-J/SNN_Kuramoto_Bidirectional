# PV2_0081 — no component freeze helps, 1.9% of the parameters suffice, and `PV2_0044` was variance reduction

**completed. Twenty-seventh route closed, with two findings that are not score gains and
one correction to my own account.** Parent `PV2_0080`. Trained runs from `BIM6_s1`, seed
0, 40 epochs, validation 300, contract v1.

## Why it was run, and the parameter counts that sharpened it

`PV2_0080` had reframed `PV2_0044`'s graph freeze as **constraining the solution** rather
than improving the objective, since six changes reached a lower loss with a worse score
on both sides of the operating point. Counting trainable parameters then showed how small
that constraint was. Of 200,114 state-dict entries only **69,042 are trainable** — the
other 131,072 are buffers, `sc` and `grid_distance` at 65,536 each:

| component | trainable | share |
| --- | --- | --- |
| `kuramoto` | 67,585 | **97.9%** |
| `dendric_layer` | 1,036 | 1.5% |
| `membrane_layer` | 256 | 0.4% |
| `graph_generator` | **132** | **0.19%** |
| `gamma_channel_proj` | 32 | |
| `gamma_phase_gain` | 1 | |

So `PV2_0044` froze 132 of 69,042, and `kuramoto` — 97.9% of the mass — had never been
frozen. `--freeze-components` now takes any list of top-level groups, with unknown names
and an all-frozen request both rejected.

## Result

| frozen | fg_ari | fg_iou | obj_iou | pred_fg | trainable | loss |
| --- | --- | --- | --- | --- | --- | --- |
| **nothing** | **0.7239** | **0.7251** | 0.4951 | 0.1290 | 69,042 | **17.86** |
| graph only (`PV2_0044`) | 0.7235 | 0.7164 | 0.4935 | 0.1296 | 68,910 | — |
| `kuramoto` | 0.6858 | 0.7100 | 0.4798 | 0.1292 | 1,457 | 16.00 |
| `dendric_layer` | 0.6316 | 0.7047 | 0.4517 | 0.1118 | 68,006 | 16.63 |
| `membrane_layer` | 0.6827 | 0.7089 | 0.4798 | 0.1284 | 68,786 | 16.21 |
| **`kuramoto` + `graph_generator`** | **0.7249** | 0.7131 | 0.4956 | 0.1305 | **1,325** | 15.70 |

**No freeze beats leaving everything trainable** — the best is +0.0010, noise.

## Finding 1: 1.9% of the parameters match full training

`kuramoto` and `graph_generator` frozen together leaves **1,325 of 69,042 parameters
(1.9%)** training — only the dendrite, the membrane and the gamma projection — and it
scores **0.7249** against the full-training 0.7239. The 67,585 `kuramoto` parameters
contribute nothing measurable once their graph is held with them.

And the pairing matters: freezing `kuramoto` **alone** drops to 0.6858, because the
graph's 132 parameters then drift away from the dynamics they were fitted to. That is the
same incoherence `PV2_0045` found when it froze the graph while switching the encoder's
input — **the graph and the Kuramoto state have to move together or not at all.**

## Finding 2: `PV2_0044` reduced seed variance; it did not improve a seed

I wrote in `PV2_0080` that the graph freeze was "the one intervention that raised the
score" and read it as constraint being the lever. At the **same seed** that is not what
happened: `FZG_s0` scores **0.7235** against the untouched **0.7239**. `PV2_0044`'s
+0.019 was in the **3-seed mean**, by rescuing the weak seed from 0.6639 to 0.7235 — it
made the worst seed as good as the others rather than making any seed better.

So "constraint is the lever" is too strong and is corrected here. The graph freeze's value
is **seed-variance reduction**, which is also why `PV2_0044`'s own record notes the three
seeds then share one graph and are not three independent runs.

## The loss pattern, instances 7 to 10

All four arms reached a **lower** loss than the reference — 16.00, 16.63, 16.21, 15.70
against 17.86 — with scores equal or worse. With `PV2_0055`, `PV2_0058`, `PV2_0068`,
`PV2_0077`, `PV2_0079` and `PV2_0080` that is ten instances, and the lowest loss of all
(15.70) belongs to the arm training 1.9% of the parameters. The objective does not rank
these solutions the way the metric does, and nothing in this sweep changes that.

## Where part 3 stands

Unmet at test 0.7603. Twenty-seven routes, thirty-six interventions. What `PV2_0081`
adds is negative but load-bearing: the score does not come from the 97.9% of parameters
in `kuramoto`, it comes from a 1,325-parameter readout path on top of a graph and a
dynamics that can be frozen outright — so there is very little in this model for an
optimiser to improve, which is consistent with every capacity and constraint experiment
failing.
