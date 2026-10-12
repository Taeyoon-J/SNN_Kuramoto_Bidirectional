# PV2_0071 — the generator's forward-pass knobs are already near their best

**completed. Eighteenth route closed, and half of the sweep measured nothing — for a
reason worth recording.** Parent `PV2_0066`. Inference only, `BIM6_s1`, validation 300,
contract v1.

## Why it was worth running

`PV2_0066` corrected the measurement this project had been steering by: the coupling
graph carries 94% of the oracle's gain, reaching 0.9671 / 0.9353 / 0.7584 at predicted
foreground 0.1121. Seventeen routes had tried to improve the generator's **input**.
Nobody had touched the generator's **own** parameters — and the reason that mattered is
the correction itself: the current values were settled while `PV2_0005` had me believing
the graph did not matter, so they could simply have been undertuned.

## Result

| knob | value | fg_ari | fg_iou | obj_iou | pred_fg | d fg_ari |
| --- | --- | --- | --- | --- | --- | --- |
| **current** | — | **0.7239** | **0.7251** | 0.4951 | 0.1290 | — |
| top-k | **8** | **0.7301** | 0.6850 | 0.5040 | 0.1440 | **+0.0062** |
| top-k | 16 / 64 / 128 | 0.7238 / 0.7228 / 0.7230 | | | | ~0 |
| geodesic steps | **2** | 0.7275 | 0.7185 | **0.5067** | 0.1301 | +0.0036 |
| geodesic steps | 1 / 5 / 8 | 0.7191 / 0.7194 / 0.7077 | | | | negative |
| geodesic contrast | 0.5 / 1 / 4 / 8 | **0.7239 every time** | | | | **0.0000** |
| spatial decay | 0.1 / 0.2 / 0.6 | **0.7239 every time** | | | | **0.0000** |
| oracle features (diagnostic) | — | 0.9671 | 0.9353 | 0.7584 | 0.1121 | +0.2432 |

**The two best changes are trades, not gains.** top-k=8 buys +0.0062 fg_ari for -0.0401
foreground IoU; geodesic steps=2 is +0.0036 fg_ari and +0.0116 obj_iou for -0.0066
fg_iou — all within the seed spread of earlier 3-seed means (0.0022-0.0045). The current
point is where it should be.

## Half the sweep measured nothing, and that was my mistake

`geodesic_contrast` and `spatial_decay` returned **identical values to four decimals** at
every setting. Not a wiring bug: both are **learned parameters stored in the
checkpoint** —

```
graph_generator.geodesic_contrast   2.0523
graph_generator.spatial_rate       -0.2093
```

`graph_generator.py:97` makes contrast an `nn.Parameter`, and line 91 does the same for
the spatial rate, so the constructor argument only sets an **initial** value and
`core.load_state_dict()` then overwrites it with the trained one. Passing them at
evaluation cannot change anything, and **training has already optimised both**. Varying
them requires retraining, and since they are free parameters under the existing
objective, there is little reason to expect a hand-chosen value to beat the learned one.

So eight of the sweep's rows were informative and seven were vacuous. `geodesic_steps`
and `top_k` respond because they are plain Python values, not state.

## What it leaves

The generator's forward-pass parameters were not the missing piece. Combined with
`PV2_0067` (transforms to the graph alone), `PV2_0068` (reconstruction training the
graph) and `PV2_0069` (the graph fed its own groups), every part of the graph route has
now been probed: its input, its training signal, its own parameters, and its behaviour
under self-reference. The headroom is real — 0.9671 with oracle input — and it is
reachable only by supplying instance identity, which no label-free procedure tried here
produces from single static renders.
