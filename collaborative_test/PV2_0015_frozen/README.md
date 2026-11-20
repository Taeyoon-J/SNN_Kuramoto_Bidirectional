# PV2_0015 — the spiking path has no capacity to learn this

**completed. The conclusion here closes the whole line of attack.** Parent
`PV2_0014`. Seed 0, fine-tuned from `BIM6_s0`, validation 300.

## Change

`--freeze-phase-path`: train only `dendric_layer` and `membrane_layer`, freezing
the graph, the gamma projection, the gamma generator and the Kuramoto layer, so
the alignment target cannot move. Plus `--init-from`, since freezing is only
meaningful on top of an already trained phase path.

## A design error first, and how it was caught

The first attempt froze a **randomly initialised** phase path. Theta was noise,
four spiking tensors were asked to match it, and the loss moved 32.4079 to
32.4090 while align weights 10, 40 and 150 produced identical metrics. That is
not a result about the method.

Three checks were added and all three now pass: `--init-from` reports the load,
the gradient norms on the trainable set are printed and a run aborts if they are
all zero, and the phase row is re-measured -- it is byte-identical across weights
(0.6769 / 0.7239 / 0.4867), which confirms the freeze holds.

## Result: gradients arrive, and nothing can be done with them

| | align term | spike fg_ari (sync 0.35) |
| --- | --- | --- |
| `BIM6_s0` | -- | 0.6211 |
| align 10 | 0.510229 -> 0.510236 | 0.6211 |
| align 40 | 0.510229 -> 0.510235 | 0.6200 |
| align 150 | 0.510229 -> 0.510395 | 0.6202 |

Gradients reach the spiking layers -- norms 4.4e-4 to 2.8e-2, non-zero -- and the
alignment term still does not move, at any weight, over 40 epochs at lr 3e-3.

## Why: the transduction point has eight weights

| | parameters |
| --- | --- |
| whole model | 200,114 |
| spiking path | **1,292** (0.6%) |
| `dendric_layer.oscillator_dense` | **8** |

`tau_n` (256x4) and `tau_m` (256) give per-unit leak freedom, but the map from
phase into the spiking layers is `oscillator_dense`, a single `(sin theta, cos
theta) -> osc_dim` linear projection of **8 weights shared across all 256
regions**.

So phase differences -- the thing synchrony is made of -- are compressed through
eight shared weights before any unit-specific processing happens. There is
nothing there to learn with.

**This is why `PV2_0007`, `PV2_0014` and `PV2_0015` all failed.** The
Kuramoto-to-SNN loss is not a badly-trained stage; it is an almost
parameterless one. No loss function can close it. For the research question that
is the substantive result: binding-by-synchrony information survives the
oscillator dynamics and is destroyed at a fixed transduction bottleneck.

## Next, and it is a core change

Give the transduction capacity, which `CLAUDE.md` requires analysing before
making and validating with a small ablation. The minimal version keeps the form
of the equations -- still a linear map of (sin, cos) -- and makes
`oscillator_dense` per-region, 256 x 4 x 2 = 2048 weights instead of 8, so units
can transduce phase with their own gain and offset instead of all sharing one.
