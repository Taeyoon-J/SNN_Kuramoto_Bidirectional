# SW0076 - preserve the trained image-conditioned graph

SW0072 established the frozen seed0-trained graph as an effective causal
intervention. SW0073-SW0075 show that encoder tuning can trade foreground
localization against grouping even with the core frozen. SW0076 adds a direct
graph-output distillation term while retaining SW0068's binding and RGB
reconstruction objectives.

For each training batch, the frozen SW0072 seed1 core computes a reference
adjacency `A_ref = graph_generator(gamma_cached)`. The learned encoder produces
`gamma_current`, and the same frozen graph generator computes
`A_current = graph_generator(gamma_current)`. Training adds
`graph_output_anchor_weight * mean((A_current - A_ref)^2)`. Gradients flow
through `A_current` into the encoder; the complete core and graph remain frozen.
This anchors the actual image-conditioned graph behavior rather than gamma
coordinates or a downstream activity summary. No masks or object counts enter
the prediction or training loss.

The default weight is zero, preserving the existing SW0068 path. A nonzero
weight requires `--core-checkpoint` and `--freeze-core`; the trainer asserts
that all core tensors remain bitwise unchanged. The anchor reference is
computed under `no_grad`, while the candidate output stays differentiable.

The seed1 stage1 pilot uses the same SW0075 data, encoder, core, objective,
learning rate, and 5-epoch recipe. Only the graph anchor weight changes. The
coarse pair is `1000` and `10000`. On 32 real training images, the prior
SW0073 weight-1 encoder changes the frozen SW0072 graph by MSE `1.34899e-4`,
which contributes about `0.135` and `1.349` loss units at these weights. The
SW0075 activity-anchor encoders have much smaller graph MSE (`4.34e-6` and
`1.92e-7`), confirming that this measures a distinct invariant. Per-step raw
anchor MSE is logged; these remain coarse pilot settings.

Run one arm on an idle GPU0 or GPU1:

```bash
bash collaborative_test/SW_0076_graph_output_anchor/run.sh 0 g1k
bash collaborative_test/SW_0076_graph_output_anchor/run.sh 1 g10k
```

After training, run the fixed 32-image aligned validation pilot:

```bash
bash collaborative_test/SW_0076_graph_output_anchor/evaluate.sh 0 g1k
bash collaborative_test/SW_0076_graph_output_anchor/evaluate.sh 1 g10k
```

Both scripts refuse existing outputs and check the selected GPU is idle. Server
unit tests, syntax checks, and a two-step real-data/checkpoint smoke passed; the
smoke changed the encoder while core and graph parameters remained exactly
fixed. The two arms were launched on GPUs 0/1. Advance only if a candidate
improves all three metrics over the SW0072 seed1 checkpoint
`.70876/.46810/.50988` under the same 32-image short pilot.
