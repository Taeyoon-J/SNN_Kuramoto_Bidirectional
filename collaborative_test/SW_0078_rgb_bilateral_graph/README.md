# SW0078 - label-free RGB correction at the causal graph bottleneck

SW0065 showed that the image-conditioned graph is the only isolated component
whose seed0-to-seed2 swap rescues all three pilot metrics. SW0072 then made that
intervention the three-seed lead. SW0078 tests a larger structural change at
that identified bottleneck instead of refining failed encoder anchors or
foreground thresholds.

The SW0072 seed1 checkpoint, gamma, Kuramoto/dendritic/membrane dynamics, and
spike classifier remain fixed. For each image, a second 16x16 graph is built
without labels from RGB patch means. Patch colors are normalized by per-image
median and MAD, then a fixed bilateral color-and-grid affinity is reduced to
the learned graph's top-k and coupling scale. The core is evaluated with:

- the unchanged learned graph;
- RGB graph only, as a mechanism diagnostic;
- learned/RGB convex blends with `alpha=0.25` and `0.50`.

Color and spatial sigmas are both fixed at 2.0. No masks, counts, or validation
scores select them. Ground truth is loaded only after all predictions exist.
The opt-in `graph_override` argument leaves the default core path unchanged.

This differs from SW0016: that experiment added an RGB-edge loss at the final
membrane objective and changed training. SW0078 changes only the per-image
coupling graph at the graph-to-Kuramoto bottleneck and directly measures how
the altered connectivity flows through every downstream layer.

The fixed seed1 stage uses IDs1320-1351 and the same T256/settle64 spike readout
as the SW0072 pilot. An arm advances only if all three metrics exceed
`.708757/.468105/.509881`. Run on an idle GPU0 or GPU1:

```bash
bash collaborative_test/SW_0078_rgb_bilateral_graph/evaluate.sh 0
```

The script refuses existing outputs. No server evaluation has been launched.

## Result

| seed1 32-image pilot | FG-ARI | foreground IoU | matched-object IoU |
|---|---:|---:|---:|
| learned graph baseline | 0.708757 | 0.468105 | 0.509881 |
| learned 75% + RGB 25% | 0.680919 | 0.422639 | 0.493573 |
| learned 50% + RGB 50% | 0.686318 | 0.441609 | 0.475547 |
| RGB graph only | 0.568771 | 0.418120 | 0.353497 |

Every RGB intervention lowers all three metrics. The default row exactly
reproduces SW0072, confirming that the opt-in override did not change the base
path. RGB-only activation still propagates and produces `4.41` groups/image,
but its object masks are substantially worse. Coarse blending does not rescue
it. Stop this prior rather than tune color/spatial sigmas or alpha.

This strengthens the graph mechanism claim: SW0072's gain comes from the
learned image-conditioned topology, not from adding a generic color-distance
graph. The next graph test should preserve learned high-confidence edges and
remove ambiguous edges rather than replace them with RGB connections.
