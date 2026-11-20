# PV2_0045 — end-to-end with a frozen graph collapses, because the graph is image-conditioned

**failed.** Parent `PV2_0044`. Seed 0, validation 300, window 1024, kernel sigma
1.0.

## The reasoning, and where it was wrong

Two measured results looked complementary:

| | result |
| --- | --- |
| end-to-end + reconstruction (`PV2_0040`) | 0.6779, the only route that changes the features at all |
| frozen graph (`PV2_0044`) | 0.7250, rescues the weak seed |

and the peer had diagnosed the end-to-end failure as "harmful core
co-adaptation" (SW0068), which freezing the graph should prevent. Their SW0069
froze the **whole** core and scored below their baseline, so freezing only the
graph looked like the untested middle.

## Result

| | fg_ari | foreground_iou | matched_object_iou | groups/image |
| --- | --- | --- | --- | --- |
| frozen features + frozen graph (bar) | **0.7235** | 0.7164 | 0.4935 | 5.49 |
| end-to-end + reconstruction, graph free | 0.6779 | 0.6651 | 0.4692 | 6.27 |
| **end-to-end + reconstruction + frozen graph, lr 1e-4** | **0.0111** | 0.0045 | 0.0007 | **0.07** |
| same, lr 1e-5 | 0.1256 | 0.0791 | 0.0127 | 1.95 |

The lr 1e-4 arm is effectively dead at 0.07 groups per image. Both are far below
either component alone.

The freeze itself worked -- all 7 graph tensors are bitwise equal to the source in
both runs -- so this is a real failure, not an inert flag.

## Why: the two techniques share an assumption

The graph is **image-conditioned**: `graph_generator` takes the features as input
and produces the coupling. Freezing its parameters while end-to-end training
changes the feature distribution leaves a graph tuned to a distribution that no
longer arrives.

The peer's SW0072 froze the graph with the features held fixed, which is
coherent. This combination froze the graph *while deliberately changing its
input*, which is not. The reasoning that the two results "fill each other's
failure" missed that both rest on the same assumption -- a fixed feature
distribution -- so they cannot be stacked.

## Where that leaves the 0.9 route

The headroom is still where `PV2_0034` measured it: fg_ari 0.9754 with features
that separate objects, against 0.7250 now, so 0.25 of it is unclaimed and all of
it is in the features.

Every route to those features has now been tried and closed:

| route | outcome |
| --- | --- |
| DINOv2 features | worse, and already measured here at 0.0897-0.1072 end to end |
| label-free feature clustering | 0.7239 -> 0.6965 |
| self-bootstrapping from the model's groups | +0.006, circular |
| encoder capacity (depth 2, 3) | diverges or degrades |
| slot-term tuning | the defaults were already optimal |
| end-to-end + reconstruction | **0.6779 -- works, but below the frozen-feature best** |
| end-to-end + reconstruction + frozen graph | this, collapses |

So the one route that trains the features at all reaches 0.6779, below the 0.7250
that frozen features now give. **There is no path to 0.9 in hand**, and the
remaining statement that the evidence supports is a specification rather than a
method: within-object feature spread at or below about 0.1 with the present
between-object margin reaches 0.8935, and doubling the margin reaches 0.9754
(`PV2_0035`/`PV2_0036`), with the open problem being an objective that produces
those features without labels.
