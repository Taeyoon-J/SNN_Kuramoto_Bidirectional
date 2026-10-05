# SW0065 - causal component swap at the graph-to-Kuramoto bottleneck

SW0064 found that SW0055 seed0 and seed2 receive identical gamma and both
preserve dendritic-to-membrane structure, but their Kuramoto PLV object margin
diverges (`.817` versus `.517`). Seed2's graph affinity is not worse, so SW0065
uses minimal checkpoint swaps to identify which learned part causes the
graph-to-phase failure.

Starting from seed2, three validation-only hybrids independently copy seed0's:

1. gamma channel projection plus phase gain (`drive0`),
2. learned graph generator (`graph0`),
3. Kuramoto parameters (`kuramoto0`).

Base seed0 and base seed2 are evaluated beside them. All use native gamma,
IDs1320-1351, T256/settle64, raw gating, and spike-CC threshold .35. Strict
state-dict loading and one-image real-asset forwards are required before the
32-image pilot. This is a causal diagnostic; a successful swap identifies one
component for the next stability intervention and the other swap directions
stop.

## Result

On the fixed 32-image pilot, seed2 scores `.3895/.2003/.1822`. Replacing only
its trained graph generator with seed0's raises this to
`.7036/.3081/.5376`, gains of `+.3141/+.1078/+.3554`. Replacing the input
projection or Kuramoto parameters does not improve all three metrics.

The learned graph generator is therefore a causal source of the seed failure.
This diagnostic does not make a train-time claim because it swaps a trained
component. SW0066 tests the corresponding train-time intervention by fixing
only graph initialization while keeping it trainable.
