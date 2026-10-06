# SW0088: jointly adapt the encoder and graph while preserving activation flow

SW0065/SW0072 identify the learned graph as the dominant seed bottleneck, while
SW0068/SW0073 show that updating the entire downstream core together with RGB
features creates harmful co-adaptation. SW0074 shows that updating only the
encoder against a completely frozen core is also insufficient. SW0088 tests the
remaining boundary: update the native RGB encoder and image-conditioned graph
together while keeping every downstream Kuramoto, dendritic, membrane, and
spike tensor bitwise fixed.

All arms start from the completed SW0072 seed1 core and registered native RGB
encoder. They use the same 2,500 training IDs, batch 16, seed1, graph/encoder LR
`3e-5`, five epochs, and fixed phase plus 5x component-spike losses. Only the
phase-slot RGB reconstruction weight changes:

| Arm | Reconstruction weight |
|---|---:|
| E | 0.0 |
| F | 0.3 |
| G | 1.0 |

The trainer asserts a nonzero encoder and graph update and exactly zero change
to every non-graph core state tensor. A real one-update preflight precedes each
arm. The fixed pilot uses IDs1320-1351, T256/settle64, membrane threshold `.06`,
and spike connected-components threshold `.35`.

The durable queue waits for SW0085. If SW0084 already has an all-three pilot
advance, SW0088 records a skip so the promising candidate can be expanded
instead. Otherwise it runs E/F/G under the existing GPU policy and writes a
fixed summary without threshold selection.
