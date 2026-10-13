# PV2_0087 — computational complexity, ours against Slot Attention

**Status:** completed. **Parent:** `PV2_0061` (the published configuration).
**Hypothesis:** none — this is an analysis of code already in the repository, not an
experiment. Nothing was trained and no metric moved.
**Contract:** v1 for the architecture being counted; the counts themselves are
contract-independent.
**Commit:** the architecture as of `cee0870`.

## Why this is recorded

The user asked for the big-O of both models. It also answers a question the route map
left implicit: whether a finer grid is affordable. It is, barely, and the reason is
structural rather than incidental.

## Where each term comes from

Counted by reading the code, not by profiling.

| term | source | file |
| --- | --- | --- |
| O(S·N³) | min-plus relaxation materialises `[B,N,N,N]` and reduces over the intermediate node | `graph_generator.py:190` |
| O(T·N²·D) | four `bmm([N,N],[N,D])` per Kuramoto step, adjacency kept dense | `kuramoto_layer.py:131-136` |
| O(T·N²) | spike-correlation affinity over the rollout | `evaluate_model.py` readout |
| O(N³) | spectral clustering's eigendecomposition | readout |
| O(P·κ²·3·C) | one valid 3×3 convolution, 8 kernels | `input_layer_generator.py:78` |

Slot Attention, from `baselines/slot_attention/model.py`: four 5×5 SAME convolutions at
64 channels on 128×128, a soft position embedding, three slot-attention iterations over
7 slots of width 64, and a spatial-broadcast decoder of six `Conv2DTranspose` layers run
**once per slot**.

## Asymptotics

| | ours | Slot Attention |
| --- | --- | --- |
| dominant | **Θ(S·N³ + T·N²·D)** | **Θ(K·L·P·κ²·C²)** |
| grows with | patches, **cubically** | pixels, linearly; slots, linearly |
| depends on the evaluation grid | yes — N *is* the grid | **no** — it always emits 128×128 pixel masks |

That last row is the asymmetry worth keeping. Moving from a 16×16 to a 32×32 contract
changes nothing in Slot Attention's forward pass; the 4×4 pooling happens after it. For
us the grid is the state.

## Counts, MACs per image, one forward pass

| component | ours 16×16 (N=256) | ours 32×32 (N=1024) | Slot Attention |
| --- | --- | --- | --- |
| Kuramoto O(T·N²·D) | 1.074e9 (88.3%) | 1.718e10 (76.1%) | — |
| geodesic O(N³) | 5.033e7 (4.1%) | 3.221e9 (14.3%) | — |
| spike affinity O(T·N²) | 6.711e7 | 1.074e9 | — |
| spectral O(N³) | 1.678e7 | 1.074e9 | — |
| encoder | 3.429e6 | 3.429e6 | 5.112e9 (24.0%) |
| broadcast decoder × 7 slots | — | — | 1.591e10 (74.6%) |
| slot attention module | — | — | 1.792e8 (0.8%) |
| position + MLP | — | — | 1.384e8 |
| **total** | **1.216e9** | **2.258e10** | **2.134e10** |

T = 1024 at evaluation. At training (T = 64) ours is 1.432e8 at N=256 and 5.458e9 at
N=1024, where the geodesic rather than the rollout dominates.

- 16×16: Slot Attention costs **17.6×** what we do.
- 32×32: **0.94×** — the crossover sits almost exactly at N = 1000.
- our own 16 → 32 step: **18.6× more expensive for a 4× finer grid**.

Parameters: **69,042** trainable for us (200,114 stored tensors less two 65,536-element
buffers, `sc` and `grid_distance`) against **~889,860** for Slot Attention, so 12.9×
fewer — and `PV2_0066` found 1,325 of ours actually matter.

## The memory wall is separate from the time wall

The geodesic's peak intermediate is `B·N³` floats:

| N | batch 16, float32 |
| --- | --- |
| 256 | 1.07 GB — consistent with the 1.56 GiB OOM recorded during `PV2_0054` |
| 1024 | **68.7 GB** |

This is why 32×32 is blocked today, and it is a memory layout problem, not an arithmetic
one: chunking the relaxation over rows leaves the operation bit-identical and brings the
peak to O(B·N²·chunk). `kuramoto_layer.py:124-130` already carries the matching fix for
the rollout, and its comment names H=1024 as the reason.

## Headroom this exposes

`graph_top_k` is 32 and N is 256 or 1024, so the adjacency has 32 non-zeros per row and
is **computed densely anyway** — `logits.topk(...)` then `scatter_` into a zero matrix
(`graph_generator.py:152-155`). A sparse rollout would be O(T·N·k·D) and a sparse
relaxation O(S·N·k²):

| | dense (today) | sparse |
| --- | --- | --- |
| ours at 32×32 | 2.258e10 | **~1.6e9** |
| against Slot Attention | 0.94× | **13× cheaper** |

So the cubic term is an implementation choice, not a property of the method. This is a
speed result and **not** a route to 0.9 — it changes no metric. It is recorded because
the route map should not carry "a finer grid is unaffordable" as a reason, when the
reason is that nobody has written the sparse path.

## Limitations

MACs, not wall clock: the dense `bmm`s run near peak on a GPU while a sparse rollout
would not, so the 13× is an upper bound on the real speedup. Slot Attention's count is
inference only; its official recipe trains far longer than ours. Neither count includes
the backward pass.

## Next

Nothing follows for the score. For the 32×32 request the blocking items are the geodesic
chunking above and the gamma recipe (see `PV2_0088`).
