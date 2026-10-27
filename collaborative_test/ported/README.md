# ported/ — the peer's `foreground_induced_merge`, and what verifying it required

Copied from `patch_final` `snn_kuramoto_bidirectional/sw240/`, which already carried the
peer's SW0240 source. Nothing was requested or transferred for this; the code was in the
repository.

| file | source | change |
| --- | --- | --- |
| `signed_energy.py` | `sw240/signed_energy.py` | `native_nodes` exposed as a parameter, default unchanged |
| `partition.py` | `sw240/partition.py` | relative import flattened; `native_nodes` passed through |
| `test_port.py` | the peer's own test plus four more | new |

## What it does

```
energy = (1 + native_nodes * (A + Aᵀ) / (2 * coupling_gain)) * logit(Q)
```

Greedy agglomeration over **four-connected foreground patches**, merging the highest
positive-gain pair until no positive gain is left. `Q` is the spike affinity, `A` the
image-conditioned graph. The sign is carried entirely by `logit(Q)`, so **a pair can merge
only when Q > 0.5**; the adjacency factor is ≥ 1 for A ≥ 0 and scales magnitude only.

The function asserts its own postcondition: the foreground support is bit-identical before
and after. It decides how foreground groups, never how much of the image is foreground.

## Why we wanted it

Our Multi-dSprites failure is fragmentation and nothing else — 4.24 predicted groups per
true object, with objects-per-group at 1.02, so we never merge two objects. Six candidates
were eliminated by measurement first (`PV2_0128`): the grid, the readout settings, sprite
overlap, object/background contrast, within-image uniformity, and the dataset variant.

`--min-group-size 32` did cut the group count, 65.37 → 2.16, which is close to the true 2.37
objects. It also cut predicted foreground from 0.1294 to **0.0532** against a true 0.0880:
it reached the right group count by throwing away half the real objects. The merge stage
cannot fail that way, because preserving foreground support is its postcondition.

## The precondition, measured before porting

The merge only fires on Q > 0.5. On our affinity, over four-connected foreground pairs at
grid 64:

| pair type | n | mean | median | above 0.5 |
| --- | ---: | ---: | ---: | ---: |
| same object | 8,523 | 0.4990 | **0.6497** | **51.3%** |
| different object | 226 | 0.0317 | 0.0000 | 4.0% |

Both conditions hold. **Our `--synchrony-threshold 0.05` was discarding this**: half of all
within-object pairs sit above 0.5 and the readout cut at 0.05 and stopped.

## Verification

`test_port.py`, 5/5. The first is the peer's own test from `patch_final`
`tests/test_sw240.py`, verbatim — if our copy fails the test written for the original, it is
not the same function. The other four are what a port needs:

| check | why |
| --- | --- |
| the peer's test, verbatim | fidelity to the original |
| foreground support preserved on a random 8×8 | the postcondition under inputs their test does not cover |
| four-connectivity respected at Q = 0.99 everywhere | disconnected blobs must not merge even when every pair looks mergeable |
| Q < 0.5 leaves a pair split | the sign convention is what we measured against |
| `native_nodes` changes magnitude, never a sign | decides whether the hardcoded constant matters |

## The one deviation, and why it is a flag rather than a decision

`NATIVE_NODES = 1024` is hardcoded in the source. That is the **grid-32** node count, and
the peer runs this at **grid 64**, where the count is 4096, without changing it — so the
adjacency term carries a quarter of the weight the formula implies at that grid.

Whether that is deliberate is unknown, so it is `--induced-merge-native-nodes`: `-1` uses
this run's actual `grid²`, and `1024` reproduces their behaviour exactly. The fifth test
shows the choice **cannot flip a merge decision**, only reorder the agglomeration, so both
settings are safe to measure and neither is assumed correct.

## How it is wired

`evaluate_model.py --induced-merge`, applied after `spatial_components_to_patch_labels`,
per image, with the raw spike affinity recomputed one image at a time so the grid-64
`[B,N,N]` tensor is never materialised twice.

**The graph A is passed as zeros.** The core does not retain it after the rollout, and the
fifth test shows the adjacency term never changes a sign — so omitting it leaves the set of
mergeable pairs identical and only neutralises the ordering. Supplying A means storing it in
the core, which is a separate change worth making only if this arm moves the number.
