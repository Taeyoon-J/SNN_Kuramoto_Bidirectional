# PV2_0130 — Tetrominoes at grid 35, where the cell is the pixel

- **status** running: one seed, grid 35, TEST 1000 evaluation chained after it
- **parent** `PV2_0127` (the Tetrominoes TEST row), `PV2_0125` (the ceilings)
- **question** how much of our distance to the published Tetrominoes numbers is a grid we
  chose, rather than anything the model does?

## Why

A literature check corrected the comparison family. On Tetrominoes — 35×35, three objects,
conditions identical to ours — the published unsupervised numbers are **Slot Attention 99.5**,
**IODINE 99.2**, against the synchrony line's SynCx 89 and CtCAE 84. We score **0.9282** on
the TEST split.

But those models emit **pixel** masks with no grid. Ours quantises to a patch grid, and the
ceiling that imposes is measured, not assumed:

| grid | source px / cell | ceiling FG-ARI (TEST 300) | ours |
| ---: | ---: | ---: | ---: |
| 32 | 1.09 | **0.9345** | 0.9282 = **99.3% of it** |
| **35** | **1.00** | **1.0000** | — |

**We are at 99.3% of what grid 32 can express**, so of the 0.066 between us and 99.5, most
is the grid rather than the model. At grid 35 the cell *is* the pixel, the conversion is the
identity, and the handicap is gone. The targets confirm it directly: `cell sizes [1]`,
purity **1.0000**, ties **0.0000**.

This is the same move that took Multi-dSprites from a 0.5619 ceiling to 1.0000. Grid 64 is
impossible here — 35 admits no 64 cells.

## Cost, and why no warm start

`map_grid.py`'s block replication needs the destination grid to be an integer multiple of
the source. 35 is not a multiple of 32, so it does not apply and the model is trained at
this grid: 1225 nodes against 1024, about **1.4× the pairwise cost**, so rather more than
the 608 minutes seed 0 took at grid 32.

**One seed.** If it does not clear 0.9282 there is no reason to spend two more.

## A correction to `PV2_0126`

That record's commit message says it fixed `build_mod_targets.py` writing the manifest from
`ranges` alone. **The fix was never committed.** It existed only in a scratch copy under
`/tmp`, which was cleared, so the server kept the old builder — and building the grid-35
targets reproduced the original bug exactly, dropping `splits` from the Tetrominoes manifest
and breaking grids 16 and 32 that were already finished.

The fix is now in `collaborative_test/build_mod_targets.py`, shipped to the server, and both
datasets rebuilt: **six dataset × grid configurations pass `preflight.py`** — Tetrominoes at
16, 32, 35 and Multi-dSprites at 16, 32, 64.

The lesson is narrow and worth keeping: a claim in a commit message is not a commit. This is
the second time this fault has cost a rebuild.

## What the result decides

- **clears 0.9282 substantially** → the grid was the binding constraint on this dataset, the
  comparison with pixel-level methods was unfair to us by construction, and the other two
  datasets deserve the same treatment
- **does not clear it** → the grid was not the constraint, the model is, and 0.9282 stands
  as what this architecture does on Tetrominoes

Either way it is one seed on one dataset and promotes nothing on its own.
