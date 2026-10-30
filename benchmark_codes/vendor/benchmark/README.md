# benchmark/ — the scripts `../benchmark.md` tells you to run

Self-contained copies, taken from `patch_v2`'s `collaborative_test/` at the commit named in
`../benchmark.md`. They import each other from this directory, so run them from here or
with this directory on `PYTHONPATH`. Nothing here imports the model.

Run them in this order.

| # | script | what it does | run it when |
| --- | --- | --- | --- |
| 1 | `bench_encoder.py` | trains the per-dataset input encoder (224 params) by reconstruction on TRAIN only, and dumps features | once per dataset |
| 2 | `build_mod_targets.py` | pixel masks -> patch targets at a grid, with the per-dataset mask axis order and a spatial-coherence assert | once per dataset *and grid* |
| 3 | `build_gamma_grid.py` | features -> the gamma sequence the model consumes, at unit variance | once per dataset and grid |
| 4 | `make128.py` | nearest-upscales images to 128x128 for a model whose input is fixed there; asserts no colour is invented and that row order matches the manifest | only for a 128-input model |
| 5 | `preflight.py` | 8 checks in the order the evaluator hits the fields | **before every evaluation** |
| 6 | `ceil.py` | the grid's own cost: quantise the pixel truth, expand it back, score it. No model | before choosing a grid |
| 7 | `pixbench.py` | converts a saved patch prediction to pixels and scores it against the pixel labels, printing the ceiling beside it | after each evaluation |

`evaluation.py` holds the three metrics and `make_targets_grid.py` the patch conversion.
**Use these two unchanged.** They are what makes the two sides' numbers comparable; a
re-implementation is not the same evaluation function, and the data contract requires both
models to go through one.

## Two things that cost us real time

**`preflight.py` exists because four consecutive evaluations died** on a missing
`--expect-contract`, then a missing `names`, then `patch_size`, then the manifest's
`splits` — each one a ten-hour training run reaching the evaluator and failing on a field.
It exits non-zero. Run it first; it takes seconds.

**`build_mod_targets.py` asserts spatial coherence** because Multi-dSprites masks are
stored `[H,W,E]` while Tetrominoes' are `[E,H,W]`. Reading one as the other raised **no
error** — it produced a period-5 stripe and a target purity of 0.2813 against a chance
level of 0.2, and the run looked fine until the metrics came back meaningless. The assert
catches it. If it fires, your axis order is wrong, not your data.
