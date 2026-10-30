# sw0255/ — running the peer's benchmark package on our side

The peer packaged their current four-model benchmark as `benchmark_codes/` on `patch_final`
(`a98b95f`). This is what we run from it, how it was verified, and why our order differs.

## Verified before anything was launched

| check | result |
| --- | --- |
| their `verify_bundle.py` | **passed** — 77 vendored files byte-identical to their server export, 79 python files compile, five dataset splits intact, 3500 unique TRAIN each |
| SHA256 of every file against `FILES_SHA256.json` | **118/118**, nothing mismatched or missing |
| environment vs `contracts/environment.json` | **exact** — python 3.10.20, torch 2.13.0+cu130, CUDA 13.0, numpy 2.2.6, same TITAN RTX |
| `preflight --dataset tetrominoes` | 3500/1000/1000, **`splits_equal_actual_server: true`** |
| `train --model 240 --admission` | real CUDA forward/backward, state change, save/resume, one-image evaluation |

**One transfer fault of mine, worth recording.** macOS `tar` added 132 `._*` AppleDouble
files, which broke both their bundle lock (`protected benchmark files differ from frozen
f9301ed snapshot`) and `verify_bundle.py` (`source code string cannot contain null bytes`,
because `._build_mod_targets.py` matches the `*.py` glob and is binary). Deleting them made
both pass. Use `COPYFILE_DISABLE=1 tar` from macOS, or ship from git on the server side.

## Why our order is the reverse

The peer runs models **252 → 253 → 254 → 240** over **clevr → multi_dsprites →
tetrominoes**, then objects_room and clevrtex. We run the reverse on both axes so the two
of us meet in the middle instead of duplicating cells.

Within that, `ourqueue.sh` puts **253 first on both of our datasets, all three seeds**:

- **253 is the only unsupervised model of the four.** Their own scheduler asserts it:
  `report['mask_opened_during_training'] == (job['model'] != 253)`. It is the only candidate
  for a table beside CAE, CtCAE, Rotating Features, SynCx and Slot Attention. 240, 252 and
  254 all open masks during training and belong to the supervision ablation instead.
- **Nobody is running 253 outside CLEVR.** The peer has 253 on CLEVR seeds 0 and 1; its
  Tetrominoes and Multi-dSprites cells are empty, and those two datasets are ours.
- **253 is about half the cost.** It is grid 32 against the others' grid 64, so the pairwise
  term is a quarter the size. Measured on their jobs: **36.0 s per update for 253 against
  74.6–76.3 s for 252**; ours measured 52.9 s on a 64-grid model with the GPU to itself.

## Measured throughput, from their live job tree

Read directly from `/work/USERS/kevinswk/experiments/SW0255_full3500/jobs/*/*/*/progress.json`
and the `initial.pt` mtimes, rather than estimated from a progress percentage:

| model | dataset | seed | updates | elapsed | s/update |
| --- | --- | ---: | ---: | ---: | ---: |
| 252 | clevr | 0 | 1019 | 21.4 h | 75.4 |
| 252 | clevr | 1 | 623 | 21.4 h | **123.4** |
| 252 | clevr | 2 | 1030 | 21.4 h | 74.6 |
| 252 | multi_dsprites | 0 | 1014 | 21.4 h | 75.8 |
| 252 | multi_dsprites | 2 | 1007 | 21.4 h | 76.3 |
| 252 | tetrominoes | 0 | 1030 | 21.4 h | 74.6 |
| **253** | clevr | 0 | **2089** | 20.9 h | **36.0** |
| **253** | clevr | 1 | 1542 | 14.9 h | 34.9 |

The 123.4 s row is contention — they run eight jobs across four GPUs, so some share a card.

**A seed is 2,190 updates**, so a grid-64 condition is about **33 hours** and a grid-32 one
about **20**. The full specification is 4 models × 5 datasets × 3 seeds = **60 runs**, which
at these rates is roughly **1,980 GPU-hours**. That is not reachable on two cards, which is
why the queue is ordered by value rather than by completeness.

**The three-seed requirement is the specification, not a choice.** `README.md` states
`seed 0/1/2` for all four models on all five datasets, and `run.py` restricts `--seed` to
`(0,1,2)`. The peer's CLEVR-heavy state is progress, not a reduced plan.

## The queue

`ourqueue.sh <gpu>` — one worker per GPU, **GPUs 2 and 3 only**, since 0 and 1 belong to
the collaborator. Each worker claims a job with an atomic `noclobber` write, skips anything
already carrying `complete.json`, runs it, then claims the next. A failed job keeps its
claim so it is not retried blindly.

We do not use the package's own `queue` subcommand: it requires preflight on all five
datasets, including objects_room and clevrtex which we have not prepared, and it takes its
order from the peer's priority file.

```
253 tet s0 · 253 md s0 · 253 tet s1 · 253 md s1 · 253 tet s2 · 253 md s2
240 tet s0 · 254 tet s0 · 240 md s0 · 254 md s0 · 240 tet s1 · 254 tet s1
```

Logs: `/work/USERS/tkim1/queue_gpu{2,3}.out` for the workers,
`/work/USERS/tkim1/sw0255/queue/<model>_<dataset>_<seed>.log` for each job.
Output tree: `/work/USERS/tkim1/sw0255/run` — never the peer's live `SW0255_full3500`.
