# PV2_0120 — route C is blocked by hardware, not refuted

Status: **blocked**. Grid 64 does not fit on a 24 GB TITAN RTX at **any** batch size with
this configuration. No score was produced and none is claimed.

## What was built first, and holds

Both inputs exist and are verified:

| | shape | statistic |
| --- | --- | --- |
| `wm10k_full_grid64.pt` | (10000, 8, 4096) | mean +0.0000, std 1.0000 |
| `targets_v3_grid64.pt` | (8000, 64, 64) | foreground 0.1200, **purity 0.9890**, ties 0.18% |

Patch purity rises with the grid exactly as route C's argument needs — **16: 0.9572,
32: 0.9779, 64: 0.9890** — and `make_targets_grid.py` verified its grid-16 output against
`targets_v1.pt` before writing. `PV2_0114` tied 23.1% of the missed patches to
narrow-majority boundary cells, and that population shrinks as the grid refines. The
groundwork is sound; only the training is blocked.

## The failure, and the probe that missed it

Training died on `torch.OutOfMemoryError` inside `signal_synchrony`, at
`torch.bmm(trace, trace.transpose(1, 2))` — the **N × N spike synchrony matrix** that
`--spike-plv-weight 5.0` asks for, built once per oscillator component because
`spike_per_component` is on.

**My first probe measured a forward pass only** and cleared at 2.08 GB, so I reported that
memory was not the constraint. That was wrong: the forward pass is not what allocates the
large tensor. The loss is.

Re-probed with forward, loss and backward together:

| batch | result |
| ---: | --- |
| 8 | OOM |
| 2 | OOM, peak 22.74 GB |
| **1** | **OOM, peak 23.16 GB of 23.47** |

**Reducing the batch does not rescue it.** At batch 1 the peak is within 0.3 GB of the
card, so there is no headroom to find.

## What this is and is not

Route C is **blocked by hardware**, which is not the same as closed by measurement. The
nine closed routes were each refuted by a number; this one has no number. It must not be
written up as evidence that a finer grid does not help.

Two ways forward, neither free:

1. **A bigger card.** The earlier rental estimate flagged grid 64 as the one case that might
   need 80 GB; that is now measured rather than guessed. At roughly 32 GPU-hours per seed
   (`PV2_0087` prices a 4× grid at 18.6× of grid 32's 103 minutes), an A100 80 GB at
   $1.19–1.43/hr is about **$38–46 for one seed** and **$114–137 for three**.
2. **Drop `--spike-plv-weight` at grid 64.** It removes the allocation, but it removes the
   entire spike-side objective with it, and every configuration since `PV2_0094` was tuned
   with it on. That is a different experiment, not a workaround, and it would not answer
   whether the grid helps.

## Where the goal stands

| route | state |
| --- | --- |
| nine readout and training routes | **closed by measurement** (`PV2_0112`–`PV2_0119`) |
| C grid 64 | **blocked — needs more than 24 GB** |

So: **no route to 0.9 has been demonstrated on this contract.** Nine were closed by
measurement and the tenth cannot be run on this hardware. The honest statement for a
write-up is that the readout is exhausted and the one untested direction is a resolution
increase whose memory cost exceeds the available card.
