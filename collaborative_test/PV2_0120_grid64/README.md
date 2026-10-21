# PV2_0120 — route C, grid 64

Status: **running** (seed 0, batch 4). An earlier version of this record said route C was
**blocked by hardware**. That was wrong and is retracted below.

## Retraction: grid 64 is not blocked

The first launch used `--batch-size 8`, died on `torch.OutOfMemoryError`, and I recorded
route C as needing more than 24 GB. The peer runs native64 at **batch 4 with a peak of
5.01 GiB** (`SW_0192`), which is what prompted the recheck.

Measured properly — **one process per batch size**, forward + loss + backward:

| batch | peak |
| ---: | ---: |
| 1 | **4.59 GB** |
| 2 | **8.90 GB** |
| **4** | **17.52 GB** of 23.47 |
| 8 | ~34 GB — what the first launch asked for |

**Grid 64 was never blocked. Batch 8 was one size too large.**

### Two measurement errors, both mine

1. **The first probe measured a forward pass only** and cleared at 2.08 GB, so I reported
   that memory was not the constraint. The large tensor is allocated by the **loss** —
   `signal_synchrony`'s `torch.bmm(trace, trace.transpose(1, 2))`, an N × N matrix at
   N = 4096, built once per oscillator component because `spike_per_component` is on.
2. **The second probe ran batch 8, 4, 2 and 1 in one process.** After the batch-8 OOM the
   exception object held references to the tensors, so `empty_cache()` could not return the
   memory and every later size ran on top of it. That is how *"batch 1 OOMs at 23.16 GB"*
   was recorded when batch 1 actually needs 4.59 GB.

Both were avoidable by the same discipline: measure the thing you will actually run, in a
process that nothing has already broken.

### Which knob costs what, at batch 1

| configuration | peak |
| --- | ---: |
| full (chunk 8, per-component, spike-plv) | 4.59 GB |
| `--spike-plv-weight 0` | 3.05 GB |
| `spike_per_component` off | 3.04 GB |
| `--geodesic-chunk 1` | 4.60 GB |

The geodesic chunking is already doing its job — tightening it further changes nothing.
The spike-side objective costs about 1.5 GB at batch 1, which is real but not decisive.

## Inputs, built and verified

| | shape | statistic |
| --- | --- | --- |
| `wm10k_full_grid64.pt` | (10000, 8, 4096) | mean +0.0000, std 1.0000 |
| `targets_v3_grid64.pt` | (8000, 64, 64) | foreground 0.1200, purity **0.9890**, ties 0.18% |

Patch purity rises with the grid exactly as route C's argument needs — **16: 0.9572,
32: 0.9779, 64: 0.9890** — and `make_targets_grid.py` verified its grid-16 output against
`targets_v1.pt` before writing. `PV2_0114` tied 23.1% of the missed patches to
narrow-majority boundary cells, and that population shrinks as the grid refines.

## Why route C is worth the compute

`PV2_0119` measured what the readout can do: the dial trades FG-ARI against foreground IoU
along the same axis `PV2_0090` caught Slot Attention on, peaking at **0.7862 with 84% of the
image called foreground**. Nine routes are closed by measurement. Raising the grid is the
one untested way to move that axis rather than slide along it, and grid 16 → 32 bought
**+0.1017** of actual pixel-protocol score while lifting the ceiling 0.6973 → 0.8355.

## Gate and falsifier, registered before the run

- **Gate:** beat `PV2_0104`'s grid-32 seed 0 — 0.7699 / 0.7531 / 0.7017 at `pred_fg` 0.1512
  — with matched-object IoU up at minimum and predicted foreground at or below 0.20.
- One seed cannot promote anything; it decides whether seeds 1 and 2 are worth 64 more
  GPU-hours.
- **Falsifier:** if seed 0 does not clear grid 32 on at least matched-object IoU, route C
  closes and every route in `ROUTE_MAP_0.9_v3` is closed by measurement.

Hyperparameters carried from the record rather than guessed: `--membrane-threshold-mode
population` because `PV2_0094` measured that the absolute threshold 0.06 kills the spike
path at 1024 oscillators, and the argument is stronger at 4096.
