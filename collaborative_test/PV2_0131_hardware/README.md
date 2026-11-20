# PV2_0131 — what actually limits a 253 grid-64 update, measured on three GPUs

Status: **completed** (the measurement). The run it was made for, `253 @ 64x64 CLEVR seed 0
at 10,000 x 10`, is **running** on the RTX 4090 and is recorded separately.

This is not a model experiment. It is the record of a hardware decision I got wrong, the
measurement that corrected it, and the number anyone repeating this work needs.

## The claim I made, and what it cost

I argued the Kuramoto coupling is **bandwidth-bound** — `bmm([B,N,N],[B,N,D])` runs at
2.0 flop/byte, far below every ridge point on this hardware — and concluded that memory
bandwidth picks the GPU. On that basis I priced an H200 NVL at 3875 GB/s, estimated
**11.8 h and $43**, and we rented one.

It ran at **14.286 s/update: 24.8 h and $90.** The estimate was off by 2.10x.

The premise was measured on a **single large bmm in isolation** on a TITAN RTX. It does not
describe the training loop, which walks 1,024 rollout steps sequentially at microbatch 1.

## The three measurements

| | s/update | 6,250 updates | $/hr | one seed |
| --- | ---: | ---: | ---: | ---: |
| TITAN RTX, frontier, **four processes on the card** | 67.41 | 117.0 h | — | — |
| H200 NVL, sole tenant | 14.286 | 24.8 h | $3.628 | **$89.98** |
| **RTX 4090, sole tenant** | **10.714** | **18.6 h** | **$0.434** | **$8.07** |

The 4090 is **1.33x faster than the H200 and 11.2x cheaper**: 14.9x on value.

The frontier figure is real, from `training.jsonl` — `(612.99 - 73.75) / 8` over updates 2-9,
discarding update 1's warmup. It is **not a clean hardware number**: GPU 3 carried the peer's
240 and 253 and our own 32x32 multi_dsprites alongside it. Separating contention from
hardware would mean killing the peer's multi-day jobs, so that separation was not made and
this row must not be quoted as "TITAN RTX is 4.7x slower than an H200".

## Why the 4090 wins, from the utilisation counters

| | GPU util | **memory util** | **sustained SM clock** | power | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| H200 NVL | 45% | **6%** | **1785 MHz** | 176 W of ~600 | 7731 MiB of 143771 |
| RTX 4090 | 100% | 75% | **2790 MHz** | 239 W of ~450 | 7564 MiB of 24564 |

Sustained clock ratio **2790 / 1785 = 1.563x**; observed speedup **1.333x**. The clock
very nearly accounts for the whole difference.

**The H200 sold us what the job does not use and charged us the thing it does.** Its memory
system idled at 6% while its clock ran 36% below the 4090's. On the H200 the kernels are
small enough relative to the machine that it spends 55% of the time with nothing resident;
the 4090 is never idle.

Note the two memory-util numbers are not a contradiction. `utilization.memory` is the
fraction of *time* the memory system is busy, not the fraction of peak bandwidth: the same
bytes at 3875 GB/s occupy the controllers for a fraction of the time they do at ~880 GB/s.

## What this means for buying compute for this model

1. **Sustained SM clock, not bandwidth and not TFLOPS.** The 4090's 5.07x FP32 advantage
   over a TITAN RTX (82.6 vs 16.3 TFLOPS) is mostly irrelevant here; its clock is not.
2. **Datacenter parts are the wrong aisle.** An A100 boosts to 1410 MHz, *below* the H200,
   and would likely be slower than either card for this job at several times a 4090's price.
3. **DLPerf does not describe this workload.** The 4090 lists 97.6 against the H200's 452.6
   and is faster in fact. DLPerf weights tensor-core throughput we do not use.
4. **VRAM is not a constraint.** 7.6 GB of a 24 GB card.

## Honest limits of this record

- One 300 s window per machine, not a distribution. The per-update variance on frontier was
  65.1-70.9 s, about +-4%, so a single window is adequate for a 10x decision and would not
  be for a 10% one.
- The three machines differ in CPU as well as GPU (EPYC 9454 at 3.8 GHz against a Ryzen 9
  7900X at 5.73 GHz). Since 55% of the H200's wall clock was *not* GPU-resident, part of the
  4090's win is its CPU. The two were not separated, and for the purpose — which card to
  rent — they do not need to be, because you rent the pair.
- **No claim is made here about any metric.** Nothing in this record touches FG-ARI,
  foreground IoU or matched-object IoU.

## The lever this leaves on the table

55% of the H200's time and a smaller share of the 4090's goes to launching 1,024 steps of
small kernels rather than running them. **CUDA graphs** would capture that. It is the only
remaining way to cut the time rather than the bill — and it edits `kuramoto_layer.py`, which
is inside `scientific_sources()`, so it would need what the fused-bmm change got: not a
microbenchmark, but the gradient after a full 1,024-step rollout compared against the
run-to-run control of 2.09e-08. **That control is the point.** The fused change passed it
(7.9e-08) and was kept; microbatch 4 (4.06e-02) and fp16 coupling (4.79e-02) failed it and
were dropped. This test deserves no less, precisely because it is the microbenchmark-versus-
full-loop distinction that produced the error at the top of this page.
