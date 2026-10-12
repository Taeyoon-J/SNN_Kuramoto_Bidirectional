# What blocks 0.9, and the 27 routes that are closed

The goal's third part asks for a way to reach `patch_fg_ari` 0.9 while keeping SNN +
Kuramoto bidirectional. **It is not met.** Best on test is **0.7603** (`PV2_0076`, three
seeds, 1000 images, contract v1, 16x16 grid), and 27 routes are closed.

This document exists so the search space is explicit. **Check it before running
anything**: one route here was almost re-run after 27 others, because its record said
`PV2_0015` had "named the fix" and I misremembered that the fix was never executed — it
was, at `PV2_0016-18`, and it failed.

## The five structural findings

Any future attempt has to be consistent with these. They are what the 27 routes bought.

### 1. The oracle numbers measure transcription, not discovery (`PV2_0072`)

`make_oracle_gamma.py` builds its tensor as `codes[true_labels]`, so the oracle gamma **is
the answer**, written as one code per object. `PV2_0034`'s 0.9754 and `PV2_0066`'s 0.9671
therefore say the readout transcribes a grouping faithfully — not that these dynamics can
find one. **An earlier claim that this "established feasibility" is withdrawn.**

### 2. The pipeline repairs incoherent damage; its own errors are coherent (`PV2_0072`, `PV2_0073`)

Feeding the graph a grouping corrupted at random, output beats input everywhere below a
perfect input: 0.0416 -> 0.1933, 0.4205 -> 0.5087, 0.7525 -> 0.7766, crossing over near
0.85-0.88. But iterating the model's **own** grouping converges at **0.7274** in one step.
Random corruption is spatially incoherent — a wrong patch among right ones — which the
geodesic relaxation and distance prior exist to repair. The model's own errors are whole
objects merged (`PV2_0053`: 38.8% of images hold such a pair), and a merged pair is
internally consistent, so nothing local is wrong to fix.

**The amplification is real and acts on the wrong error type.**

### 3. The geodesic barrier is two thin walls with a free corridor (`PV2_0075`)

Adjacent-pair dissimilarity, which is all the step cost sees:

| pair type | mean `1 - cosine` |
| --- | --- |
| within one object | 0.0331 |
| **background to background** | **0.0054** |
| object to background | 0.1743 |
| between two objects | 0.1587 |

The wall is real at 5.27x a shading step, but the corridor is **six times cheaper than a
step inside an object**. At the learned contrast, reaching an object six patches away
costs about 6.8 against 6.4 for crossing a six-patch-wide object — a tie, and that tie is
the merge. The contrast cannot break it because it scales walls and shading steps alike
(`PV2_0074`), the superlinear fix helps only at inference (`PV2_0076`, `PV2_0077`), and
lowering the per-step toll does not help (`PV2_0078`).

### 4. The objective and the metric are decoupled (`PV2_0080`, ten instances)

Ten separate changes reached a **lower loss** than the 17.86 operating point with an
**equal or worse** score, on both sides of every axis:

| change | loss | fg_ari |
| --- | --- | --- |
| osc_dim 3 (`PV2_0080`) | 16.21 | 0.6783 |
| **operating point** | **17.86** | **0.7239** |
| osc_dim 8 (`PV2_0079`) | 14.19 | 0.5670 |
| position in the graph (`PV2_0055`) | 13.91 | 0.5556 |
| chroma11 (`PV2_0058`) | 17.71 | 0.3599 |
| trained under the superlinear cost (`PV2_0077`) | 16.71 | 0.4268 |
| all four freeze arms (`PV2_0081`) | 15.70-16.63 | 0.63-0.72 |

**Lowering this loss is not evidence of progress**, and any search that optimises it is as
likely to hurt as to help. The PLV terms — bimodality, balance, target density — are
statistics of the synchrony matrix, and a matrix with the right statistics and the wrong
groups satisfies them.

### 5. There are 1,325 parameters that matter (`PV2_0081`)

Of 200,114 state-dict entries only **69,042 are trainable** (`sc` and `grid_distance` are
buffers at 65,536 each), and the mass is lopsided: `kuramoto` 67,585 (97.9%),
`dendric_layer` 1,036, `membrane_layer` 256, `graph_generator` **132**.

Freezing `kuramoto` and `graph_generator` **together** leaves 1,325 parameters training —
1.9% — and scores **0.7249** against the full-training 0.7239. Freezing `kuramoto` alone
drops to 0.6858, because the graph's 132 parameters then drift from the dynamics they were
fitted to: **the graph and the Kuramoto state move together or not at all.**

So there is very little here for an optimiser to improve, which is why every capacity
(`PV2_0079`, `PV2_0080`) and constraint (`PV2_0081`) experiment failed.

## The 27 closed routes

**Features (11).** DINOv2 and label-free clustering (`PV2_0037`); self-bootstrap, +0.006
and circular (`PV2_0038`); encoder capacity, depth 2 diverges and depth 3 gives 0.3910
(`PV2_0041`); slot-term defaults already optimal (`PV2_0042`); end-to-end alone 0.6779,
below frozen features (`PV2_0039-40`); the frozen core (`PV2_0045`); data scale, by score
(`PV2_0048`) and by mechanism (`PV2_0051`); position in the graph's code (`PV2_0053-55`);
chromaticity (`PV2_0057-58`); inference-time smoothing (`PV2_0062-63`); the matched Slot
encoder's own features (`PV2_0065`).

**Graph (8).** Transforms confined to the graph (`PV2_0067`); reconstruction training the
graph with features frozen (`PV2_0068`); the graph fed its own groups (`PV2_0069`,
`PV2_0073`); the generator's forward-pass knobs (`PV2_0071`); per-component freezes
(`PV2_0081`, four arms).

**Geodesic cost (3).** The learned contrast (`PV2_0074`); the superlinear power, which is
the **one** route that transferred, +0.0070 on test and only outside the objective
(`PV2_0076`, `PV2_0077`); the per-step toll (`PV2_0078`).

**Readout (2).** The integration window, saturated (`PV2_0052`); the fg_ari/IoU dial, where
spectral k trades one metric for another (`PV2_0059-60`).

**Objective (1).** Penalising agreement with the geometric template — pointless, since our
partition overlaps it by only ARI 0.1164 (`PV2_0056`); cross-view consistency, where the
invariance was achieved and the score fell 0.0482 (`PV2_0070`).

**Dynamics (8).** `osc_dim` 2, 3, 8, 16 and coupling 64, 128, 512, 1024 — 4 and 256 are a
two-sided optimum (`PV2_0079`, `PV2_0080`).

**Already closed before this cycle:** per-region transduction (`PV2_0016-18`) — "the shared
map is the prior, not the bottleneck".

## The metric is not the obstacle (`PV2_0082`)

It is tempting to suspect the measurement, since the field's standard method scores .6663
here and the peer's 32x32 contract puts comparable models near .78. **It does not hold.**
Labelling patches from the TRUE pixel masks under rules other than the contract's own:

| rule | fg_ari |
| --- | --- |
| majority, ties to low ID — contract v1 | 1.0000 (check) |
| majority, ties to **high** ID | 0.9996 |
| an object must hold **half** the patch | 0.9756 |

17.3% of foreground patches straddle two or more objects, and a perfect segmenter still
scores .9756 under a far stronger rule change than a tie-break. **Contract v1 admits 0.9
comfortably**, so the 0.14 gap is model error and moving to a finer grid would change the
number without addressing the cause. Do not spend effort on the contract.

## What a credible attempt would have to do

- **Not** optimise this loss (finding 4).
- **Not** rely on a separation statistic. Four transforms improved one — Euclidean ratio,
  cosine margin, path-cost ratio — without the score following (`PV2_0058`, `PV2_0062`,
  `PV2_0064`, `PV2_0078`). They are computed on ground-truth object pairs, while the
  readout acts on whatever survives the rollout, the gate, the membrane threshold and the
  component extraction.
- **Judge gains in the calibrated regime only.** Predicted foreground is 0.129 against a
  target of 0.118. fg_ari is computed over true-foreground patches, so widening predicted
  foreground raises it for free: chroma3 reached 0.7380 at 0.509, spectral k=20 0.7718 at
  0.808, whitening 0.7670 at 0.590. Slot Attention's own 0.6663 comes with 0.721.
- **Supply instance identity**, not feature geometry (findings 1-3). Separating two
  adjacent similarly-coloured objects is what is missing, and in a single static render the
  only cue is their separation, which the pipeline already uses.

The label-free sources of object-level information are all tested: reconstruction with
slot competition (`PV2_0039`, `PV2_0068`), cross-view consistency (`PV2_0070`) and
self-reference (`PV2_0069`, `PV2_0073`). What remains is **data this dataset does not
contain** — motion, or several views of one scene — or supervision the contract forbids.

## Protocol notes earned the hard way

Two devices caught wrong conclusions repeatedly this cycle and belong in every script:

1. **A regression row first.** Re-score an existing checkpoint and refuse to read anything
   below it unless the recorded value reproduces. It caught a patch that replaced the
   learned graph with the static one at `graph_extra_channels = 0`, scoring 0.0033 instead
   of 0.7239 — a bug that fired only in the configuration every existing result uses, while
   the new path's smoke test passed.
2. **A calibration gate on fg_ari gains.** Three automated verdicts in this cycle were wrong
   because they read fg_ari alone: "TRANSFERS" at predicted foreground 0.85, "THE LEVER
   WORKS" counting an oracle row, and "THE TOLL MATTERS" at 0.65.
