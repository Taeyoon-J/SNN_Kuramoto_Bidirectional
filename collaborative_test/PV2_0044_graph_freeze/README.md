# PV2_0044 — freeze a trained graph, retrain the rest: best on all three

**completed, new best on every metric.** Parent `PV2_0027`. Ported from
`origin/patch_v2_sw` SW0064/SW0065/SW0072. Validation 300, window 1024, kernel
sigma 1.0, sync 0.05.

## Why this axis was still open

Graph work here was closed on **quality**: a ground-truth coupling graph was worth
only +0.06 (`PV2_0005`). The peer's axis is graph **stability across seeds**, which
was never varied. Their evidence:

- **SW0064**: identical gamma diverges most at the graph-to-Kuramoto conversion,
  late PLV object margin 0.8167 for seed 0 against 0.5167 for seed 2, with
  dendrite-to-membrane propagation nearly lossless.
- **SW0065**: swapping seed 0's trained graph generator into seed 2 moved their
  32-image result from `.3895/.2003/.1822` to `.7036/.3081/.5376`, where swapping
  the drive or the Kuramoto parameters did not.
- **SW0072**: freeze the trained graph, retrain the other seeds downstream, 3-seed
  mean `.781407/.478038/.616746`.

Our seed problem was the same size: seed 0 chronically weak and two of six seeds
unusable (`PV2_0012`).

## One adaptation, not a blind port

They froze **seed 0's** graph because seed 0 was their strongest at 0.8171. Here
seed 0 is the **weakest** at 0.6639 and seed 1 the strongest at 0.7239, so **seed
1's** graph was frozen and seeds 0 and 2 retrained. Freezing seed 0's would have
pinned the worst graph.

## Result

| metric | seed 0 | seed 1 | seed 2 | **mean** | std | range | previous | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.7235 | 0.7239 | 0.7275 | **0.7250** | 0.0022 | 0.0040 | 0.7059 | **+0.0191** |
| `patch_foreground_iou` | 0.7164 | 0.7251 | 0.7148 | **0.7188** | 0.0055 | 0.0103 | 0.6778 | **+0.0410** |
| `patch_matched_object_iou` | 0.4935 | 0.4951 | 0.4974 | **0.4953** | 0.0020 | 0.0039 | 0.4903 | +0.0050 |

The weak seed carries it: seed 0 goes 0.6639 to 0.7235, **+0.060**, matching the
peer's pattern of rescuing weak seeds. Seed 1 is the unchanged graph source.

**Freeze verified**: both checkpoints have all **7 graph tensors bitwise equal** to
the source. That check exists because two freeze-style flags were silently
inert earlier in this work -- `--encoder-lr` produced byte-identical checkpoints,
and the first `--freeze-phase-path` froze a randomly initialised path.

## Two things that must travel with this number

**It is not three independent runs.** The seeds share a graph by construction, so
the variance the graph contributes is removed rather than reduced. The range of
0.0040 is a consequence of that sharing, not evidence of a stabilised model. Any
report of 0.7250 has to say so.

**It reached the measured ceiling rather than raising it.** `PV2_0032` put this
architecture's ceiling at fg_ari **0.7338**, using the true object count *and* the
phase readout, which is a diagnostic and cannot be a score. At 0.7250 the graph
freeze has brought the spike readout to within 0.009 of that, so it closed the
remaining gap to the ceiling instead of moving the ceiling.

Which makes the 0.75 target firmer, not closer: 0.75 is above the ceiling, so no
graph, readout, threshold or objective arrangement reaches it. `PV2_0034` measured
where the headroom actually is -- 0.9754 with features that separate objects, so
0.27 of it is in the features -- and `PV2_0037`/`PV2_0038`/`PV2_0039`/`PV2_0042`
found that DINO, label-free clustering, self-bootstrapping, encoder capacity and
every slot-term setting fail to deliver it, while end-to-end training with
reconstruction at weight 1.0 is the one route that works at all.
