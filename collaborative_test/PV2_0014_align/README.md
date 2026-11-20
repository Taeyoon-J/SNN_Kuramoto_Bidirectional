# PV2_0014 — align spike synchrony to the phase contrast (failed, informatively)

**failed.** Parent `PV2_0013`. Seed 0, validation 300.

## Change

`--spike-align-weight`: standardise each row of the spike synchrony matrix and of
the phase PLV matrix and maximise their correlation, with theta detached. Where
`PV2_0007` matched magnitudes by MSE and fused the graph, this matches the
ordering within each row -- the part the connected-component rule reads -- so a
constant offset or rescale of spike synchrony is free.

## Result

| seed 0 | spike fg_ari | phase fg_ari | gap |
| --- | --- | --- | --- |
| BIM6, no alignment | 0.625 | **0.678** | 0.053 |
| align 2 | 0.625 | 0.636 | 0.011 |
| align 10 | 0.568 | 0.590 | 0.021 |
| align 40 | 0.487 | 0.491 | **0.004** |

The term optimised cleanly: 0.354, 0.146, 0.111 as the weight rose. **The gap did
close -- and by the phase readout falling to meet the spikes.** At weight 40 the
two readouts are 0.004 apart and both have collapsed to 0.49.

## Why, and what it means for the whole approach

Detaching the teacher is not enough. The student's gradient still reaches
`graph_generator`, `gamma_channel_proj` and `kuramoto`, which are what generate
theta, so theta is reshaped into something the spikes can already match instead
of the spikes learning to match theta.

`PV2_0007` and this experiment failed the same way. The magnitude-versus-contrast
diagnosis was a second-order detail; **the shared upstream parameters were the
actual cause**, and any loss that asks the spikes to imitate the phases while both
sides remain trainable will take this route.

Hence `PV2_0015`, which freezes the phase path so the teacher is genuinely fixed.
