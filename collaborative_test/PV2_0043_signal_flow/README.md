# PV2_0043 — verifying the score really is spike pattern classification

**completed. Protocol check, not an experiment.** `CLAUDE.md` requires the final
score to come from a classifier on spikes or membrane, so this verifies what the
scored path consumes, on `BIM6_s1`, window 1024, settled window 512 steps.

## The path

    core.last_component_spikes -> per-component correlation -> product over the
    four components -> times a spatial Gaussian -> connected components -> masks

## 1. The tensor is a gated spike, not a pure binary train

`MembraneLayer` computes `act_fun_adp(membrane - threshold) * g_wave_t`, a binary
threshold crossing times a continuous gamma gate, so the values are continuous.
The question is whether the structure comes from the crossings or from the gate.

| | value |
| --- | --- |
| binary crossing rate, settled window | 0.7847 |
| fraction of units always on | **0.0050** |
| fraction never crossing | 0.0000 |
| per-unit std of the binary part | 0.4041 |
| **correlation between the binary-only affinity and the scored affinity** | **0.9822** |

**Spike timing carries the structure.** Rebuilding the affinity from the binary
part alone reproduces the scored affinity at r = 0.982; the gate scales amplitude
without creating the grouping.

## 2. The peer's saturation finding does not hold here

Their SW_0034 reported the binary threshold "constant 1 after settle" with
"gated spike variation gate-driven", which would mean the classifier reads the
gate. On this branch only **0.5%** of units are always on and the per-unit std of
the binary part is 0.40. The membrane-threshold work in `PV2_0023` and the
1024-step readout window from `PV2_0019` act on exactly this, so the same
architecture saturates or does not depending on configuration.

## 3. The spatial kernel is geometry and is reported separately

| | fg_ari |
| --- | --- |
| spike affinity, no kernel | 0.6887 |
| spike affinity times the kernel | **0.7059** |
| spatial kernel alone, same readout (spectral) | 0.3902 |
| spike affinity, same readout (spectral) | 0.6068 |

The kernel is worth **+0.017**. Under a readout where both can be scored, geometry
alone reaches 0.390 and the spikes add **+0.217** on top. The kernel contributes
little but has to be positionally correct to contribute at all: permuting which
patch each row belongs to takes fg_ari to 0.0033 (`PV2_0029`).

## A measurement bug found and fixed here

The first version of this check reported the spike rate as NaN. `comp` is
`[B, D, N, T]` and the slice `[:, :, 512:]` cut the **region** axis rather than
time, giving an empty slice. That was the diagnostic, not the model.

## Conclusion

The score satisfies the protocol: masks from a classifier on spikes, with phase
and PLV readouts reported only as diagnostics throughout. The classifier reads
correlations between per-component spike trains, and those correlations are
driven by threshold crossings rather than by the gate.
