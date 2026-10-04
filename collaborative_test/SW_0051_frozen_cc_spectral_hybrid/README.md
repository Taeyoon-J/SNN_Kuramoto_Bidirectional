# SW_0051: frozen spike-component spectral hybrids

This tests whether spectral grouping can improve the aligned BIM6 readout while retaining its foreground locations. Every prediction starts from the frozen spike connected-component (CC) foreground mask at synchrony threshold 0.35, minimum group size 2, and largest-component background convention. Ground-truth masks are used only by the scorer after prediction.

The seed-0 seed checkpoint is `SW_0042_HDF5_aligned_BIM6_s0/core.pt`, evaluated on the fixed HDF5-aligned validation IDs 1320-1639. The aligned gamma manifest and core configuration are passed explicitly. Short and long inference windows are distinct runs (T256/settle64 and T1024/settle512).

The evaluator compares these GT-free readouts, separately for membrane and gated-spike histories:

* **freeze**: spectral-cluster the full signal-affinity-times-Gaussian matrix (`sigma=1.5`, `k=10`), then set every patch outside the spike-CC foreground mask to background.
* **restrict_k10**: spectral-cluster only the frozen foreground submatrix, with `k=10` capped by foreground node count.
* **restrict_dynamic**: same restricted clustering, with `k=max(2, number of spike-CC foreground components)`, capped by foreground node count.

The frozen mask itself is emitted as `spike_cc_baseline`. Results include all three mask metrics, predicted group count, foreground fraction, and an explicit boolean checking that each hybrid's non-background mask matches the frozen CC mask. The spectral partition may change object boundaries, but cannot move foreground/background locations. No threshold or method is selected using ground truth by the evaluator.

Run seed 0 short first after SW0044 has completed:

```bash
collaborative_test/SW_0051_frozen_cc_spectral_hybrid/evaluate.sh 0 0 short
```

The sequential launcher waits for the requested seed's SW0042 short/long JSONs, its `SPATIAL_EVALUATED` marker, and an idle assigned GPU, then evaluates both windows without overwriting outputs:

```bash
collaborative_test/SW_0051_frozen_cc_spectral_hybrid/launch_after_spatial.sh 0 0
```

The short seed-0 run is the initial implementation target; seeds 1/2 can use the same launcher after their aligned and spatial evaluations finish. The stage diagnostic motivating this readout found distance-controlled macro AUC phase .8567, membrane .8410, gated spike .8355; sigma1.5/k10 spatial membrane readout .723731/.211605/.370475 and spike .705382/.210102/.364788, versus SW0042 long spike-CC .598174/.605847/.461779 at threshold .35. Those results motivate a constrained hybrid, but do not establish that it improves any score.

Before the full evaluation, `preflight.sh GPU_ID SEED` performs a four-image
T256/64 forward/scoring run from that seed's actual checkpoint and aligned
gamma. It requires all seven outputs to contain finite metrics and all six
hybrids plus the CC baseline to preserve exactly the same foreground mask. A
versioned marker is bound to checkpoint, gamma, manifest, and evaluator-code
hashes; stale code/assets require a fresh preflight. The evaluator and
preflight refuse to overlap an occupied GPU. Temporary preflight JSON/logs are
removed after validation; only the marker remains. The full JSON/log are
separate and have overwrite guards.

Seed 0 completed both windows. Long T1024/settle512 results are:

| Readout | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| spike-CC baseline | .598427 | .606040 | .461779 |
| membrane freeze | **.630592** | .606040 | .449873 |
| spike freeze | .620047 | .606040 | .448135 |
| membrane restricted dynamic | .588565 | .606040 | **.465839** |

All rows preserve the baseline foreground mask exactly. Full membrane
spectral grouping therefore raises FG-ARI by `.032164`, while the dynamic
restricted variant raises object IoU by `.004059`; neither dominates the CC
baseline on both grouping metrics. The full short and long JSONs are in
`results/seed0/`. Seeds 1/2 are not launched until the real-asset preflight
passes for their checkpoints and this seed-0 tradeoff justifies expansion.
