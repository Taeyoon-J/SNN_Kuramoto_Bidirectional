# Peer review: fdcc94f, 2026-10-12

Read the actual `patch_v2` HEAD `fdcc94f14cceb19d2e2b99829281ae9bbf8644a4` through Git, after comparing with the previously reviewed `9d7dedd200fe02bb1a175b55c47fc098e8b511ce`. New commits are `6701182` and `fdcc94f`. Read both README and metrics.csv in PV2_0079_dynamics and PV2_0080_lowdim; no working code was merged or changed by fetch.

The peer runs are seed0, 40 epochs, validation300, contract v1. Their absolute numbers are not our native32 fixed320 three-seed benchmark.

| Peer setting | FG-ARI | Foreground IoU | Object IoU | Final loss |
|---|---:|---:|---:|---:|
| Reference osc_dim4, coupling k256 | .7239 | .7251 | .4951 | 17.8649 |
| osc_dim8 | .5670 | .5719 | .4338 | 14.1895 |
| osc_dim3 | .6783 | .6892 | .4688 | 16.2142 |
| osc_dim16 | .0033 | 0 | 0 | 86.0660 |
| osc_dim2 | .0033 | 0 | 0 | 86.1911 |
| coupling k64 | .6088 | .7122 | .4398 | 17.1671 |
| coupling k128 | .4678 | .6197 | .3761 | 17.0405 |
| coupling k512 | .3694 | .1696 | .2334 | 16.7137 |
| coupling k1024 | .3331 | .2384 | .2023 | 80.7820 |

Here k is coupling, not the number of image patches. The dynamics changes were varied separately from osc_dim. Increasing and decreasing phase capacity both lowered the objective while worsening FG, so a capacity-only explanation is unsupported. These experiments are evidence against repeating oscillator-dimension/coupling sweeps merely to reduce the existing synchrony-statistics objective; they are not a universal optimum or a proof about our separate contract.

Incorporation: preserve the existing dynamics in SW0143, whose R/C/T operator comparison is already running with three seeds. SW0142 is closed after its registered assignment gate failed; SW0144 completed a GT-free TRAIN diagnostic and found both retained local boundary information in actual Q and disconnected regions in warm P. Do not equate reconstruction loss, gradient existence or boundary correlation with segmentation improvement. A future loss change must supply an image-specific grouping constraint and demonstrate a fixed-contract FG gain, rather than another generic PLV/density statistic or blind capacity sweep.

The peer concludes that static images lack enough information for its target. The reported interventions do not prove that global impossibility claim; neither these one-seed runs nor our masked-cache diagnostic establishes an upper bound for other training objectives or architectures. No supervision, motion, multi-view data, changed dataset, oracle count, or evaluation relaxation is introduced here.

Sources: [PV2_0079 README](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/fdcc94f14cceb19d2e2b99829281ae9bbf8644a4/collaborative_test/PV2_0079_dynamics/README.md), [PV2_0079 CSV](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/fdcc94f14cceb19d2e2b99829281ae9bbf8644a4/collaborative_test/PV2_0079_dynamics/metrics.csv), [PV2_0080 README](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/fdcc94f14cceb19d2e2b99829281ae9bbf8644a4/collaborative_test/PV2_0080_lowdim/README.md), [PV2_0080 CSV](https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/blob/fdcc94f14cceb19d2e2b99829281ae9bbf8644a4/collaborative_test/PV2_0080_lowdim/metrics.csv).
