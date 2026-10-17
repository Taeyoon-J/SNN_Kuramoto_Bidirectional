# SW0190 — oracle-count global spike-affinity partition pilot

Preparation: TRAIN16, all three frozen SW0186 seeds, two global spectral
readouts versus unchanged production masks. This tests an oracle-count upper
bound before investing in a learned count/assignment model. No optimizer steps,
validation masks or leaderboard result. See [registered design](DESIGN_KO.md).

Raw TRAIN imagewise reports stay private; only code and aggregate findings are
published. SW0188 continues independently.

Registered: eight tests passed locally/server, actual CLI and all three exact frozen incumbent CPU model loads passed. Three supervisor PIDs1688076/1688085/1688094 were independently verified alive and queued. No GPU worker has started yet; current free-device opportunity closed before admission and foreign owners occupy the other devices. SW0188 continues on its existing worker. Peer patch_v2 was checked this cycle and remains cab6c676. Metadata: results_archive/cpu_v1 and start_v1.

Completed: both oracle-count spectral variants are worse in all three seeds on TRAIN16 and fail admission. Baseline FG .872016 versus .782037/.771040. These are TRAIN diagnostic values, not validation scores or an automatic count classifier. Close these recipes without a sweep. See [full findings](RESULT_KO.md); raw TRAIN reports remain private.
