# Peer review before native32 validation

Fetched both research branches on 2026-10-10. `origin/patch_v2_sw` remains at approved publication c3e2434. The colleague's `origin/patch_v2` latest collaborative evidence is commit 697eeb67dc031325b63153e678cea8cbc602da31 (2026-10-08).

PV2_0049 reports validation means .7250/.7188/.4953 for their model versus .6635/.1184/.0869 for matched Slot Attention. Their comparison has 300 validation images, shared frozen graph across model seeds and a 320-epoch Slot baseline versus the official 457-epoch schedule. It does not replace our 320-image native32 baseline or establish our victory.

PV2_0051 measured feature geometry and cancelled a proposed full-convergence scaling run. It supports examining encoder quality, but it does not establish that adequately converged larger-data training cannot help. Keep the requested fresh-data scaling study open.

PV2_0052's longer integration window improvement was only .0005 under their graph-frozen recipe. Preserve the current T1024 protocol rather than spending training budget on a speculative longer window.

PV2_0053 diagnoses same-appearance object confusion in graph features. Adding coordinates reduced their oracle-defined unseparable-image fraction from 38.8% to 18.1% at weight1. This is a mask-based feature diagnostic, not trained segmentation evidence. Their next coordinate-augmented training PV2_0054 has no committed result in this snapshot. Our current native32 graph already retains a physical spatial prior and full geodesic operator; neither is equivalent to coordinates inside graph feature projection. Position-aware learned graph features remain a separately registered candidate if the present actual-spike joint pilot fails. Do not inject coordinates into the spike-only binder or silently change the frozen SW0135 protocol.

Immediate incorporation: retain jointly trainable encoder/graph, fixed integration window, strict matched native32 baseline and explicit scaling controls. Re-check peer results before the next model hypothesis; no unsupported port of their scores or oracle geometry.
