# Peer update review: c6ea3f2 (2026-10-12)

Reviewed actual `origin/patch_v2` commit `c6ea3f2e2433b6759bae86cec532ebfaae816c6b`, including PV2_0081 README, PV2_0082/PV2_0083 READMEs and CSVs, and PROPOSAL_0.9.md. The prior reviewed head was fdcc94f. Peer numbers use contract v1, native240x320 images and a16x16 grid; they are not scores on our native128,32x32,fixed320 contract.

## Evidence and its scope

- PV2_0081: same-seed reference FG0.7239, graph-only frozen0.7235, Kuramoto+graph frozen0.7249. The latter trains1325/69042 parameters. These single-seed near-equalities do not establish that trained dynamics are unnecessary: freezing fitted parameters still uses their forward computation. The peer also corrects its earlier graph-freeze claim: the reported three-seed gain primarily rescued a weak seed, rather than improving the reference seed. Lower loss again accompanies equal/worse mask metrics.
- PV2_0082: GT-derived majority/tie/high-ID/half-patch rules score FG1.0000/0.9996/0.9756. This is an oracle labeling-convention diagnostic, not a usable segmenter or an image-inference upper bound. It refutes a low ceiling caused by the labeling convention on that peer contract.
- PV2_0083: one-object patches account for82.7% of foreground patches and70.7% of mapped-label errors; their agreement is0.7979 versus0.5999 on mixed-object patches. Thus errors are not confined to patch boundaries. These are GT-stratified agreement diagnostics, not a decomposition of FG-ARI or measurements on our32x32 model.
- The motion-data proposal is prospective. No new trained0.9 result is supplied. Finite failed interventions do not prove static-image object grouping impossible. Warped-Q consistency alone also admits a constant or coherently merged Q; independent motion needs an objective that actually penalizes those solutions before it can be considered a remedy.

## Incorporation into our current cycle

Keep SW0146 immutable. Its paired encoder/graph-only versus jointly plastic core arms directly test whether image-specific boundary learning benefits from coordinated changes, with fixed source-Q preservation. Evaluate actual FG-ARI and both IoUs; lower seam loss does not count as an improvement. The first64 TRAIN seam/Q/event strata are descriptive, and final prediction arrays are fixed before any GT scoring.

Do not copy the peer's foreground<0.20 admission threshold into our different contract. Do not use oracle patch rules as model performance, change our mask labels, abandon native32, or replace the requested70000-image dataset with motion data. A model can still make coherent merges on pure patches; local image-boundary targets are a testable partial hypothesis, not a promise to resolve every merge.

Future gate claims must distinguish frozen trained parameters from removed/bypassed computations. SW0145's actual gradient and parameter changes establish learning credit only; SW0146's scores and later matched ablations/scaling remain necessary.
