# SW0073 - reconstruction-held joint features on the frozen trained graph

SW0072 establishes that a stable trained graph is causal: freezing the seed0
graph lifts the fixed three-seed mean to `.781407/.478038/.616746`. SW0068
showed that joint encoder/core training with the generic binding objective
causes harmful co-adaptation. Peer PV2_0039 now supplies the missing constraint:
phase-slot RGB reconstruction prevents the encoder from making every patch
alike while the PLV losses shape synchrony.

This experiment ports that loss into our aligned HDF5 path while retaining the
exact SW0072 graph. Seed1 first compares reconstruction weights `0.3` and `1.0`
at encoder LR `3e-5`; all other data, core loss, optimizer, epochs, and the
32-image contract remain fixed. The graph must remain bitwise unchanged. An
arm advances only if it exceeds SW0072 seed1 on all three patch metrics.

## Result

No arm advances. Weight `.3` scores `.67885/.39600/.49133`; weight `1.0`
scores `.68802/.46849/.50854` against SW0072 seed1
`.70876/.46810/.50988`. The trained graph remains bitwise exact in both arms.

Component swaps isolate the failure. Weight-1 learned features passed through
the unchanged SW0072 core score `.71894/.43119/.53055`, improving FG-ARI and
matched-object IoU while losing foreground IoU. The jointly trained core with
native gamma scores only `.65326/.45116/.46236`. Reconstruction prevents total
feature collapse and produces useful features, but core co-adaptation is again
harmful. The next test freezes the entire SW0072 core and tunes only the encoder
with reconstruction plus a gamma anchor to retain foreground localization.

