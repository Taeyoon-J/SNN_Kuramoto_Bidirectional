# Peer evidence reviewed before the next scaling study

Source: origin/patch_v2 commit697eeb6, fetched2026-10-08. Read
PV2_0048_data_scale, PV2_0051_data_quality, PV2_0052_window and
PV2_0053_position_diag. Their scores use a different evaluation contract;
none replaces our native320 Slot reference.

PV2_0049_slot_draws was also reviewed. Peer Slot inference now uses three
seeded draws per training checkpoint and verifies identical labels on a
repeated same-seed run. Its large third-training-seed degradation explains
the changed three-seed reference. Our existing Slot SW0092 uses fixed
inference seed0; report that distinction instead of importing peer means.

At approximately240k presentations, peer unique-image counts6000/20000/40000
gave FG-ARI .6779/.5983/.4790. Repetitions per image also fell40/12/6.
This demonstrates degradation at that fixed compute budget, not that more
data cannot help with sufficient optimization. Their within/between-feature
diagnostics describe those fitted encoders; they do not resolve an unrun
longer-training counterfactual. SW0109 likewise must report its result as
matched-compute downstream scaling from inherited2500-image cores.

Peer PV2_0053 finds a close-object-pair separation issue in appearance-only
features. Appending coordinates reduces its GT-defined unseparable-image
fraction. This is a diagnosis, not a trained mask-score improvement. Our
graph already has a spatial-distance prior; explicit coordinate features
would be a distinct proposed change requiring a controlled actual-spike
evaluation. Inspect our own feature geometry before adopting the change.

Peer PV2_0052 gains only about.0005 FG-ARI from extending inference1024 to
4096. Avoid another expensive window sweep without evidence that our
current checkpoints have a sampling bottleneck. Our SW0098 longer-training
window already failed the fixed three-seed FG-ARI comparison.

Current causal gate evidence remains pending corrected SW0108 baseline
reproduction. Test-time neutralization measures checkpoint sensitivity;
matched retraining ablations are still required for a stronger necessity
claim in the paper. No GT-derived teacher or classifier setting is adopted.
