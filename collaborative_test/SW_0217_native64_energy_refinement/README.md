# SW0217 native64 signed-energy split/refinement

Use the actual best-reference models0/1/2, same fixed16 images and native64
baseline readout. Spectral proposals may undo a source merge only if the
original signed Q/A/gain energy improves after enforcing4-connectivity.
Split-only is secondary; split-then-warm-remerge is primary. No model update,
source foreground change, GT count, dropped component or new normalization.

New native rollouts are required:268-channel anchor caches cannot reconstruct
full energy. Actual induced Q/A remain server-private for exact energy and
operator replay audit. Baseline masks must bitwise reproduce the saved reference.
See DESIGN_KO.md. Local tests cover optimizer invariants and full-N induction;
actual CPU source/disk admission precedes exclusive-GPU queue registration.
This is a mini test; the full320 three-seed .848615 incumbent remains unchanged.

## Actual admission and start (2026-10-18)

Eight local and eight server tests passed, followed by real CPU model/data/disk
admission for each of three independent seeds. Seed0 and seed1 started on
exclusive GPUs1 and3; seed2 is registered to acquire the next idle device.
Actual process identities and NVIDIA rows are archived in started_v1; registration
and source SHA bindings are in deployment_v1/registration_receipt.json.
Peer patch_v2 was rechecked at a0a39dfb081c1012e1beeae1136c4e7a9ef6073b.

No new result is claimed at registration. This tests whether the existing native
signed energy can correct grouping by splitting and remerging, rather than
training another unconstrained merger. It does not establish gating causality,
data scaling, fresh native64 70k training, or full320 FG-ARI above .90.
