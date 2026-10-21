# SW0249: four frozen models, matched additional TRAIN35,000

The user ends open-ended model/loss improvement. Compare SW0240, SW0237,
SW0244-within16 and SW0245-within16 only. Twelve model/seed jobs share the
same deterministic35,000 TRAIN IDs, ten complete shuffled passes, Adam reset,
learning rates3e-5/3e-4/3e-4, clipping1, native64, rollout1024/settle512 and
fixed320 evaluation IDs1320..1639. Model-specific existing loss coefficients
remain1/1/16/16. No loss or architecture sweep and no candidate-screen gate.

The new adaptation stage uses the same trainable whole encoder and sensory
projection for all models; all other core states and the readout remain frozen.
This standardizes new training conditions, rather than reproducing the
encoder-only freeze policy of244/245's historical128-update stages.
All use the unchanged original217 teacher and established mask supervision.
SW0240 has already received70k additional training; this inherited history is
retained and reported, so the comparison does not equalize lifetime compute.
SW0244 and245 share a recipe and are not independent architectural hypotheses.

35,000 is not divisible by16: use2,187 full batches and one8-image batch per
pass, ten passes,21,880 updates and350,000 presentations exactly. Never pad,
drop images, or silently substitute35,008/34,992. Shuffle matches across models
within a seed. The35k subset is sampled once from the original70k TRAIN pool;
reserved IDs1000..1639 are excluded.

User explicitly authorized reproducing SW0244's missing seed0/2 using its
original128-update encoder-only within16 recipe, original64 TRAIN IDs/teacher,
and original244 RNG seed convention. These are labeled new reproductions,
not original historical checkpoints. Reproduction updates are reported
separately from the common35k stage. Seed1 uses the actual244 checkpoint.

Each job completes CPU source/gradient and GPU loss/readout admission before
training. Independent worker checkpoints contain encoder, actual projection,
Adam, RNG, history and immutable bindings; a failed job retains its logs and
last committed boundary. No foreign process arrival triggers yielding. Shared
GPU admission requires remaining memory for the unchanged80%-allocation cap
plus512MiB headroom. Cooperating jobs still lease one worker per physical GPU.

Retain intermediate full-state checkpoints near first-pass2,500 and10,000
distinct images and at complete passes1/2/4/6/8/10. They permit later learning
curve evaluation without retraining. Do not infer score saturation from loss
alone; this fixed final comparison does not select checkpoints using scores.

All twelve final prediction protocols and Slot references precede GT scoring.
Compute all three metrics, paired image contrasts and per-seed values. Audit
actual final Adam counters/history, checkpoint state and raw masks. Archive
aggregate results and share research text only. End the goal after all twelve
jobs, scoring, audit and final report; do not queue exploratory followups.
