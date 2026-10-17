# SW0186 — full source RGB encoder plus unchanged QCE

Three training supervisors, three prediction supervisors, GT-last scoring and
an independent final auditor are registered. All three seeds have actually trained
on GPU. All three seeds have completed all64 proposals with34/36/43 accepted
updates (seed0/1/2). Their actual checkpoints, optimizers and histories passed
validation, including the original CNN's changed weights. Metadata is preserved
in `results_archive/completed_seed0_v1`, `completed_seed1_v1` and `completed_seed2_v1`.
Seed0's fixed320 conditional/standalone native32 predictions are now frozen and
validated against their row ledger and unchanged model; receipts are preserved in
`results_archive/frozen_seed0_v1`. GT has not been scored. Seed1/2 prediction
supervisors remain registered on exclusive idle devices. Foreign owners are preserved.
No fixed320 evaluation score is available.

Only the original source CNN's trainability changes relative to the QCE setup:
source_base is a fifth Adam family at LR3e-5. Existing four rates, tensor
initialization, zero-output RF31 context, QCE, TRAIN512x2/64-proposal budget and
native32 S→Q→QCC→SW165 readout remain registered in [the design](DESIGN_KO.md).
This is supervised continuation, not a fresh70k or unsupervised experiment.

All three actual CPU factories matched SW181's initial tensor SHA and the
original native gamma forward exactly. Sixteen training tests, then36 combined
training/evaluation tests and four real executable CLI checks passed. Eleven
additional auditor tests, its CLI and all three actual Slot archive loads passed.

Seed2's actual first two proposals had positive emitted-S/five-family credit
and full-forward parity. Both were rejected and rolled back. By the recorded
later boundary, a committed update had actually changed the source CNN SHA and
its Adam state. This proves a working learning path, not improved segmentation
or each gate's usefulness. See `results_archive/start_v1` for exact scoped proof.

The final score requires actual source-CNN learning in all seeds before GT
access. The auditor uses the explicit Slot archive loader from185v2, not an
assumption about candidate reference keys. No new fixed320 score is available;
SW181 remains the supervised FG incumbent.
