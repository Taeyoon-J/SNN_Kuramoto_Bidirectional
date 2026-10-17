# SW0191 — production-partition error feedback

Completed and independently audited. Standalone three-seed means:
FG-ARI .799773 / foreground IoU .478037 / object IoU .659912.
All three decreased versus SW0186; this recipe is closed without a small sweep.
See RESULT_KO.md and results_archive/score_v1/summary.json. SW0186 remains best.
Keep the SW0186 incumbent and production masks, adding a class-balanced loss
on actual split/merged TRAIN patch pairs. Mined labels stay fixed across each
proposal's gradient and acceptance measurements. Separately check that this
auxiliary loss reaches emitted spikes, beyond the existing QCE gradient.

See [registered design](DESIGN_KO.md). Native32 joint70k, FG.87, automatic count
prediction and gate-causality proof remain outstanding. No new performance claim.

Seed0 training completed: all64 registered proposals were measured, with49
accepted updates. The completed model, Adam steps, frozen feedback contexts and
actual GPU ledger passed artifact validation; text receipts are in
results_archive/completed_seed0_v1. Seed2 also completed all64 proposals, with38
accepted updates and the same completed-artifact checks. Seed0 fixed320 masks
are frozen and validated without GT; receipts are in results_archive/frozen_seed0_v1.
All three fixed320 predictions and the independent score audit are now complete.

Seed1 completed all64 proposals with50 accepted updates. Its actual worker ledger
records GPU1, GPU1, then GPU2; the final history preserves the original first-two
model/Adam/RNG transaction digests. This exercised continuation on another
physical GPU without a fresh experiment restart. It does not establish bitwise
equivalence to uninterrupted training. All-seed receipts and hashes are in
results_archive/all_training_v1/completion_summary.json.

Local/server47 tests and six actual CLIs passed. All three exact incumbent CPU initializations and four-GPU hardware compatibility were admitted before launch. Eight actual supervisors were verified live. All three seeds' first-two real proposals passed frozen-context full-forward parity, emitted-S and feedback-only emitted-S/trainable-parameter credit, five-family total credit and committed CNN/Adam learning. Scoped receipts are in results_archive/cpu_v1 and start_v1. No new FG result or gate-causality proof is claimed.
