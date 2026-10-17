# SW0191 — production-partition error feedback

Training active; all three seeds have passed actual GPU feedback admission. No validation score yet.
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
Other predictions and the three-seed score remain pending.

Local/server47 tests and six actual CLIs passed. All three exact incumbent CPU initializations and four-GPU hardware compatibility were admitted before launch. Eight actual supervisors were verified live. All three seeds' first-two real proposals passed frozen-context full-forward parity, emitted-S and feedback-only emitted-S/trainable-parameter credit, five-family total credit and committed CNN/Adam learning. Scoped receipts are in results_archive/cpu_v1 and start_v1. No new FG result or gate-causality proof is claimed.
