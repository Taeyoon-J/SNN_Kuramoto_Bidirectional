# SW0191 — production-partition error feedback

Training queued; seed0 has actually started on GPU2. No validation score yet.
Keep the SW0186 incumbent and production masks, adding a class-balanced loss
on actual split/merged TRAIN patch pairs. Mined labels stay fixed across each
proposal's gradient and acceptance measurements. Separately check that this
auxiliary loss reaches emitted spikes, beyond the existing QCE gradient.

See [registered design](DESIGN_KO.md). Native32 joint70k, FG.87, automatic count
prediction and gate-causality proof remain outstanding. No new performance claim.

Local/server47 tests and six actual CLIs passed. All three exact incumbent CPU initializations and four-GPU hardware compatibility were admitted before launch. Eight actual supervisors were verified live. Seed0 first-two real proposals passed frozen-context full-forward parity, emitted-S and feedback-only emitted-S/trainable-parameter credit, five-family total credit and committed CNN/Adam learning. Scoped receipts are in results_archive/cpu_v1 and start_v1. Other seeds remain queued; no new FG result or gate-causality proof is claimed.
