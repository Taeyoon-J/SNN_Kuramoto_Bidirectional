# SW0169 training and automatic evaluation registration

2026-10-15. Two supervised TRAIN512 pilots were launched after the actual
512-row CPU admission and eight-row production-rollout diagnostic passed
(562.49 seconds, no validation data). Both use fresh source initialization,
the same frozen current normalization, maximin loss, B16/micro4, and 64 updates.
Only explicit membrane-history gradient credit differs; emitted-spike feedback
remains live in both arms. Disposable admission weights and Adam are not reused.

Training supervisors: history_live 3716549 / worker 3716709 (physical GPU2);
history_detached 3716550 / worker 3716710 (physical GPU1). At this archived
observation both actual workers were live in CPU preparation. Queue status
`running_training` alone is not evidence of CUDA learning or a committed update.

Automatic dependencies are registered and independently observed live:
prediction supervisors 3719599 and 3719615 wait for both final training artifacts;
CPU scorer 3719633 waits for both complete fixed320 prediction artifacts. Prediction
uses no validation masks. GT is opened only after all prediction bytes are frozen.
Original Source-T, SW164 and Slot reference predictions/protocols were checked
against the same 320 IDs and native32 schema on the real server, without GT.

Local evaluation/queue tests: 11 passed. Server training tests: 36 passed;
server evaluation/queue tests: 11 passed. Root rechecked deployed file hashes,
CPU receipts and actual /proc argv/liveness before archiving this evidence.

There is no SW169 performance result yet. SW165 remains the verified three-seed
incumbent (.804298915 / .609975448 / .651148975). This pilot does not prove
FG-ARI .87, fresh native32 70k scaling, or causal utility of every gate.

Evidence: results_archive/training_and_evaluation_registered_20261015/
server_records.json and independent_verification.json. Generated files physically
reside under collaborative_testing/generated via the existing archive junction.
