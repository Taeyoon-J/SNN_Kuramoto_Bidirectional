# SW0188 — incumbent weight continuation

Preparation in progress; no new GPU training or performance claim yet.
The complete matching-seed SW0186 weights are retained, including its learned
RGB CNN and context. Five-group Adam is explicitly reset. Additional TRAIN512
twice/64 proposals, QCE and production masks remain unchanged. This tests
continued learning from the actual incumbent, not new data scaling.

See [registered design](DESIGN_KO.md). Evaluation compares fixed320/native32
three-seed results directly with SW0186 and comparable Slot70k. No .87,
unsupervised, full native32 joint70k or gate-causality claim is made.

Registered and running: all eight supervisors verified live; seed0/1 actual GPU workers started, seed2 is queued. Local/server39 tests, six actual CLIs, all three exact incumbent tensor digests, trained CNN hashes, five-family empty Adam and Slot archive admission passed. The first seed0 actual GPU update has already passed full-forward parity and changed the trained CNN. First-two admission and full320 scores remain pending. Peer patch_v2 was checked this cycle and remains cab6c676. Receipts: results_archive/cpu_v1 and start_v1.

Seed0 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed0_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed performance result.

Seed1 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed1_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed performance result.

Seed0 training completed and its actual checkpoint/Adam/CNN and matching SW0186 start were validated:64 proposals,33 accepted; metadata are in results_archive/completed_seed0_v1. Seed2 has started on GPU2. Seed1 yielded GPU3 to a foreign process with its11-proposal/6-accepted checkpoint preserved and the same supervisor waiting to resume. Predictions/scoring remain queued. No completed fixed320 score or incumbent change is claimed. Model sources are unchanged.
