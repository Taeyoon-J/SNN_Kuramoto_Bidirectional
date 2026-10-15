# SW165 seed0/2 input/API checklist

Design/admission checklist only. Do not modify sealed SW164 or another running experiment.

- [ ] SW164 seed1 score has independently validated complete receipt; save exact score/prediction/protocol and frozen-solver hashes. Preserve seed1 discovery provenance.
- [ ] Call `SW130.run.source_contract(seed)` for each seed,which delegates to `SW0110.run.source_paths/train_indices`; require matching source model seed,source manifest,4096 ordered IDs and117+seed order. Source core paths are `trained_models/SW0097_graph_adaptation/seed{seed}_positive_frozen/core.pt` via that API,not a hand-built substitute.
- [ ] Expected core SHA seed0=`36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf`; seed2=`798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661`. Verify the actual source audit/proof assets required by `source_paths`; no seed1 proof substitutions.
- [ ] Use generic `SW135.foundation.load_native32_foundation(seed,device)` strict mapping and `SW143.geodesic_operator.install_operator(graph,'T')`. Verify1024 units,adaptive_pool32,source encoder/statistics hashes,source geometry/non-feedback factorized backend,no pulse/no new emitter/threshold.
- [ ] Resolve own source rows/IDs for eight TRAIN positions0–3/2048–2051. All receipt arrays must be converted to JSON-native lists before dict equality or serialization. Do not read TRAIN masks.
- [ ] Own native32 source reference: `SW137.run.validate_prediction(SW137.run.OUTPUT_ROOT/f'seed{seed}',seed=seed)`.
- [ ] Own T reference: `SW143.run.validate_arm_prediction(SW143.run._arm_dir('T',seed),seed=seed)`. Require source/encoder/stats/operator/preflight/R-reference lineage and IDs1320..1639. Never reuse T seed1 labels as seed0/2 controls.
- [ ] Capture native `prepare_coupling(...)[0]` from the same unchanged rollout. Preserve original tuple,one static preparation per image,and restore instance callable finally. Same-seed gain is `exp(found.wrapped.core.graph_generator.log_coupling_gain)`.
- [ ] Production Q uses actual four component spikes and settle512 exactly once. Run untouched/captured six-trace/Q/QCC parity at B1/T1024/live0; full320 fresh QCC must exactly replay archived own-seed T.
- [ ] Call sealed `SW164.foreground_signed.foreground_induced_merge(Q,A,gain,source_labels,grid_shape=(32,32))`; no seed argument or new parameter belongs inside this scientific kernel.
- [ ] Empty FG→all0;isolated FG→singleton;source-FG disconnected components cannot merge. All source FG is retained and source BG remains0. No second min8/largestBG pass.
- [ ] Slot paths: `trained_models/SW0092_slot_our70000_eval/seed0_epoch10/{predictions.npz,protocol.json}` and corresponding `seed2_epoch10`. Call `SW135.evaluation_contract32.validate_slot_protocol(...,seed=seed,prediction_path=...)` and `slot_native_pixels_to32`.
- [ ] Slot checkpoint SHA seed0=`0301a456a586da5e1132c703d095f4736c4351cb86388eabe2aff7f1c96ae091`; seed2=`c665b2b52cb59ee5044b91cebff039a4b5199fa3edc2b5fa769094fcadadb200`. Model/preprocessing/BG/IDs and prediction/protocol bytes are also bound by the actual validator; checkpoint SHA alone is insufficient.
- [ ] New runner paths/protocols/manifests/row ledgers include explicit seed and new experiment identity. Do not call seed1-only SW164 `_references/_out/_work` or SW162 `prepare_preflight_cpu` under a global seed monkeypatch.
- [ ] Source/candidate/Slot predictions and hashes are frozen before GT;standard modal4px/native32 scorer only. Require candidate-vs-source binary foreground equality and per-image foregroundIoU exact equality at scoring.
- [ ] Full all3 summary reuses validated seed1 plus both0/2 regardless outcomes;shared-image bootstrap and discovery/replication distinction are explicit. No new model training or reserve access.
