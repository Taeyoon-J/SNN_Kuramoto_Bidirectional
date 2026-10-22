# Final four-model comparison: completed

SW0249 completed all twelve model/seed training and evaluation jobs, final
checkpoint/Adam/history restores, and independent scoring of 4,800 masks.
The frozen research source files were not changed during the comparison.

Contract: models 240/237/244/245, seeds 0/1/2, same additional TRAIN35,000
images x10 full passes, fresh Adam/LRs/batch, native64 and fixed320 evaluation.
Each model retained its existing loss. Authorized original244 seed0/2
reproductions completed before matched additional training.

Final mean FG-ARI ranking: 240 (0.889187), 237 (0.887640),
244 (0.879873), 245 (0.879739). 240 leads mean object IoU;
245 leads foreground IoU. The 237-minus240 FG-ARI image-bootstrap
interval includes zero. Prior training histories remain different.

All four models are mask-supervised. Slot is a pre-existing reconstruction
reference, not matched35k training. Fixed320 is reused validation, not an
untouched test. No architectural, gate-causal or unsupervised claim follows.

See SW_0249_four_model_35000/RESULT_KO.md and PUBLIC_RESULT.json.
Local registry and mask visualizations are under collaborative_testing/.
All twelve jobs and coordinator exited normally; no followup training or
exploratory queues remain. Complete this goal after publication and final report.
