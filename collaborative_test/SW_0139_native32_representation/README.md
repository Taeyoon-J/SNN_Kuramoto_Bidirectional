# SW0139 native32 representation pilot

Compare old-loss control, trainable context residual encoder, and cross-view spike consistency. All arms use genuine 32x32 patches, the same ordered4096 TRAIN images and256 logical updates. Encoder and learned graph train together; oscillator, dendrite, membrane and integration parameters are frozen. No TRAIN masks, object counts or learned assignment head. Cross-view doubles view exposures and is not FLOP matched.

Local/server CPU tests18/18, actual source RGB gamma parity and optimizer groups verified. Supervisor2711934 and three GPU disposable preflight/calibration workers verified live. Fresh training follows successful disposable preflights; fixed320 endpoint predictions precede GT scoring. This is a seed1 pilot, not fresh70000-image training or evidence of gating necessity. Publication remains pending scope approval.

Control training PID2712399 GPU1 and context training PID2712528 GPU2 subsequently verified live after disposable preflights passed. Cross-view calibration remains running, its training queued. See training_start_20261010.json.

Control and context_residual seed1 training completed and checkpoint/optimizer/history hashes verified. Archived histories independently confirm the same4096 unique ordered images,256 B16 updates/B4x4 and one pass without GT. Both mask evaluations have started; cross_view training continues. Actual run times were5127.19s/5152.43s, about20.03s/20.13s per logical update. The roughly24.34h/24.46h70k one-pass projections are extrapolations only. All256 updates in both arms activated global clip1; median clip multipliers.1823/.1315. This is a potential optimization diagnostic, not a causal explanation or task result. See results_archive/completed_control_context_training_analysis.json.

Completed control/context predictions were frozen and scored with the registered evaluator in an additional CUDA-hidden CPU process2859188; final all-arm evaluation path is preserved for the queue. Per-image source metrics exactly reproduceSW0137; means and paired10k image intervals were independently recomputed. Seed1 FG/foregroundIoU/objectIoU: source.754135/.606900/.582201; control.726187/.592371/.557006; context.666608/.650807/.477581. Context minus source FG is-.087528 (95%[-.104987,-.070645]); minus control-.059579 (95%[-.077038,-.042826]). Better foreground coverage accompanies worse object assignment. The registered context expansion gate failed; do not expand this fixed recipe to other seeds or70k. Cross_view remains pending, not failed. No incumbent change or three-seed superiority claim. See results_archive/control_context_analysis.json and evaluation_seed1/evaluation_control_context.json.


## Seed1 partition-error diagnostic

The completed control/context output was decomposed on true-foreground patch pairs. The context arm shows a larger background-label merge component and higher missed-foreground fraction; see [the analysis](partition_failure_analysis.md) and the hash-bound [diagnostic JSON](results_archive/partition_diagnostic_seed1.json). This is descriptive and does not change the source97 incumbent or establish a causal layer claim.


## Completed all-arm seed1 endpoint

All11 queue tasks completed successfully; terminal state/log snapshots and all3 prediction protocols are archived. Fixed320 per-image means and common paired10k bootstrap intervals were independently reproduced.

| Arm | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| control | 0.726187 | 0.592371 | 0.557006 |
| context_residual | 0.666608 | 0.650807 | 0.477581 |
| cross_view | 0.738505 | 0.592047 | 0.566686 |
| source97 | 0.754135 | 0.606900 | 0.582201 |

Cross-view improves FG over old-loss control by.012319 (paired95% CI[.004287,.021231]) but remains below source by.015630 (CI[-.024289,-.007376]). Both registered candidate expansion gates fail. Close these exact recipes without seed/70k expansion; incumbent staysSW0137. The conditional C-only/C+source-Q objective pilot is now justified for preparation, with actual integration/preflight required before training. No new pilot has run. This endpoint is one-seed continuation, not fresh70k native32 training or a scaling proof.
