# SW0140 native32 causal screen

Frozen source97, native1024 patches, full1024-step dynamics. Three seed jobs with first four ordered TRAIN RGB images each; no GT or optimizer. Compare unchanged dynamics and six isolated interventions: Kuramoto interaction, delayed sinusoidal gate, dendritic history, membrane voltage history, spike reset and event selection. Changed emitted spikes feed later dendrite/reset recurrence. Exact zero-gate holds remain as specified.

Local/server9 CPU tests, actual three-source order and CPU native1024 T64 baseline parity passed. Final two-file return-contract correction was verified before GPU launch. Supervisor2726916, seed0 PID2726941 on GPU3 verified live; seed1/2 queued. Global exclusive GPU leases, CUDA-hidden supervisor, no retries. This is a trace/resource screen, not segmentation performance or proof of stage necessity. Fixed endpoint evaluation follows technically valid screens. Current introduced integration scalars are zero; their usefulness cannot be credited by this source screen. Publication remains pending scope approval.

All three GT-free screens completed and independently verified: native1024 full1024 NONE six-trace exact parity, source/encoder state preserved, downstream theta invariance. All six interventions remain eligible for fixed320 endpoint evaluation; no GT or effect-size filter. See results_archive/causal_screen_analysis.json.

Fixed320 endpoint queue launched after15 CPU tests and actual three-seed source/Slot/RGB verification. Supervisor2808950; Kuramoto-off seed0 PID2808974 verified live on physicalGPU3. Six conditions x three seeds, genuine1024-patch actual-spike QCC; all18 predictions must validate and freeze before GT scoring. Other jobs wait for exclusive global leases and run automatically when GPUs become available. No optimizer or new training in140;139 training continues independently. Endpoint scores are complete; the fixed320 results and interpretation are summarized below. See endpoint_execution_start_20261010.json.

## Fixed320 causal endpoint results

The endpoint queue completed **19/19 tasks**. It validated and froze all 18 intervention predictions (six conditions x three seeds) before opening targets. Scoring used the fixed native32, 320-image endpoint (IDs 1320 to 1639); prediction did not use ground truth and the endpoint made no optimizer updates. The six intervention conditions start from the frozen SW0097 source dynamics and change only the listed operation; source97 and Slot32 are fixed references.

| Arm | FG-ARI | Foreground IoU | Matched object IoU | delta FG-ARI vs source97 (paired 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Source97 | 0.781919 | 0.605427 | 0.597014 | n/a |
| Slot32 reference | 0.781652 | 0.204360 | 0.203165 | n/a |
| Kuramoto interaction off (K=0) | 0.056074 | 0.235064 | 0.085188 | -0.725844 [-0.741371, -0.709412] |
| Delayed gate = 0.5 | 0.000000 | 0.000000 | 0.000000 | -0.781919 [-0.797199, -0.765770] |
| Dendrite history off | 0.771650 | 0.597424 | 0.580655 | -0.010269 [-0.012928, -0.007722] |
| Membrane voltage history off | 0.782225 | 0.604921 | 0.598467 | +0.000306 [-0.000702, +0.001454] |
| Spike reset off | 0.781883 | 0.605771 | 0.596583 | -0.000036 [-0.000584, +0.000452] |
| Events forced on | 0.781883 | 0.605771 | 0.596583 | -0.000036 [-0.000584, +0.000452] |

Intervals are paired over the same images with 10,000 `RandomState(140)` bootstrap resamples; the table reports FG-ARI differences from source97. The Slot32 reference is included for comparison. Root-verified per-seed source and Slot prediction hashes, all intervention prediction hashes, queue evidence, and evaluation hashes are recorded in the [endpoint execution receipt](results_archive/causal_endpoint_execution_receipt_20261010.json).

The frozen intervention effects differ by operation. Removing Kuramoto interaction (`K=0`) sharply lowers all three metrics; replacing the delayed gate with 0.5 in the carrier, integration, emission, and hold path collapses the readout to zero. These show strong sensitivity to those particular frozen-source interventions. Turning off dendritic history causes a smaller but consistent FG-ARI decline. Turning off membrane-voltage history gives a tiny higher FG-ARI mean, but its paired interval crosses zero; matched object IoU rises only about 0.00145. Turning off spike reset has essentially no FG-ARI change. Forced-on events and reset-off produce identical saved prediction payloads for each seed; that establishes equality of saved labels/IDs, not equality of raw trajectories.

This screen does not prove that every gate is useful or necessary after retraining, and it is not a data-scaling result. SW0139 cross-view is still incomplete, so there is no promotion or incumbent change from this endpoint. The source97 incumbent remains unchanged. See the [full evaluation](results_archive/causal_evaluation.json) and [independently audited analysis](results_archive/causal_endpoint_analysis.json).
