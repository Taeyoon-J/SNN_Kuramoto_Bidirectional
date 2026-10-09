# SW0122: joint analytic-RGB replication on source seeds 1 and 2

SW0117 seed 0 showed a paired candidate-minus-control FG-ARI difference of `+0.009316`, but did not pass its preregistered `+0.01` promotion gate. SW0122 tests whether that signal replicates on the two other registered SW0097 source seeds. It preserves the earlier SW0117 result and status.

For each seed, the control and candidate start from the same complete source core and registered encoder. Both train the encoder, legacy graph, and remaining core parameters for 256 updates on that seed's exact matched 4096-image order, with batch 16, 64/32 time/settle, Adam rates `3e-5` for core/graph and `3e-6` for the encoder, and joint clip norm 1. The candidate adds the unchanged SW0115 analytic partition RGB loss with the immutable SW0117 seed-0 coefficient `7.865416617457706`; it is never recalibrated per seed. Each source's already completed 320-image SW0097 evaluation and SHA are validated as its baseline.

Preflight retains the SW0117 cached-versus-live gamma and old-loss checks and exact production labels, plus same-input reference identity. It checks the first four registered training batches for finite gradient credit and uses a disposable update to verify the actual optimizer path. No ground truth is read during preflight or training. Evaluation uses the unchanged actual-spike classifier on IDs 1320–1639 with the registered 1024/512 evaluation horizon, threshold `.50`, minimum component size 2, and largest-component background.

The replication gate averages all three seeds, combining historical SW0117 seed 0 with new seeds 1 and 2. It requires candidate mean FG-ARI to exceed both source and matched control means, candidate gains over source in at least two seeds, positive 95% paired-bootstrap lower bounds for candidate versus both control and source, and both IoU margins over the matched Slot reference. Each of 10,000 bootstrap draws uses the same sampled image indices across all three seeds. All arms and seeds are reported; no seed is selected after observing results. This is a 256-update replication, not a further 70k training pass.

The registered runner completed actual source validation, preflight, training and evaluation; see final results below.

## Actual execution (2026-10-09)

The six frozen research files passed SHA-verified deployment. Actual server validation passed for historical seed0 and both seed1/2 source contracts, with 13 correctly bound queue commands. The owner-aware parallel supervisor (PID1407513) holds global per-GPU leases and schedules four paired pipelines. Seed1 control/candidate and seed2 control passed actual B16 GPU preflight and started training on GPUs0/1/3 (PIDs1407943/1407979/1407958). Seed2 candidate is queued for a free device; occupied GPU2 is not shared. Completed seed0 is not rerun, and no stage is automatically retried after failure. Scores remain pending. Server logs/state are in results_archive/queue_status_20261009.json and sw0122_*_20261009.log.

## Final results (2026-10-09)

All 13 registered stages validated. Supervisor1407513 and final evaluation1414256 were absent at terminal verification. Historical SW0117 seed0 was reused unchanged. Metric order: FG-ARI / foreground IoU / matched object IoU.

| Arm | Seed0 | Seed1 | Seed2 | Three-seed mean |
| --- | --- | --- | --- | --- |
| source | 0.824868/0.719872/0.655663 | 0.742658/0.590283/0.584964 | 0.761519/0.421056/0.563985 | 0.776348/0.577070/0.601537 |
| control | 0.816894/0.719649/0.648146 | 0.733538/0.595065/0.577770 | 0.749537/0.424454/0.557701 | 0.766656/0.579723/0.594539 |
| candidate | 0.826210/0.722913/0.659548 | 0.730189/0.580067/0.572957 | 0.745623/0.420761/0.552367 | 0.767341/0.574580/0.594957 |

Candidate-minus-control FG: +0.000685, paired-image 95% CI [-0.003592,+0.005012]. Candidate-minus-source FG: -0.009007, CI [-0.014236,-0.003776]. Same bootstrap indices were used across all three seeds. Candidate improves source seed0 but lowers seeds1/2; its mean FG is below matched Slot .774933. Registered promotion failed; SW0097 remains incumbent. End this RGB recipe without coefficient sweeps or 70k expansion. This replication establishes neither every gating contribution nor data-scaling improvement.

Full provenance and gate checks: `three_seed_summary_20261009.json`. The earlier running paragraph is historical and superseded by these final results.
