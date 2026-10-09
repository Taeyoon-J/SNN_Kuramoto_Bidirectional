# SW0129: shared RGB loss replication

This is a new, preregistered replication of the SW0106 shared RGB reconstruction recipe on seeds 1 and 2. SW0106?s original seed-0 expansion gate failed and stays unchanged. A reviewed posthoc analysis of seed 0 motivates this independent replication; it does not retroactively promote SW0106.

Both arms start from their corresponding SW0095 source model and the same seed-specific, 32-batch decoder warmup. The candidate adds the frozen seed-0 coefficient times the original hard-partition RGB assignment loss. The paired control runs the same objective without assignment credit to the core and encoder. Data order, optimizer settings, training horizon, and native spike-based evaluation follow SW0106 exactly. There is no ground-truth training signal.

The copied `results_archive/seed0_original_preflight.json` is the byte-for-byte original passed seed-0 coefficient record. Before preflight, the runner verifies its coefficient, SW0095 source fingerprint, warmup artifact SHA, and the actual warmup artifact bytes. Seed-1/2 gradient ratios are recorded for diagnosis only; they do not recalibrate the coefficient.

Run the stages only through the owner-aware queue after deployment review:

- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage preflight --seed 1 --device cuda:0 --output ...`
- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage train --seed 1 --arm candidate --device cuda:0 --output-dir ...`
- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage eval --seed 1 --arm candidate --device cuda:0 --output-dir ...`

The registered continuation gate requires candidate mean FG-ARI above both source97 and matched controls, at least two per-seed FG-ARI gains against each, positive 95% shared-image paired bootstrap lower bounds against both, and IoU means above the fixed matched Slot baselines by 0.05. It reports every seed even on failure. Passing would support this reconstruction-credit recipe only, not prove gating contributions or data scaling. See `protocol.json` for the frozen contract.
