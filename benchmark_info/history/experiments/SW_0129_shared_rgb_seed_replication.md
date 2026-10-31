# SW_0129_shared_rgb_seed_replication

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## mechanistic_status_20261009.md

<details>
<summary>mechanistic_status_20261009.md</summary>

Source SHA256: `582655895f02c7723375dd561e713d5ef6f27c9e1dcf926af0f403fc8225e07b`

````markdown
# Mechanism and scaling status after the seed-1 replication

Neither contribution from every gating stage nor improvement with dataset size is established. A reconstruction-credit improvement alone cannot establish either claim.

## What the existing implementation actually gates

`MembraneLayer.forward` computes the usual leaky update and reset, and preserves the previous membrane only when `g_wave_t == 0`. The native sinusoidal gate is continuous: `(1 + sin(delayed_phase_mean))/2`. It is generally nonzero; therefore a small gate normally reduces the emitted spike amplitude but does not reduce the membrane integration update. This code observation identifies a possible mismatch between phase-dependent integration and the implemented dynamics. It does not prove that replacing the recurrence improves segmentation.

The SW0108 seed-0 intervention measured binary-event occupancy about 0.999785 before forcing events on. Under this condition the emitted trace `event * gate` nearly equals the continuous gate. Forced-on events and removed dendritic/membrane retention did not reduce the evaluated FG-ARI materially. Conversely, disrupting upstream coupling/drive caused collapse. These are frozen-model interventions, not retrained architecture ablations, and cannot establish an independent contribution from every stage or from their training gradients.

SW0126 made events vary through history-centered thresholds but degraded native QCC performance; SW0127 showed degradation even before training the history adapter. Repeating sparsification or replacing the classifier without preserving a native source comparison is not supported by those results.

## What more data has shown

The existing equal-update seed-0 SW0109 comparison gave FG-ARI 0.822080, 0.819832, and 0.818679 at 2,500, 10,000, and 70,000 unique images. This does not establish a scaling benefit or a statistically reliable decrease. Fixed updates also change repeat exposures per image. Both matched-compute and adequately trained learning curves are needed before attributing the result to dataset size.

SW0129 uses 4,096 unique additional training images from the registered 70k pool, starting from SW0095. It is a continuation pilot, not a new complete 70k training result. Seed 2 failed its unchanged assignment-use preflight and was not trained. Seed-1 endpoint evidence must stay separate from selected historical seed 0 and must not be reported as a three-seed mean.

## Peer check

The peer `patch_v2` branch remains at `697eeb67dc031325b63153e678cea8cbc602da31`, verified by fetch in this cycle. PV2_0053 reports GT-audited appearance/position geometry, not an improved trained score. Our SW0110 already tested an XY graph route without FG-ARI improvement. PV2_0048 also found worse scores with more unique images at fixed compute, with fewer exposures per image; its full-epoch large-data follow-up was cancelled. Neither peer result proves a solution to our scaling requirement.

The next architecture decision is under separate review: preserve useful native event/phase information while making the downstream temporal mechanism testably functional. No new architecture training, recipe change, or full-data expansion has been launched by this note.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `56d0149b9fbd457ce7fb5e10cac1cf3bc063235a72338a36cf00b811bab3c654`

````markdown
# SW0129: shared RGB loss replication

This is a new, preregistered replication of the SW0106 shared RGB reconstruction recipe on seeds 1 and 2. SW0106?s original seed-0 expansion gate failed and stays unchanged. A reviewed posthoc analysis of seed 0 motivates this independent replication; it does not retroactively promote SW0106.

Both arms start from their corresponding SW0095 source model and the same seed-specific, 32-batch decoder warmup. The candidate adds the frozen seed-0 coefficient times the original hard-partition RGB assignment loss. The paired control runs the same objective without assignment credit to the core and encoder. Data order, optimizer settings, training horizon, and native spike-based evaluation follow SW0106 exactly. There is no ground-truth training signal.

The copied `results_archive/seed0_original_preflight.json` is the byte-for-byte original passed seed-0 coefficient record. Before preflight, the runner verifies its coefficient, SW0095 source fingerprint, warmup artifact SHA, and the actual warmup artifact bytes. Seed-1/2 gradient ratios are recorded for diagnosis only; they do not recalibrate the coefficient.

Run the stages only through the owner-aware queue after deployment review:

- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage preflight --seed 1 --device cuda:0 --output ...`
- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage train --seed 1 --arm candidate --device cuda:0 --output-dir ...`
- `python -m collaborative_test.SW_0129_shared_rgb_seed_replication.run --stage eval --seed 1 --arm candidate --device cuda:0 --output-dir ...`

The registered continuation gate requires candidate mean FG-ARI above both source97 and matched controls, at least two per-seed FG-ARI gains against each, positive 95% shared-image paired bootstrap lower bounds against both, and IoU means above the fixed matched Slot baselines by 0.05. It reports every seed even on failure. Passing would support this reconstruction-credit recipe only, not prove gating contributions or data scaling. See `protocol.json` for the frozen contract.

````

</details>
