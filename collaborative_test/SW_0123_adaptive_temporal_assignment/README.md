# SW0123: adaptive temporal dynamics and a spike-derived assignment head

**Current outcome: all27 stages/nine trajectories completed; pilot failed. No70k expansion and no incumbent replacement. See pilot_summary_20261009.json and completed_outcome_20261009.json.**

Prospective three-arm, three-seed experiment. The fixed dataset, modal16x16 patch GT and three patch metrics are unchanged. The primary classifier is new and consumes actual spike history; original QCC is secondary.

Compare legacy full dynamics, adaptive temporal full dynamics, and gate-only counterfactual, all with the same temporalCNN/11-slot assignment/nativeRGB spatial decoder. Adaptive dynamics use registered TRAIN-only calibration, .9 initial retention and adaptive binary event threshold. No raw RGB/gamma/theta classifier bypass. Detailed constants, schedule, failure/expansion gates and scientific limits are registered in design.md and protocol.json.

## Local implementation review (2026-10-09)

The opt-in dynamics module, classifier/decoder, and deterministic slot readout are implemented locally. Eleven focused CPU tests and compilation passed. A separate design/code review confirmed component fold b*4+d, binary-event adaptation, gate holds, weighted-spike reset/feedback and live gradients; the head/decoder has no direct RGB/gamma/theta input. This PASS covers modules only. Calibration/runner/checkpoint/real-data GPU preflight and nine-arm training/evaluation are pending. No SW0123 training, deployment or performance score exists yet.

The SW0109 single-seed older frozen-recipe scaling diagnostic is separate evidence; it does not prove this model scales. Positive masks alone will not establish every gating contribution: the registered follow-up includes frozen interventions and matched retrained ablations, followed by70k/scaling evaluation and untouched confirmation.

## Actual TRAIN calibration (2026-10-09)

Eleven code/protocol/test files were SHA-verified deployed. Actual server17CPU tests passed; all three strict source checkpoint/manifest/4096ID-order contracts passed. The four original TRAIN B16 cached-gamma batches were then run for each seed on GPUs0/1/3, using inherited global GPU leases. All three processes1438381/1438386/1438401 are terminal with durable passed calibration reports. No optimizer or source checkpoint update.

Adaptive binary event occupancy ranges: seed0 .1142-.1374, seed1 .1548-.1641, seed2 .1855-.1991. Mixed-event image/patch unit fractions are >=.99835 across all components/seeds; all fixed guards pass. Fixed TRAIN-only kappa/RMS arrays and implementation fingerprints were revalidated before download. Persisted results are in calibration_results_20261009.json and calibration_summary_20261009.json. The candidate avoids always-on/silent events at initialization; this is not proof of useful object binding or a segmentation gain. Full training/evaluation remains pending.

Predeployment integration review caught and corrected legacy tuple concatenation and NumPy order comparison; execution-path regression tests were added before any GPU run. Historical failures/recipes are untouched. Calibration is not rerun or recalibrated for the forthcoming matched arms.

## Actual runner preflight (2026-10-09)

SHA-verified runner deployment passed21 server CPU tests, all three preserved calibration bindings and15 fingerprint files. Adaptive seed0 actual B16 preflight PID1449390 stopped with Triton JIT cache metadata OSError116 Stale file handle; no scientific artifact was produced and its failure log is preserved. After confirming that process absent, identical code/recipe was rerun with a fresh /tmp cache and an inherited idle GPU1 lease. PID1450058 is terminal: its durable preflight report and final log both pass.32 head warmup steps and the throwaway two-Adam update verified actual changes in encoder, graph, oscillator drive, dendrite, membrane, assignment head and decoder. This proves execution and gradient credit only; it does not prove segmentation gain or individual gating contribution. All nine-arm training/evaluation remains pending.

Post-preflight provenance audit found that load_calibration reused its path variable while checking dependency fingerprints, returning the last dependency instead of the calibration report. The activity arrays and optimizer/gradient guards remain evidence of executed checks, but calibration_sha256 in this attempt is incorrect. This attempt cannot authorize joint training. Its report and warmup will be preserved, the path binding corrected with an execution-path regression, and preflight rerun under the corrected fingerprint. No joint training has started.

## Nine-trajectory server queue (2026-10-09)

The corrected frozen seven-file runner/evaluator/dispatcher bundle passed30 local and30 actual server CPU tests and bounded independent reviews. Actual load_calibration paths and the three report SHA bindings are now correct. Prior seed0 preflight/warmup and runner version are preserved under results_archive/preflight_attempt_invalid_calibration_binding_20261009; scientific constants/calibration are unchanged. Owner-aware supervisor1455390 and initial workers1455436/GPU0,1455437/GPU1,1455438/GPU3 were verified live. All27 preflight/train/evaluate jobs across three arms and three seeds are queued. All adaptive-seed preflights must pass before joint training; scientific post-training guard failures still preserve completed trajectories for evaluation. Only exclusively idle GPUs are leased, children inherit leases, runtime Triton caches use local/tmp, foreign workers are untouched and failed tasks are not automatically retried. No SW0123 segmentation score or joint-training completion is claimed.

All nine actual B16 preflights have now passed under the corrected runner fingerprint. Supervisor1455390 advanced to joint training; workers1458054/GPU0 seed0 legacy,1458052/GPU1 seed0 adaptive,1458053/GPU3 seed1 gate-only were verified live. The remaining six matched training jobs and nine fixed320 endpoints stay queued. No trajectory or segmentation endpoint is complete yet.

## Completed fixed320 outcome (2026-10-09)

All nine B16/256-update training trajectories and B8/T1024/settle512 endpoints passed artifact validation. The supervisor is terminal and absent; all27 stages passed execution checks. Primary three-seed means (FG-ARI/foregroundIoU/objectIoU): legacy .011633/.013688/.007256; adaptive .038607/.048265/.006983; gate-only .009951/.011867/.007290; unchanged source97 .776348/.577070/.601537. Adaptive post-training activation and P-shuffle guards passed in all seeds, but performance promotion fails. Thus active events/nonzero gradients and merely positive reconstruction excess are insufficient evidence of useful object binding.

Secondary actual-spike QCC means: legacy_full 0.770964/0.496290/0.588318; adaptive_full -0.006426/0.016306/0.010476; gate_only_control 0.770834/0.483354/0.589608. These are separate readouts, not replacement primary scores. Legacy QCC retains much of source grouping while new primary assignments collapse; adaptive QCC also loses grouping. No coefficient/threshold/seed sweep or70k expansion follows this failed recipe. Read-only assignment/latent/decoder and horizon diagnosis is next. Source97 remains the incumbent, and all-gating usefulness/data scaling remain unproved. Full per-image endpoint evidence and source/Slot SHA bindings are persisted.

## Read-only assignment diagnostic completed (2026-10-09)

All nine first64 TRAIN diagnostics completed on exclusive inherited GPU leases; all three supervisors are terminal and absent. No GT or optimizer updates were used. Trained encoder gamma was used without requiring equality to the original source cache; the cache difference is descriptive only. Legacy TRAIN mean foreground coverage is .172913/.144958/.196716, but only about1 foreground group per image. Its fixed320 endpoint contains304/259/302 all-background images. Thus TRAIN already fails instance separation even where it reconstructs foreground, and the further TRAIN/VAL collapse needs same-image short/long comparison before attributing causality to temporal pooling. Adaptive seed0 is all-background in every one of64TRAIN and320VAL images; its other seeds are mostly binary grouping atTRAIN and their original QCC also collapsed. See assignment_diagnostic_summary_20261009.json and all nine raw diagnostics.

The failed recipe stays closed. The next bounded read-only check pairs these same first64legacy TRAIN images at T64/32 and T1024/512, comparing assignment/embedding diversity and actual temporal relations without changing checkpoints. It is not a horizon sweep or new training recipe. A successful future classifier must preserve relative temporal information and separate multiple instances; downstream gate usefulness and70k scaling still need independent evidence.
