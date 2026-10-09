# SW0123: adaptive temporal dynamics and a spike-derived assignment head

Prospective three-arm, three-seed experiment. The fixed dataset, modal16x16 patch GT and three patch metrics are unchanged. The primary classifier is new and consumes actual spike history; original QCC is secondary.

Compare legacy full dynamics, adaptive temporal full dynamics, and gate-only counterfactual, all with the same temporalCNN/11-slot assignment/nativeRGB spatial decoder. Adaptive dynamics use registered TRAIN-only calibration, .9 initial retention and adaptive binary event threshold. No raw RGB/gamma/theta classifier bypass. Detailed constants, schedule, failure/expansion gates and scientific limits are registered in design.md and protocol.json.

## Local implementation review (2026-10-09)

The opt-in dynamics module, classifier/decoder, and deterministic slot readout are implemented locally. Eleven focused CPU tests and compilation passed. A separate design/code review confirmed component fold b*4+d, binary-event adaptation, gate holds, weighted-spike reset/feedback and live gradients; the head/decoder has no direct RGB/gamma/theta input. This PASS covers modules only. Calibration/runner/checkpoint/real-data GPU preflight and nine-arm training/evaluation are pending. No SW0123 training, deployment or performance score exists yet.

The SW0109 single-seed older frozen-recipe scaling diagnostic is separate evidence; it does not prove this model scales. Positive masks alone will not establish every gating contribution: the registered follow-up includes frozen interventions and matched retrained ablations, followed by70k/scaling evaluation and untouched confirmation.
