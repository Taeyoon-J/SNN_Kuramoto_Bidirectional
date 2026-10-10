# SW0132 partition-relative full RGB reconstruction

This experiment replaces the weak absolute-XY patch-average RGB objective diagnosed in SW0131 with full128x128 RGB reconstruction in each predicted partition's relative coordinate frame. Source97, zero-initialized SW0130 state integration and native actual-spike QCC remain unchanged. Both matched arms receive the same new supervision.

Geometry and content pooling keep assignment credit; gamma content is detached. The decoder receives pooled8D content and relative2D coordinates, with no separate absoluteXY, center, scale or alpha head. All three source seeds must pass fixed TRAIN-only checks before the seed1 paired pilot.

## Outcome (2026-10-09)

The source0 preflight failed the registered hard-H row-scramble guard: mean full-RGB loss excess was `-9.807680180529132e-5` in both phase and constant arms, with only20/64 images showing positive excess. This is a scientific screen failure, not a runtime failure: source/live-gamma and both-horizon trace parity passed, all RGB gradient families were finite and positive, and each throwaway arm changed the expected core, encoder, decoder, and integration parameters. The decoder completed its32-step warmup and each arm took one disposable optimizer step; joint-training optimizer updates remained0. Ground truth was not used. Seeds1/2 were blocked, and there was no training, endpoint evaluation, score, or incumbent change. No retry or guard relaxation is registered.

Exact server report, queue state, and log bytes with SHA-256 verification are archived under [terminal preflight evidence](results_archive/terminal_preflight_failure_20261009/summary.json). The earlier [deployment launch record](results_archive/deployment_launch_20261009.json) documents startup and is retained as historical evidence. This result does not establish gating necessity or data-scaling success.

See protocol.json for exact prospective equations, budget, objective calibration, screens and claim limits. Sol6.1 reviewed the design and clarified live W pooling; implementation is assigned to Luna.
