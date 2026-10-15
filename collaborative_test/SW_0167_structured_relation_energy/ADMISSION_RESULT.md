# Actual TRAIN/GPU admission, 2026-10-15

The frozen seed-1 source passed the actual eight-image TRAIN admission on an exclusively leased GPU1. Both the structured-partition objective and matched local-margin control completed two disposable Adam updates. This admits the fixed 256-update comparison; it is **not a performance result** and does not replace SW165 as incumbent.

All eight rows retained the original cached Q/A/gain and exact zero-residual baseline energy, merge log and output labels. Foreground support came from production CPU-B4 QCC on the original saved component spikes. GPU-B1 rho was combined with original cached context Q, with exact feature parity against the old wrapper checked separately on same-mode inputs. The two arms used identical features and identical initial head state (`489211c0dbb25db3729364642cd2e45590b7c5003d1436b23f11ed45238b0a98`). No VAL images or masks were read.

| Objective | Disposable loss, step 1 → 2 | Output-layer gradient, step 1 | Hidden-layer gradient, step 2 |
|---|---:|---:|---:|
| Structured partition | .277337 → .277266 | .072757 | .000101931 |
| Local margin | .548849 → .548685 | .132403 | .000236500 |

Gradient figures sum weight and bias norms; these differently normalized losses must not be compared as model scores. Actual output parameters changed on update1 and hidden parameters changed on update2 in both arms. Independent Sol5.6 review found no concrete math, gradient, device or provenance blocker in the final implementation.

The first admission attempt stopped during CPU target preparation: the existing mask loader returns a Tensor, while the caller used `torch.from_numpy`. No GPU child or training was started by that attempt. Its source and failure evidence remain preserved. The separate `preflight_v2.py` uses `torch.as_tensor`; before relaunch, all ten CPU contract checks and an actual eight-row TRAIN mask/support smoke check passed on the server. The v2 queue then completed successfully with supervisor PID3647147, child PID3647205 and report SHA256 `31757d610b799fef70a28a9096ca71f69cc49f5abf1627cfb4f27732e98e3217`.

Evidence: `results_archive/actual_train_admission_20261015/server_records.json` and `first_attempt_failure.json`. Training remains a separate next stage: same 512 TRAIN images, eight passes, 256 updates, fresh identical initialization, fixed learning rate and final-only evaluation. No hyperparameter or validation selection was performed during admission.
