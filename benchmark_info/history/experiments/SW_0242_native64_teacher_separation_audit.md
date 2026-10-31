# SW_0242_native64_teacher_separation_audit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `def5f66933fd8b9085384524a5c752dbf6155b66cd2bc5ec4450d1027eca7860`

````markdown
# SW0242: frozen frequency teacher separation audit

The actual SW0239/240 loss is already class-balanced per image, including background. It denoises effective frequencies towards the original217 teacher's object means and anchors gamma. It contains no explicit repulsion between different objects. A controlled native64 counterexample checks that the exact registered loss can be zero with two different objects sharing identical frequency targets. This is a structural observation, not evidence that actual data regressions have this cause.

Measure actual original217 teacher outputs on 128 distinct TRAIN images sampled without replacement with RandomState242000 from the registered70k split, reusing the same sample across seeds0/1/2. Use the validated canonical input cache. Restore the actual original teacher with the existing239 source-teacher CPU path. Shared restoration can load validation RGB and archived predictions; only sampled TRAIN RGB/masks enter these statistics. No GPU, optimizer update or evaluation ground truth is used. Masks are supervised TRAIN inputs, not model predictions.

For each image report foreground-object-center nearest distances, ratios to pooled within-object frequency spread, and nearest-teacher-center patch/object-balanced TRAIN accuracy. Aggregate distributions are descriptive teacher diagnostics, not FG-ARI, IoU, held-out accuracy or proof that achievable trained masks are bounded by these numbers. Empty/single-object cases are excluded from multi-object accuracy distributions. Zero within-object scale is counted separately.

Retain actual frequency vectors and TRAIN labels privately and reproduce all statistics from those arrays. Record frozen teacher/model/cache/source bindings and verify no state changes. Three seeds are sequential CPU processes; existing GPU training and evaluation continue. This diagnostic does not alter the ongoing70k training or reopen the failed228/230 margin hypotheses.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `07bfa782b986ccb0c744a4063d002103fd9289afc86e8986d4191252fcb4f573`

````markdown
# SW0242 completed: frozen teacher separation diagnostic

The same128 sampled TRAIN images were analyzed with each original217 teacher used by the current239/240 loss: 128 unique images, 384 seed/image cases. All cases had multiple foreground objects. Source-bound retained frequency vectors and masks were re-read and all descriptive statistics reproduced. Four local and four Linux tests passed. No GPU or optimizer update was used.

The pooled median nearest foreground-object center distance was0.145956 in actual four-component frequency units. Dividing each nearest-center distance by the square root of the two objects' summed within-object frequency variances gave a median ratio0.646069 (quartiles0.341708 and1.123532). Thus many sampled original-teacher objects overlap in their within-object frequency spread. This is a descriptive property of the frozen teacher, not a measurement of the currently trained student's final masks.

The existing loss is already object-balanced, including background. It pulls patches towards the frozen teacher's per-object frequency mean and anchors gamma; it has no explicit requirement that different object means remain separated. A controlled native64 input to the exact loss gave zero loss with two foreground objects sharing frequency targets. That mathematical counterexample does not establish that those gamma/frequency tensors are realizable by the actual fixed core.

These findings do not prove why more data changed prior performance, impose a performance ceiling, or supply FG-ARI/IoU scores. The nearest-teacher-center TRAIN accuracies in the JSON use ground-truth object centers and are diagnostic quantities. The denoising objective itself is intended to reduce the measured within-object spread, so original-teacher overlap alone does not prove that the objective will fail after training.

SW0228 already failed after strengthening absolute source-center preservation. SW0230 already failed after adding source-relative separation. This result does not justify reopening either coefficient sweep. Keep the requested70k training unchanged, await full320 SW0237 confirmation, and use SW0241's actual event-readout intervention before selecting a new gating or objective change. SW0227 remains the audited full320 incumbent.

````

</details>
