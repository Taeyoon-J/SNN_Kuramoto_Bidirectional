# SW_0204_native64_component_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `8e7c038e44c33b6bb0651257deaba4ad4181819a74b86586852161f8f88b4df3`

````markdown
# SW0204 native64 per-component learning credit

Hypothesis: positive-product Q=product_d ReLU(rho_d) gates off all pair learning
when one component correlation is negative. QCE on signed component correlations
can provide credit before that hard product without bypassing emitted spikes.
Objective = original class-balanced QCE(Q) + mean of the four class-balanced
QCE(rho_d), using the same temperature .1. No new loss weight or target margin.
Unlike SW0175 this uses all class-balanced query/prototype comparisons, not
minimax edges of a GT route forest. Unlike SW0199 it acts before the product,
not on the calibrated final Q. This is supervised mask training.

Only objective changes versus SW0202 finite-Adam control. Same actual SW0192
TRAIN8 model/Adam/RNG, two B4 proposals on indices8..15, live64, original graph,
encoder, gating structure and production inference. Finite uphill steps retained,
nonfinite proposals rolled back. Same sixteen development IDs, GT after masks.
Require paired FG improvement >=.01 with bootstrap lower bound >0 against202,
nonnegative FG versus warm, nonnegative object IoU against both, applied update
and changed CNN. Mini success is not full320/three-seed .90 or causal proof of
gating usefulness. No LR/weight micro-rescue after a failed screen.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `8bf111a8b191ed51062e31303a50a2d66d6133d2fdb02bed88e5588482bc26e3`

````markdown
Native64 original QCE plus signed-component QCE. Ready for preflight; no performance claim. See DESIGN_KO.md.

2026-10-18:18 local and18 Linux tests, AST checks, two Linux CLI checks
and actual CPU model/Adam/RNG admission passed. All four GPUs are eligible.
Supervisor2049352 registered; no actual GPU start or improved score claimed yet.
Control is the completed finite-Adam202 candidate, not the guarded199 control.
The legacy report completion status mentions finite_Adam because the step policy
is retained; experiment bindings and fingerprint specify the new component loss.

Actual GPU1 worker2052928 started after203 finished. NVIDIA PID, initial
model, Adam moments and CPU/CUDA RNG restoration independently verified.
Start evidence is under results_archive/started_v1. No evaluation result yet.

Completed, independently audited: fixed16 development images, seed0.

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| component credit | .9005389599 | .8605773316 | .7276216788 |
| matched finite-Adam202 control | .9001477304 | .8595864826 | .7281376470 |
| frozen TRAIN8 warm | .9244014538 | .8598964167 | .7276384095 |

Two new finite steps applied, actual Adam step4, CNN/model changed. Actual48
masks, checkpoint, Adam/history and inherited model/RNG were independently
verified. FG difference versus202 +.0003912294, CI[-.0009170428,.0018393708];
object IoU slightly lower. Warm FG difference -.0238624940. Extension gate
failed; recipe closed. No loss-weight/LR micro-rescue. This does not establish
that component learning is generally useless or prove gate contribution.
No full320/three-seed .90 result. Best full-evaluation incumbent is unchanged.

Follow-up clarification: SW0208 has now evaluated this exact checkpoint on
full320: .876242/.790093/.693262, versus warm192 .877709/.791224/.696284.
The preceding "No full320" sentence describes the original204 mini admission,
not the later208 confirmation. No full320 .90 or fresh native64 TRAIN70k claim
is established. See SCALE_DECISION_KO.md for the expansion decision.

````

</details>

## SCALE_DECISION_KO.md

<details>
<summary>SCALE_DECISION_KO.md</summary>

Source SHA256: `fea09e06b9c1392a589371c4981aa1f6b02d9e042fd87c7d407d031f17f5eb3f`

````markdown
# 204번의 .90과 70,000장 확대 판단

204 seed0는 고정 16장 평가에서 FG-ARI .900539를 기록했다. 이 결과를
무시한 것이 아니라, SW0208에서 동일 checkpoint의 평가를 전체 320장으로
확대했다. 학습을 추가하지 않은 확인 평가다.

| 동일 checkpoint 평가 범위 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 204 고정16 | .900539 | .860577 | .727622 |
| 204 전체320 | .876242 | .790093 | .693262 |
| 기존 warm192 전체320 | .877709 | .791224 | .696284 |

고정16 수치는 원본204 summary 및 README, 전체320 수치는 SW0208의
검증된 summary에 근거한다.
204와 warm192의 전체320 paired FG 차이는 -.001467,
95% CI[-.006263,+.003277]이다. 현재는 개선이 확인되지 않았다.

16장 .90은 실제 관측이지만 전체 평가 .90이나 독립 세 seed 평균 .90의
증거는 아니다. 평가 이미지 수가 달라진 비교이므로 데이터 증가 학습이
성능을 떨어뜨렸다는 증거로 해석하지 않는다.

70,000장 학습과 320장 평가는 다른 작업이다. 204는 기존70k 학습 lineage를
이어받고 native64에서 추가 TRAIN16으로 적응한 supervised 모델이며,
native64 TRAIN70,000 재학습을 완료한 결과가 아니다.

따라서 현재204 그대로의 새70k 학습은 등록하지 않는다. 개선 후보는 같은
mini의 기존 모델과 비교하고, 확대 평가 및 독립 seed로 효과를 확인한 뒤
큰 학습 예산을 검토한다. 이는 모든 지표가 매번 동시에 개선되어야 한다는
규칙이 아니다. SELECTION_POLICY의 FG 우선 및 mask 품질 후보를 함께
보존한다. 전체320도 반복 사용한 validation이며 untouched test는 아니다.

근거: ../SW_0208_native64_full320_confirmation/RESULT_KO.md 및 해당
results_archive/completed_v1/score_v1/summary.json.

````

</details>
