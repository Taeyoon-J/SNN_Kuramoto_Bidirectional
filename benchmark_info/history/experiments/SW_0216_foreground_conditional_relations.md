# SW_0216_foreground_conditional_relations

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `ffc19f8608630d516ae63bb70374842c5dfff6f96b282a987bae909f4659a671`

````markdown
# SW0216: foreground-conditional TRAIN relation target

214 histogram은 정답 background를 포함한 component 전체 크기로 나누었다.
215의 TRAIN-only loss/decision 보정은 큰 하락을 완화했지만 baseline을 넘지
못했다. FG-ARI는 정답 foreground patch에서 계산하므로, 같은 물체의 두
component에 background가 섞였다는 이유로 relation target까지 작아지는
차이를 이번에 바꾼다. 평가 데이터/label rule/세 metric은 바꾸지 않는다.

Source205 seed2, 기존 cache/TRAIN32/DEV16/native64, 동일548 pair 특징,
동일MLP548→64→1/초기seed214002/Adam1e-3/full-batch128/clip1을 유지한다.
Feature table과 image-pair weight의 SHA가214와 정확히 같아야 한다.
오직 TRAIN histogram의 분모를 해당 component의 정답 foreground 수로
바꾼다. Foreground가 없는 component의 histogram은0으로 두며 배경끼리를
물체로 합치지 않는다. Mixed foreground는 soft target을 유지한다.

Positive/negative-normalized BCE와215의 TRAIN-only log(N/P) 보정을 사용한다.
새 target에서 P,N을 다시 계산하고 보정 후 strict logit>0 complete-link를
적용한다. DEV threshold/epoch/LR sweep 없음. Original component 내부와
예측 foreground는 유지한다. RGB/GT count/DEV GT를 inference에 넣지 않는다.

Primary와 frozen geometry-only control을 미리 고정한다. 기존215의 보정된
all-patch target endpoint를 matched 비교로 포함한다. 모든 prediction을
검증한 뒤 GT를 읽는다. 이후 두 GT-assisted component-merging 참고값도
계산하되 모델 점수/학습 가능한 상한/인증된 optimum이라고 부르지 않는다.

Primary FG+.01, paired95% CI lower>0, ObjectIoU 감소 없음이면 확대를 검토한다.
FG만 유용한 후보도 selection policy에 따라 보존한다. 실패하면 target/계수의
작은 sweep을 하지 않고 merge-only 제약과 표현 일반화 문제를 판단한다.
Core forward/update0, supervised pilot이며 전체320/세seed .90 또는새70k
학습을 완료했다고 주장하지 않는다. 기존 .848615 reference는 보존한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b6d47a47b78b05d4028b479e849e83bf3c6373f8387fff7361ce44e58859ad65`

````markdown
# SW0216 foreground-conditional relation supervision

Same source core/cache/TRAIN32/DEV16, same548 features/weights/MLP/seed/128 Adam
as214. Only TRAIN relation target is conditioned on true foreground. The215
TRAIN-derived odds correction is applied to the new target statistics. Evaluation
contract is unchanged. See DESIGN_KO.md.

Component interiors and source foreground remain fixed. Compare with frozen215
all-patch target endpoint. Posthoc GT-assisted merger references are explicitly
not predictions or certified ceilings. No new core forward or core/encoder update.
Five target tests plus inherited merger/alignment tests cover background dilution,
mixed objects, identity, complete-link/ID invariance, actual Adam resume and GT gate.
Registration and verified outcomes will be under results_archive.

Completed after17 local/17 Linux tests and actual6648-pair/native64 admission.
Primary .781651/.445474/.665117 versus original .788414/.445474/.674335:
no improvement. Matched215 FG unchanged. Exact input/weight/initialization match,
actual CUDA Adam128 and64 endpoints independently audited. GT-assisted merge
references .847321/.849997 are not predictors or certified upper bounds.
No further small target/threshold sweep. See RESULT_KO.md. Full320 reference stays.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `2512877c088ba2b94a00d326e4c353b8b67eea1d9cd50981f6c0835c26b4cd27`

````markdown
# SW0216: FG-conditional target 변경도 개선 실패

동일 source205 seed2/cache/TRAIN32/고정16/native64에서214와 feature table,
image-pair weight, MLP 초기화가 정확히 같음을 검증했다. TRAIN target의
분모만 정답 foreground 수로 바꾸고, 새 TRAIN 통계로 판정 보정을 계산했다.
Local17/Linux17 테스트와 실제6648-pair gradient/초기 readout 사전검증 후
GPU에서 Adam128을 완료했다.64개 mask endpoint와GT-assisted 참고값까지
독립 재검증했다. 데이터나 세 평가 metric의 정의는 바꾸지 않았다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| matched all-patch target + 보정215 | .781651 | .445474 | .664934 |
| **FG-conditional target + 보정216** | **.781651** | **.445474** | **.665117** |
| frozen geometry control | .788414 | .445474 | .674335 |

FG 차이는 기존 대비 -.006763, paired95% CI[-.018548,+.000049]다.
Matched215 대비 FG 차이는0이다. TRAIN positive prior는 .036833→.048753,
log-odds shift는3.263823→2.971002로 바뀌었으나 유효한 개선은 없었다.
Primary의 평균 merge는2개, 예측 group19.8125개, 정답 평균5.375개였다.

정답을 알고 같은 complete-link 규칙으로 합친 **참고 partition**은
all-patch target에서 FG .847321, FG-conditional target에서 .849997이었다.
이는 실제 모델 점수도, 인증된 최적값/학습 가능한 상한도 아니다. 다른
merger가 이보다 높을 가능성을 배제하지 않는다. 다만 이 예시와 실제 결과는
현재 target 분모 변경만 반복할 근거를 주지 않는다.

Merge-only 접근은 원래 한 component에 섞인 여러 물체를 나눌 수 없다.
또한 현 native solver의 각 그룹은 이미4-connected이므로 단순 connected
component 분리는 identity이며 새 테스트로 반복하지 않는다. 다음은 실제
spike Q와 coupling A/gain의 기존 signed energy에서 잘못된 merge를 되돌릴
수 있는지 확인하는 분할 보정 방향이다. 기존 energy와 source foreground를
유지하고 새 판정은 GT 없이 수행해야 한다.

현재 전체320·세 seed 기준점 .848615/.592533/.672047을 유지하며 .90 목표는
미달이다. 추가 core forward/encoder update0, supervised classifier pilot이다.
증빙: results_archive/completed_v1/score_v1/summary.json,
head_training/report.json, completion_audit.json 및 deployment_v1의17-test/admission
영수증. Feature/weight/initial-model equality와 실제 CUDA Adam128을 검증했다.

````

</details>
