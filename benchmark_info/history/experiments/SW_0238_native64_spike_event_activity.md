# SW_0238_native64_spike_event_activity

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `16486e73435f11d703857c2ba5352964e0b1686b7b7392f8f3836c83daff29da`

````markdown
# SW0238: 현재 native64 spike event 활동 진단

235번에서 검증한 실제 SW0232 누적512회 모델을 그대로 사용한다. 각 seed의
기존 TRAIN 순서 첫2장만 T1024로 추론해 마지막512 step의 actual component
spikes/gates를 관찰한다. 진단의 입력은 RGB와 실제 trace이며 라벨·성능 평가는
사용하지 않는다. 기존 checkpoint 복원에서 과거 TRAIN artifact를 검증할 수 있지만
이 진단의 정답 입력으로 쓰지 않는다. 추가 학습·mask 변경은 없다.
과거 native32 all-on event 관측을 현재 native64에 그대로 적용하지 않는다.

S=event*gate에서 gate>0일 때만 event occupancy와 transition을 관측한다.
gate=0에서는 event를 추정하지 않는다. 실제 S와 gate의 bitwise equality 및
component-shared gate 여부를 기록한다. 이6개 TRAIN trace의 관찰은 성능 결과나
counterfactual membrane/gate 제거의 인과 증명이 아니다. 후속 loss 선택의 진단이다.

독립 actual checkpoint CPU admission 뒤 기존 독점 GPU 큐를 사용한다. 외부 사용자가
접근하면 우리 worker만 양보하고 완료한 이미지부터 재개한다. 기존237번은 변경하지
않으며70,000장 실험 범위 확인에 의존하는 학습/추론도 시작하지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `c40afd16e17aa31dc95beb5ee01c24ae64d08e12f58d54acb492ab47df335e54`

````markdown
# SW0238: actual native64 event / gate diagnostic

Two registered TRAIN images per fixed accepted232 seed. Six actual forwards,
no GT, no optimizer updates, no new model or performance claim. Pending real
CPU admissions and GPU queues; do not confuse preparation with execution.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `a1c991b9d750a1dc66ea4fd788fcc7660ed61a617b2631651b667d494b620909`

````markdown
# SW0238: completed six-case native64 event diagnostic

All three fixed accepted232 seed checkpoints were checked on two actual TRAIN images each. Source, checkpoint and row bindings and reported integer-count arithmetic were validated. No evaluation labels or optimizer updates were used.

Positive-gate event occupancy was 0.996797/0.997802 for seed0, 1.0/1.0 for seed1 and 0.993265/0.993510 for seed2. On these two seed1 cases, the final spike tensor was bitwise equal to the gate. In the other four cases, observable event suppression was at most 0.006735. All observed cases used identical gates across components.

This supports investigating event saturation: for these traces the membrane/event operation mostly passes the gate. It does not prove that membrane removal preserves segmentation, that all inputs are saturated, or that a threshold change improves performance. Raw traces were not retained, so this archive validates scalar consistency, not independent trace recomputation. A next causal test should compare the actual mask outputs under an explicit event bypass on the same frozen model and input; any subsequent learned intervention needs held-out metric confirmation.

The ongoing70k training is unchanged. This diagnostic produces no new segmentation score and does not replace the audited SW0227 full320 incumbent. Private dataset and sample metadata are omitted from this public aggregate report.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `44cbb447e5f7f95a3a4421900be28c37acfb5a5699a741928a0b06572732b698`

````markdown
# SW0238 실제 복원 검증 및 큐 등록

로컬7검사, Linux7검사, CLI2개를 통과했다. 세 실제 SW0232 checkpoint를
CPU에서 복원하고 원래235번 accepted 모델과의 binding을 검증한 뒤 seed0/1/2
독점 GPU 큐를 등록했다. 추가 optimizer update는0이다.

첫 관측에서는 세 supervisor가 실제 살아 있었으나 모두 빈 GPU를 기다리고 있었다.
따라서 이 기록은 큐 등록 증거이며 실제6개 forward 완료나 진단 결과가 아니다.
`START_EVIDENCE.json`에 배포 SHA와 실제 관측을 보존한다.

기존237번 full320 평가는 변경하지 않는다. 70,000장 작업은 별도 범위 확인을
기다리고 있으며 이 진단으로 대체하지 않는다. 결과가 나와도 특정 TRAIN6개 trace의
관측으로 제한하며 전체 데이터 성능이나 gate 제거의 인과 효과를 주장하지 않는다.

````

</details>
