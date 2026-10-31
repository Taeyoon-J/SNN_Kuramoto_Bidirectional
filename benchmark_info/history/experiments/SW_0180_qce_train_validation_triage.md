# SW_0180_qce_train_validation_triage

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `0ef61e9a0dbbf865d18f1fcb7369557045028676442ecc648a3e9077aa51428e`

````markdown
# SW0180 — 같은 TRAIN8의 학습 전후 paired replay

SW0179 QCE의 고정320 FG-ARI는 seed 0/1/2가
0.849988 / 0.818029 / 0.616842였습니다. Conditional SourceT foreground를
유지한 seed 2도 0.681013이므로 배경 정책 하나로 실패를 설명하지 않습니다.
기존 SW0165를 유지하고, 실패한 QCE 설정의 70,000장 확대·seed 2 단독
재학습·온도/LR/배경 정책 sweep을 하지 않습니다.

이번에는 공통 SW0171 TRAIN512의 고정 positions 256..263, 총 8장을 씁니다.
각 seed 0/1/2에 대해 fresh same-seed SourceT+zero-context와 학습 완료 QCE를
같은 RGB·정답·B4·RNG에서 비교합니다. 두 checkpoint 상태 × 8장 × 3 seed,
총 48 image forwards이며 T1024 / settle512 / live-tail0, no-grad·no-optimizer입니다.
추가 학습이나 validation/reserve 이미지·정답 읽기는 하지 않습니다.

Fresh 모델은 SW0179의 seed별 초기화 factory를 사용합니다. Seed 1 fresh
tensor SHA는 기존 SW0178 초기 상태와 일치해야 합니다. Final seed 0·2는
SW0179, final seed 1은 변경하지 않은 SW0178 product-QCE checkpoint입니다.
같은 seed의 실제 모델을 strict-load하고 CPU admission 후 GPU를 초기화합니다.
데이터의 seed 1 producer provenance와 각 모델의 source seed를 분리합니다.
Seed 0·2에서 seed 1 cached spike/support를 대신 사용하지 않습니다.

모든 mask는 실제 emitted component spikes → native positive product Q →
production QCC → 기존 SW0165 classifier 경로로 만듭니다. Fresh SourceT의
TRAIN8 QCC foreground를 conditional support로 보존하고 final Q/A와 trained
graph gain으로 분류합니다. Standalone은 각 모델 자신의 QCC support를 씁니다.
Validation에서 저장된 foreground는 TRAIN 입력으로 쓰지 않습니다.

이미지별로 등록된 product-QCE loss(temp 0.1, full modal GT/BG0, LOO,
class-balanced mean), true-class minus best-other-class margin, 두 readout의
FG-ARI/Foreground IoU/Object IoU, foreground precision/recall을 기록합니다.
표본/클래스 집계와 singleton의 유효성은 기존 QCE loss 정의에 맞춥니다.
Source/final hash, 공통 RGB/정답/IDs hash, RNG, GPU, model/state 불변과
optimizer update 0을 기록합니다. 다른 사용자 GPU 작업은 건드리지 않습니다.

결과를 본 뒤 다음 방향을 고르는 작은 진단이며 원인 확정이나 성능 승격이
아닙니다. 같은 이미지의 raw CE·margin·mask 변화와 기존 고정320 결과를
함께 보고 다음 세 가설을 구분합니다.

- TRAIN에서도 CE 상승·mask FG 하락: optimization/representation instability 쪽 증거.
- CE 하락에도 TRAIN mask FG 하락: loss와 QCC/최종 readout 불일치 쪽 증거.
- CE 하락·TRAIN mask FG 상승에도 고정320 하락: 작은 학습 표본에 대한 적합/일반화 쪽 증거.

혼합되거나 변화가 작으면 불확실하다고 기록합니다. 8장으로 가설을 입증하지
않으며 nonzero gradient만으로 gating의 유용성을 주장하지 않습니다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `da5bd0ae92e7c6ce2428c6fe2dd3b2656e51c6cc0ddb12b39c223ad3bbfd773c`

````markdown
# SW0180 — 동일 TRAIN8 학습 전후 진단 완료

세 seed 모두 같은 학습 이미지 8장에서 원래 모델과 QCE 학습 완료 모델을
비교했습니다. 추가 학습·optimizer update·validation/reserve 읽기는 없습니다.

| Seed | QCE loss 전 → 후 | Standalone FG-ARI 전 → 후 | Conditional FG-ARI 전 → 후 |
|---|---:|---:|---:|
| 0 | 0.411989 → 0.378687 | 0.822280 → 0.847263 | 0.822280 → 0.815183 |
| 1 | 0.619215 → 0.476528 | 0.644924 → 0.727048 | 0.644924 → 0.652200 |
| 2 | 0.526248 → 0.936206 | 0.784225 → 0.636125 | 0.784225 → 0.674977 |

Seed 2는 같은 TRAIN 이미지에서도 목적 loss와 mask 성능이 함께 악화됐고,
true-class margin도 4.579420 → 2.538839로 낮아졌습니다. 이 작은 cohort는
현재 학습 설정의 optimization/representation instability 쪽 증거를 제공합니다.
일반화 문제만으로 설명하거나 standalone 배경 정책만 고쳐 해결된다고
단정할 근거는 부족합니다. 어떤 optimizer family가 원인인지는 아직 모릅니다.

훈련 history의 두 pass 평균 loss도 seed 2는 0.473019 → 0.537881로 상승하고,
seed 0·1은 하락했습니다. 이는 다른 순서·변하는 모델의 online loss이므로
그 자체를 학습 발산의 증명으로 쓰지 않습니다. 같은 이미지의 paired replay가
주된 근거이며, 8장의 결과를 전체 TRAIN512/70,000장으로 일반화하지 않습니다.

등록한 [DESIGN_KO.md](DESIGN_KO.md)에 따라 실제 S → native Q → QCC →
SW0165 분류를 사용했습니다. Conditional은 각 seed의 fresh TRAIN8 QCC
foreground를 유지하고, standalone은 각 모델 자신의 QCC를 사용합니다.
같은 RNG·B4·T1024/settle512에서 총 48 image forwards를 수행했습니다.
모든 모델의 전후 tensor hash는 동일했고 10개 CPU 테스트를 통과했습니다.

`results_archive/result_v1/`에는 세 원본 report·완료 queue·history 요약과
입력/모델 hash·집계 검사가 있습니다. 검사는 저장 보고서의 provenance와
산술을 재확인했으며 mask 추론을 독립적으로 다시 실행하지는 않았습니다.
새로운 고정320 성능 실험이나 최고 모델 갱신은 아닙니다. SW0165의
0.804299 / 0.609975 / 0.651149를 기존 최고 unsupervised 기록으로 유지합니다.

````

</details>
