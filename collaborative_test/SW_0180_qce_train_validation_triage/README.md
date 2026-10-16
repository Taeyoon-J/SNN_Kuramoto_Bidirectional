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
