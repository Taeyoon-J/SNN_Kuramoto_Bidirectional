# SW0183 완료: unique-data 이득은 확인되지 않음

고정 native32×32, 동일 320장, seed 0/1/2 평균입니다. TRAIN modal mask를
사용한 supervised 추가 학습이며, 70,000장 신규 학습 결과가 아닙니다.

| 모델/조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 512장 반복 control | 0.813732 | 0.577707 | 0.665824 |
| 1,024장 candidate | 0.810934 | 0.608932 | 0.661713 |
| 기존 supervised SW181 | 0.814836 | 0.589302 | 0.671681 |
| 기존 unsupervised SW165 | 0.804299 | 0.609975 | 0.651149 |
| Slot Attention, 우리70k 학습 | 0.781652 | 0.204360 | 0.203165 |

512/1,024는 동일 32-proposal prefix의 실제 model/Adam/RNG를 분기했고,
각 조건은 B16×64=1,024 proposal-image 노출로 맞췄습니다. 실제 accepted
updates는 control 35/39/40, candidate 31/37/34로 다릅니다. 따라서 동일
proposal/노출 budget 비교이며 동일 accepted-update 수 비교는 아닙니다.

1,024−512 FG-ARI는 −0.002798, paired-image bootstrap 95% CI
[−0.006642, +0.001133]입니다. 증가 이득이 입증되지 않았고, 통계적으로
확정된 성능 하락이라고도 단정하지 않습니다. Slot 대비는 +0.029282,
CI [+0.010226,+0.047724]이며 세 지표 모두 초과합니다. .87은 미달입니다.

SourceT foreground를 고정한 conditional FG는 0.802937→0.803142로 거의
변하지 않았습니다. standalone foreground IoU 증가는 주로 seed0에서
배경 오검출이 줄어든 것과 동반됩니다. 이것이 FG-ARI 개선으로 직결되지는
않았습니다. seed0 control FG 0.864375는 이번 실험의 단일 seed 최고이나,
세 seed 평균 최고 SW181을 교체하지 않습니다.

독립 검산은 여섯 실제 prediction archive와 각 conditional/standalone의
3,840 image endpoint를 사용했습니다. NumPy contingency ARI/IoU와 SciPy
Hungarian 결과의 원 점수 최대 차이는 3.34e−16 미만입니다. GT modal
변환, 모든 평균·paired bootstrap, 공통 prefix history/model/Adam/RNG와
첫 분기 transaction 동일성도 검증했습니다.

다음 방향: 같은 QCE를 더 반복하는 rescue sweep 대신, QCE가 학습하는
상대적인 class-affinity와 실제 QCC/foreground readout이 필요로 하는
absolute 연결 강도의 불일치를 TRAIN-only 고정 probe에서 먼저 측정합니다.
관측 근거를 얻은 뒤 readout에 맞는 loss를 등록합니다. validation에서
threshold를 검색하거나 기존 183번 소스를 수정하지 않습니다.

원본: [점수](results_archive/score_v1/evaluation.json),
[독립 검산](results_archive/score_v1/completion_audit.json),
[검산 소스](completion_audit_source.py).
