# SW0179 — QCE seed 재현성 검증 결과

완료·독립 검산된 고정 32×32 patch / 동일 320장 결과입니다.
이 supervised QCE 추가 학습은 세 seed 재현에 실패했습니다. 기존 최고
unsupervised SW0165를 유지하며, 이 설정 그대로 70,000장으로 확대하지 않습니다.

| Standalone 결과 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| QCE seed 0 | 0.849988 | 0.642330 | 0.692904 |
| QCE seed 1 — 기존 SW0178 | 0.818029 | 0.437500 | 0.686364 |
| QCE seed 2 | 0.616842 | 0.328093 | 0.374701 |
| **QCE 세 seed 평균** | **0.761620** | **0.469308** | **0.584656** |
| 기존 SW0165 세 seed 평균 | 0.804299 | 0.609975 | 0.651149 |
| 비교 가능한 Slot Attention 세 seed 평균 | 0.781652 | 0.204360 | 0.203165 |

QCE 평균 FG-ARI는 SW0165 대비 -0.042679이며, paired 95% CI는
[-0.052731, -0.033229]입니다. Slot 대비 -0.020032,
95% CI [-0.040078, -0.000144]입니다. 같은 이미지의 세 seed 점수를 먼저
평균하고, 공유 320개 image index를 10,000회 재표집했습니다(RandomState 178).
두 IoU는 Slot보다 높지만, 등록한 primary gate와 FG-ARI 0.87 목표는 실패했습니다.

Conditional SourceT foreground를 그대로 유지해도 평균은
0.767848 / 0.609975 / 0.598112입니다. Conditional FG-ARI는
seed 0/1/2가 각각 0.845145 / 0.777384 / 0.681013입니다.
따라서 seed 2의 문제를 standalone 배경 판정 하나로 설명할 수는 없습니다.
학습 불안정성, loss와 최종 mask 분류의 불일치, 일반화 중 어느 원인인지
이 평가만으로 확정하지 않습니다.

각 seed는 자기 SW0097 source checkpoint에서 시작합니다. seed 0·2는
새 optimizer로 공통 TRAIN512장을 두 pass(1,024 exposures, 64 updates)
추가 학습했고, seed 1은 변경하지 않은 SW0178 product-QCE 결과입니다.
TRAIN 정답 modal mask를 사용한 supervised 실험이며 native32/70,000장
신규 학습이나 unsupervised 개선으로 해석하지 않습니다. seed 0·1의 유용한
checkpoint는 탐색 결과로 보존하며 실패한 세-seed 후보를 승격하지 않습니다.

실제 학습·예측·점수 계산은 모두 완료됐습니다. 최종 모델과 boundary tensor,
Adam step, 두 pass의 데이터 노출 및 첫 두 update의 spike gradient와 네
optimizer 그룹 변경을 검증했습니다. 모든 후보 예측을 확정한 뒤 fixed320
정답만 읽었고, reserve data는 읽지 않았습니다. 별도 검산은 각 seed·endpoint
점수, 기준 파일, bootstrap, gate와 전후 artifact hash를 재확인했습니다.

최종 점수 SHA256: `758e48f83b44dc31efeae73154c13dbcd8ffa6132a85c7055279a0f4b2649c3d`.
원본 점수와 독립 검산, 예측 protocol·모델 불변 기록은
`results_archive/score_v1/`에 있습니다.
