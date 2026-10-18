# SW0217 결과: 원래 에너지의 분할·재결합은 FG-ARI 개선 없음

64x64 patch, 같은 고정16장, 독립 seed0/1/2, 원래1024-step spike readout이다.
원래 전경과 모델을 고정하고 actual Q/A의 signed energy를 높이는 spectral
split과 인접 재결합을 적용했다. 학습0 update, fresh70k 학습 없음.

| 같은16장의 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 기준 모델 | .842891 | .600692 | .702161 |
| split-only | .842891 | .600692 | .702180 |
| split 후 재결합 | .842891 | .600692 | .702180 |

FG 차이0, paired image95% CI[0,0]. ObjectIoU 차이+.00001847로 확대 근거가
아니다. seed0/1/2 FG는 각각 .924401/.815856/.788414로 동일하다.
원래 에너지는 seed1/2 일부 이미지에서 증가했지만 FG가 개선되지 않았다.
이것은 모든 가능한 분할의 불가능성이나 global optimum의 증명이 아니다.
이 방향의 추가 eigenvector/recursive/threshold 소규모 탐색은 진행하지 않는다.

실제48개 Q/A 행에서 full-N energy를 재구성하고 optimizer를 다시 실행했다.
144 mask endpoint와 독립 세 지표/paired CI를 감사했고 최대 metric 차이는
2.22e-16이다. 실제 세 GPU worker와 큐는 정상 완료했다. 전경 정확 보존,
모든 그룹4-connected, source model SHA 불변과 energy nondecrease를 확인했다.

## 완료 후 GT를 이용한 에너지 정렬 진단

추가 core forward/update 없이 저장한 Q/A를 사용했다. 아래 pair AUC는 source
전경에 잡힌 GT 전경 patch 쌍에서 같은 물체를 더 높게 정렬하는 정도다.
잘못 합쳐진 쌍의 양수 에너지 비율은 해당 오류가 존재하는 이미지 평균이다.

| seed | Q pair AUC | energy pair AUC | 잘못 합쳐진 쌍의 양수 energy 비율 |
|---|---:|---:|---:|
| 0 | .9778 | .9737 | 99.58% (10장) |
| 1 | .9237 | .8859 | 98.18% (9장) |
| 2 | .9366 | .9138 | 89.59% (13장) |

반대로 잘못 쪼개진 같은 물체 쌍의 양수 energy 비율은3.35/14.22/4.85%다.
GT를 알고 각 기존 그룹을 GT별4-connected 조각으로 나누는 구체적 참고
행동은 모든 seed에서 평균 energy를 낮춘다. 참고 행동은 predictor, 최적해,
학습 가능한 상한이 아니다. Pair AUC도 최적 partition 성능의 증명이 아니다.

Q에 물체 구분 순위 정보는 남지만 현재 부호와 graph 가중 energy가 실제
잘못된 묶음을 지지하는 경우가 많다. 따라서 동일 energy optimizer만 바꾸는
시험을 종료한다. 다음은 실제 native Q/graph evidence와 연결 판정의 TRAIN-only
정렬을 검토한다. 206의4-rho MLP,199의core Bernoulli loss,214-216의component
merger/odds/target 시험은 이미 실패했으므로 그대로 반복하지 않는다.
DEV GT를 이용한 threshold 선택이나 fitting은 허용하지 않는다.

Full320의 세 seed 기준 .848615/.592533/.672047은 그대로다. 고정16장의
점수를 전체320, native64 신규70k, unsupervised, gating 인과 기여 또는 .90
목표 달성으로 표기하지 않는다. 모든 source/결과는 기존 patch_v2_sw에 공유한다.

근거: results_archive/completed_v1/score_v1/summary.json,
completion_audit.json 및 energy_diagnosis_v1/aggregate_summary.json.
