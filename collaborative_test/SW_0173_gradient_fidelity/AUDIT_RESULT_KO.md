# SW0173 — 전체 역전파만으로 학습 방향이 복구되지는 않았다

원래 SourceT seed1의 같은 native32 forward에서 tail64와 full1024 gradient를 비교했다. Cohort TRAIN4의 image-mean gradient로 각각 fresh Adam의 projected 가상 업데이트를 만들고, 고정 두 배율 1/16·1을 cohort4 및 별도 TRAIN4에 적용했다. 모델 원본은 복원했고 실제 학습 update는 0회다. 고정320 VAL 성능이나 seed 평균 실험이 아니다.

Worker 3783371/GPU1에서 172.849초에 정상 완료했다. 동일 RNG에서 두 horizon의 gamma, 실제 A, projection, Q, 여섯 trace가 모두 같았고 원본 반복 출력도 같았다. State hash와 source digest는 실행 전후 동일했다. 로컬·서버 CPU 테스트 각각 22개 및 실제 source의 CPU 가상 Adam/복원 검증을 통과한 코드다.

## gradient와 실제 변화

두 gradient의 전체 cosine은 **-0.896668**이다. Family별 full/tail L2는 context **752550.41 / 0.062098**, graph **503277.39 / 0.036891**, native core **89438.59 / 0.008220**이다. 전체 horizon은 매우 큰 sensitivity를 만들었지만 그 크기가 실제 finite-step 개선을 예측하지 못했다.

| 방향 | 배율 | cohort loss 변화 | 별도 TRAIN loss 변화 |
|---|---:|---:|---:|
| tail64 | 1/16 | +0.00069437 | -0.00001727 |
| tail64 | 1 | -0.00054224 | +0.00020728 |
| full1024 | 1/16 | +0.00002896 | -0.00083539 |
| full1024 | 1 | -0.00016094 | +0.00005762 |

Cohort source loss는 0.26514053, 별도 TRAIN source loss는 0.35634674다. Full gradient의 cohort 기반 1차 예측은 배율 1/16에서 -60.658, 배율 1에서 -970.528이었다. 별도 TRAIN에 같은 예측 수치를 표시한 것은 **cohort gradient 방향의 예측**이며 transfer 자체 gradient를 계산했다는 뜻이 아니다.

Full 방향은 두 배율 모두 transfer에서 tail보다 나았지만, small-step cohort 및 nominal-step transfer의 native loss가 증가했다. 따라서 사전 등록한 full-horizon pilot 필요 조건은 통과하지 못했다. Forest를 원본에 고정한 nominal-step transfer loss는 감소했으므로 route 변화도 결과에 관여할 수 있다. 두 배율의 실패가 모든 learning rate의 불가능성을 증명하지는 않는다. 추가 scale 선택이나 같은 eight-update 학습 확장은 하지 않는다.

## mask와 학습 신호

Source-F 조건의 cohort FG-ARI는 원본과 모든 trial에서 **0.777961**로 같았다. Transfer는 원본 **0.603791**, full small-step **0.603803**, full nominal-step **0.601712**였다. Native-QCC support까지 새로 계산하면 원본 **0.600381**, full small-step **0.602103**, nominal-step **0.599636**이었다. 소수 TRAIN 이미지의 작은 변동이며 최고 모델 갱신이나 안정적인 task 개선 근거로 쓰지 않는다.

일부 이미지의 positive forest loss gradient mass 중 약 **89.8–92.2%**가 모든 component partial이 0인 product-clamp edge에 걸려 있었다. 다른 이미지에서는 0이므로 dataset 전체 현상으로 일반화하지 않는다. 실제 active gate에서 event-on 비율은 이 여덟 이미지에서 약 **0.999888–1.0**이었다. Membrane/surrogate 또는 Kuramoto의 장기 sensitivity 중 어느 경로가 큰 gradient를 만들었는지는 아직 분리되지 않았다.

다음 변경은 gradient의 크기·nonzero 여부만으로 정하지 않는다. 원본 forward를 보존한 경로 분리로 원인을 구분하고, 필요 시 새로운 학습 개입을 등록해 실제 unseen TRAIN mask 개선을 확인한다. SW165 최고 기록과 fixed320 평가 계약은 유지한다.

원본 report SHA256: `d3163bf1ef8bf963ed18d72ccfdbc4dbac1655b47348bf1afb4115c89da01657`. 근거: `results_archive/completed_audit_20261015/{server_records,independent_verification}.json`. Worker와 supervisor 모두 종료됨을 확인했다. 데이터셋·checkpoint는 공유하지 않는다.
