# SW175 결과: critical-component 손실 후보 종료

두 seed1 모델 모두 64 updates를 완료했다. 동일한 고정320 이미지와 native32 계약으로 예측을 확정한 뒤 정답을 채점했으며, 완료 예측과 정답에서 세 지표·bootstrap을 독립 재계산했다. 아래 결과는 세 seed 평균이 아니다.

| 모델/출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0164 | 0.770932 | 0.614993 | 0.630523 |
| native_Q_conditional | 0.728783 | 0.614993 | 0.595189 |
| critical_component_conditional | 0.725044 | 0.614993 | 0.579214 |
| native_Q_standalone | 0.699792 | 0.448026 | 0.538209 |
| critical_component_standalone | 0.674915 | 0.399856 | 0.509791 |

conditional은 기존 Source-T foreground를 유지하면서 학습한 실제 Q/A로 물체를 연결한다. standalone은 학습한 spike trace의 native QCC로 전경도 생성한다. 두 경우 모두 후보가 기존 SW164 seed1과 대조군을 넘지 못했다. conditional 후보의 대조군 대비 FG 차이 95% CI는 [-0.012715, 0.005511], standalone은 [-0.039863, -0.010130]이다.

조건부 목표 두 가지를 모두 실패했으므로 seed0/2 또는 70,000장으로 확대하지 않고 이 손실 변경을 종료한다. 기존 최고 SW165는 유지한다. 손실을 바꾸는 것만으로 학습의 일반화나 spike foreground가 개선된다는 증거는 없다. 이 결과만으로 모든 gating이 불필요하다고 결론내리지는 않는다.

고정 TRAIN8 사후 진단은 학습 전후의 gate/spike·연결·partition 변화를 설명하기 위한 것이며 모델 선택이나 종료 판정을 뒤집는 데 사용하지 않는다. 다음 후보는 NEXT_HYPOTHESIS_KO.md에 기록한 전체 settled-window local phase credit이다. 작은 loss/threshold/epsilon 재조정은 이번 recipe의 구제책으로 진행하지 않는다.

검증된 최종 score SHA: `7311ee86f0ad30a3670edf8bf798f379f200fcd0372cd199e5b0af95fbe5fd02`. 원본 결과·독립 검증·terminal PID 관측은 `results_archive/completed_score_20261016/`에 보존했다.
