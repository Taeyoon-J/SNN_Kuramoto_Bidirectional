# SW0140: 완료된 native32 causal endpoint 검토

2026-10-10, Sol6.1. `causal_endpoint_analysis.json`, `causal_evaluation.json`, prediction protocols 및 실제 `causal_rollout.py`를 검토했다. 고정 source97 all3/320, native1024 patches, full1024/settle512, actual-spike QCC 결과이다. 학습·queue·incumbent를 변경하지 않는다. SW0139 cross_view는 아직 진행 중이다.

## 현재 frozen 모델에서 확인된 기여

| Frozen 조건 | 평균 FG-ARI | Source 대비 FG 차이 | 해석 |
| --- | ---: | ---: | --- |
| Source | .78191867 | — | 현재 incumbent |
| Kuramoto interaction off | .05607437 | −.72584430, CI [−.741371,−.709412] | 현재 결합 상호작용에 강한 의존 |
| Delayed gate=.5 | .00000000 | −.78191867 | 해당 sinusoidal gate 경로의 강한 의존 |
| Dendrite h-history off | .77164959 | −.01026909, CI [−.012928,−.007722] | h-history가 현재 grouping에 기여 |
| Membrane voltage-history off | .78222498 | +.00030630, CI [−.000702,+.001454] | FG 이득 입증 없음 |
| Spike reset off | .78188260 | −.00003608, CI [−.000584,+.000452] | 실질적 FG 이득 입증 없음 |
| Binary events forced on | .78188260 | 위와 동일 | 실질적 FG 이득 입증 없음 |

D-history 제거는 foreground IoU도 −.00800344, object IoU도 −.01635863이며 두 CI 모두 음수다. 짧은 TRAIN trace에서 seed1 효과가 작았어도 실제320 masks에는 평균 손실이 나타났다. 작은 raw trace 차이가 CC에 영향을 줄 수 있으므로 TRAIN effect size로 조건을 걸러서는 안 된다는 점도 확인된다.

M-history 제거의 object IoU는 오히려 +.00145341, CI [+.000465,+.002587]이다. Reset/event 제거는 foreground IoU가 +.00034369, CI [+.000015,+.000741]이고 FG/object CI는 0을 포함한다. 이 작은 이득을 새로운 incumbent나 GT-selected ablation recipe로 채택하지 않는다. 현 결과는 모든 downstream 연산의 성능 기여를 긍정하는 근거가 아니다.

## Intervention의 정확한 범위

K-off는 coupling interaction만 0으로 두고 native drive/다른 경로를 유지한다. Gate=.5는 carrier=`sin(theta)*.5`, integration gate, spike emission과 exact-zero hold 조건을 함께 바꾼다. 따라서 **현재 delayed sinusoidal gate 전체 경로**의 유용성 근거이며, delay 길이만의 효과나 각 carrier/integration/emission 역할의 독립 증명은 아니다.

D-history off는 이전 h 항만 제거하면서 현재 dense 입력 gain `(1−sigmoid(tau_n))`을 유지한다. 이전 emitted spike의 dense 입력은 남는다. M-history off도 이전 voltage 항만 제거하고 현재 입력 gain/reset/hold를 유지한다. D-history 결과는 해당 recurrence의 frozen 기여이지, dendritic layer 전체를 제거한 뒤 재학습해도 회복할 수 없다는 뜻이 아니다. Source의 추가 a_D/a_M scalars는 0이므로 그 새 integration 조절의 기여를 이 실험으로 주장할 수 없다.

Forced E=1은 readout factor만 바꾸지 않고 다음 dendrite/reset에 전달되는 emitted S도 바꾼다. Reset-off는 reset 항만 삭제한다. 세 seed 각각 두 조건의 prediction NPZ SHA가 **완전히 동일**하고, 320장의 세 metric per-image arrays도 동일하다. 따라서 저장된 labels/IDs 수준에서 동일한 예측 artifact라는 근거가 있다. Raw membrane/E/full trajectory의 동일성은 검증되지 않았으며 서로 다른 state가 같은 settled labels를 만들 수 있다.

Forced-event 조건의 E occupancy=1은 intervention 정의다. 그것만으로 원본 전체 validation E가 100%였다고 추론하지 않는다. 원본 첫4 TRAIN/full1024 occupancy `.99901056/.99992502/.99685615`는 높은 포화의 별도 관찰이고, settled512/all320 occupancy와 동일한 통계가 아니다. MG Gaussian surrogate의 backward credit는 포화 binary forward에서도 존재할 수 있다. Nonzero gradient, 파라미터 변화, event 결정 변화, grouping 이득은 각각 다른 증거다.

## 다음 학습 우선순위는 여전히 조건부

현재 가장 정확한 연구 문장은 **이 frozen native32 predictor는 K coupling, delayed sinusoidal route와 dendritic h-history로 이득을 얻지만 membrane voltage history/reset/binary selector의 추가 grouping 이득은 확인되지 않았다**이다. Frozen intervention은 현재 사용 경로의 효과이며 retrained layer necessity나 모든 gate의 보편적 필수성을 입증하지 않는다. Image bootstrap CI는 고정 세 seed에 조건부이고 seed 모집단 주장이 아니다.

1. 먼저 진행 중인 cross_view를 완료하고 source 및 matched old-loss control과 비교한다. Source를 보존하며 FG 개선을 보이면 `event_selection_review.md`의 task event-credit 대 terminal event-detach pair를 다음 후보로 검토한다. 같은 native forward/recurrence와 실제 S=E×g를 보존하고 b/tau_m만 추가 학습 가능하게 한다. GT-free fixed TRAIN에서 실제 E decision 변화와 candidate/control credit 차이를 확인하며 event-rate target을 만들지 않는다. 학습된 candidate의 closed-loop event-forcing mask 악화까지 있어야 새로운 event 의존 근거가 된다.
2. Cross_view도 source를 훼손하면, 실패한 global-prior objective에 selector capacity를 바로 더하지 않는다. `SW0139/source_preservation_design.md`의 C-only 대 C+soft source actual-Q anchor가 먼저 검토할 목적함수 변경이다. Binary pseudo-instance teacher 실패를 재개하지 않으며 source fidelity를 object truth나 gating 해결로 부르지 않는다. 이 역시 신규 등록 전 후보이고 자동 실행 계획이 아니다.

어느 조건에서도 completed context/causal recipe의 threshold·LR·rate를 재조정하지 않는다. 향후 gate 학습의 이득은 실제 masks와 matched training 대조로 확인해야 한다. 현재 source의 Slot32 FG 평균 차이는 +.00026670에 불과하고 두 seed는 Slot에 진다. 최종 all3 comparable Slot 세 지표 초과와 unique-data 증가에 따른 개선 목표는 유지한다. Source70k lineage continuation/현재 frozen ablation은 native32 training 또는 fresh scaling 증거가 아니며, 유망한 all3 학습 recipe가 확보되기 전 full70k 확장을 약속하지 않는다. Reserve90640–90959는 읽지 않는다.
