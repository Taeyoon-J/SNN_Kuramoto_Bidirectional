# 사전 등록된 TRAIN8 사후 진단 준비

`train8_audit.py`는 TRAIN512의 고정 위치 256..263에 대해 원본 Source-T+zero-context, 완료된 native_Q, 완료된 critical_component를 비교한다. 새 학습이나 checkpoint 선택에는 사용하지 않는다.

기록할 항목은 두 손실, conditional/standalone의 세 지표, 양성 route의 pair mass와 component rho 분포, 실제 Q 곱의 모든 component 편미분이 0인 pair mass, gate/spike의 평균 진폭과 event 비율, 원본 대비 label permutation에 불변인 partition ARI다. Q=0이라는 사실만으로 모든 gradient가 막혔다고 판정하지 않는다. 정답은 이미 등록된 TRAIN 캐시에 한정한다.

`train8_queue.py`는 수정된 score_queue_v3의 실제 완료 상태와 score SHA, 공개 score validator를 확인한 뒤에만 독점 GPU를 확보한다. 고정320 평가보다 먼저 GPU를 사용하지 않는다. 모델별 state SHA가 추론 전후 같아야 하며 추가 optimizer update는 0이다.

로컬 CPU 검사 12개가 통과했다. 성공 경로의 실제 supervisor 호출을 mock으로 끝까지 실행해 score에 묶인 worker command가 Popen까지 전달되는지 확인했고, launch 전 오류의 terminal 기록과 기존 상태 보존도 확인했다. **이는 서버 실행이나 실제 GPU 진단 완료의 증거가 아니다.**

현재 새 파일의 서버 전송 시도는 SSH banner timeout으로 실행되지 않았으며, 실제 서버 CPU 입력 준비 및 큐 등록은 아직 확인되지 않았다. 연결 복구 후 실제 완료 모델과 TRAIN 캐시로 CPU admission부터 검증하고 등록한다. 실행 중인 고정320 평가와 학습 산출물은 건드리지 않는다.

## 후속 완료 (2026-10-16 UTC)

연결 복구 후 네 파일의 실제 배포와 서버 CPU 12개 검사, 실제 캐시 입력 준비를 확인했다. 고정320 채점이 완료되고 공개 validator로 재확인된 뒤 GPU1에서 TRAIN8 진단을 실행했다. 부모·worker 모두 정상 종료했고 immutable 보고서 SHA가 terminal 상태와 일치한다. 각 행에서 평균을 독립 재계산했다. 위의 서버 미실행 기록은 당시 상태이며 현재 완료 증거는 `results_archive/completed_train8_audit_20261016/`에 있다.

| 모델 | TRAIN8 conditional FG | TRAIN8 standalone FG | standalone foreground IoU | 양성 pair mass의 zero-partial 비율 |
|---|---:|---:|---:|---:|
| critical_component | 0.604887 | 0.606784 | 0.385742 | 0.113366 |
| native_Q | 0.561564 | 0.581960 | 0.564056 | 0.051811 |
| source_T_zero_context | 0.645323 | 0.643618 | 0.597822 | 0.044274 |

이 표는 학습에 사용된 8장의 설명용 진단이며 일반화 성능이 아니다. 최종 판정은 RESULT_KO.md의 고정320 음성 결과를 따른다. 두 loss와 event/rho 원자료는 함께 보존했으며, 이 결과만으로 gating의 인과적 필요성을 입증하지 않는다.

각 component의 active gate 조건부 spike event 비율을 8장 평균하면 원본·대조군·후보 모두 0.9998 이상이다. 따라서 이번 학습으로 membrane의 binary event가 새롭게 유용한 선택성을 얻었다는 증거도 없다. 실제 S는 연속 gate 진폭을 포함하므로 이 비율을 S의 평균 진폭과 혼동하지 않는다. 후보는 양성 route의 모든 component 편미분이 0인 pair mass 비율도 원본 0.0443에서 0.1134로 증가했다. 이는 해당 8장과 해당 route에서의 관측이며 전체 데이터나 모든 gate의 원인을 단정하는 수치는 아니다.
