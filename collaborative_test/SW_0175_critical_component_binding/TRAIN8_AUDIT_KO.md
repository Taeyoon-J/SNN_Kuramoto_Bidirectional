# 사전 등록된 TRAIN8 사후 진단 준비

`train8_audit.py`는 TRAIN512의 고정 위치 256..263에 대해 원본 Source-T+zero-context, 완료된 native_Q, 완료된 critical_component를 비교한다. 새 학습이나 checkpoint 선택에는 사용하지 않는다.

기록할 항목은 두 손실, conditional/standalone의 세 지표, 양성 route의 pair mass와 component rho 분포, 실제 Q 곱의 모든 component 편미분이 0인 pair mass, gate/spike의 평균 진폭과 event 비율, 원본 대비 label permutation에 불변인 partition ARI다. Q=0이라는 사실만으로 모든 gradient가 막혔다고 판정하지 않는다. 정답은 이미 등록된 TRAIN 캐시에 한정한다.

`train8_queue.py`는 수정된 score_queue_v3의 실제 완료 상태와 score SHA, 공개 score validator를 확인한 뒤에만 독점 GPU를 확보한다. 고정320 평가보다 먼저 GPU를 사용하지 않는다. 모델별 state SHA가 추론 전후 같아야 하며 추가 optimizer update는 0이다.

로컬 CPU 검사 12개가 통과했다. 성공 경로의 실제 supervisor 호출을 mock으로 끝까지 실행해 score에 묶인 worker command가 Popen까지 전달되는지 확인했고, launch 전 오류의 terminal 기록과 기존 상태 보존도 확인했다. **이는 서버 실행이나 실제 GPU 진단 완료의 증거가 아니다.**

현재 새 파일의 서버 전송 시도는 SSH banner timeout으로 실행되지 않았으며, 실제 서버 CPU 입력 준비 및 큐 등록은 아직 확인되지 않았다. 연결 복구 후 실제 완료 모델과 TRAIN 캐시로 CPU admission부터 검증하고 등록한다. 실행 중인 고정320 평가와 학습 산출물은 건드리지 않는다.
