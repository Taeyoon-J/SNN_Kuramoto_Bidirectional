# SW0235: 고정512회 모델의 secondary64 확인

232번 고정16장 FG .884622는 128회 대비 평균 +.024988이나 CI에0이 포함된다.
이 원래 gate=false는 보존한다. 추가 학습·loss 조정 없이 실제512회 모델을
기존 first64 validation(IDs1320–1383)에서 확인한다. 원본16장은 그대로
재사용하고 seed마다48장만 새로 추론한다. offset0 mask와 frequency를
bitwise 재현해야 한다. GPU0–3의 독점 idle lease와 기존 checkpoint를 유지한다.

비교군은 동일64장의 원래 기준과 실제128회 SW0223/SW0225다. 세 비교군의
protocol과 actual512 checkpoint를 모두 확인한 뒤 GT를 읽는다.
FG gate는 원래 기준 대비+.01 이상 및 paired image CI 하한>0,
그리고128회 모델의 FG 평균 초과다. balanced gate는 두 비교군 각각에
대해 두 IoU 평균도 낮아지지 않아야 한다. CI RNG235,10000 draws다.

기존16장 gate를 바꾸거나 secondary64를 untouched test로 부르지 않는다.
전체320장 최고는 SW0227 .858916으로 유지한다. 이번 결과만으로 FG .90
목표나70,000장 신규 학습을 주장하지 않는다. 모델 선택이나 threshold sweep은 없다.
