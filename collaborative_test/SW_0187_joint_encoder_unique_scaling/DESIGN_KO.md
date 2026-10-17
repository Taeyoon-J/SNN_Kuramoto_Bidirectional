# SW0187 — 공동 encoder 조건의 데이터 증가 검증

186번은 고정 native32/320장 세 seed 평균 FG를 .814836에서 .834681로 높였다.
원본 encoder까지 학습하는 이 조건에서 고유 학습 데이터 증가를 검증한다.
기존186과183 소스·결과는 변경하지 않는다.

각 seed0/1/2는 같은 Source097 초기 tensor, zero RF31 context, 빈 Adam으로
시작한다. 원본 RGB encoder source_base LR3e-5를 포함한 다섯 family와
QCE temp.1, B16/micro4, T1024/settle512/live-tail64, clip1, actual full-forward
accept/rollback은186과 같다. 원본 CNN 외 frozen preprocessing은 유지한다.

첫512 이미지를 사용하는32 proposals를 한 번만 계산하고 model/Adam/RNG,
accepted counter를 두 arm으로 정확히 복제한다. control512는 기존512를
두 번째 permutation으로 반복한다. candidate1024는 같은 index+512의 새512를
사용한다. 각 arm의 논리 예산은64 proposals/1024 exposures, 실제 연산은
공통32+두 branch32=96 proposals/seed다. Accepted 수를 맞추는 retry는 없다.
같은 기존 TRAIN1024 cache를 재사용하고 첫512 RGB/GT/ID byte parity와
나머지512의 canonical TRAIN 소속을 검증한다. VAL/reserve는 학습하지 않는다.

실제 세-seed CPU factory에서186과 초기 model/gamma parity, 다섯 family의
중복 없는 optimizer, empty Adam, CNN gradient와 변경, frozen preprocessing을
확인한다. 공통 prefix의 첫 두 GPU proposals에서 emitted-S 및 다섯 family의
positive gradient/proposed delta와 full-forward parity를 확인한다. 각 branch
최종 checkpoint의 실제 CNN SHA 변경과 모든 Adam step도 검증한다.

두 arm×세 seed의 conditional/standalone fixed320 예측과 protocol을 모두
확정한 뒤에만 GT를 채점한다. Primary는 standalone이다. 고유 데이터 이득은
candidate FG mean>동시대control이며 paired-image bootstrap95% CI 하한>0일
때만 인정한다(RandomState178,10000회, 같은 이미지의 세 seed 평균 차이).
Slot70k 대비 세 지표 엄격 초과 및 FG CI, 기존186/181 대비 차이와 FG.87도
별도로 보고한다. Foreground IoU 저하를 숨기지 않는다. 독립 검산은 실제
candidate3840 endpoint-image 행과 Slot960 행, modal GT, means/CI/gate 및
동일 prefix/분기 RNG/Adam, 실제 CNN 학습을 확인한다.

과거186 prefix 해시 비교는 진단 기록이며 실행 허용 조건이 아니다. 같은
현재 prefix에서 분기하는 비교가 인과 비교의 근거다. 실패 시 이 recipe의
작은 LR/온도 rescue sweep은 하지 않는다. 성공해도 supervised pilot이며
fresh70k, unsupervised 개선, gate 인과 기여나 ordering 최적성을 주장하지 않는다.
원래70k/.87/논문 기전 목표는 남는다.

유휴 GPU만 독점 사용하고 외부 작업이 시작되면 우리 worker만 양보한다.
관측 timeout으로 작업을 재시작하지 않는다. CPU/실제 CLI/cache/fork와
평가·검산 admission을 먼저 통과시킨 뒤 배포·큐 등록한다.
