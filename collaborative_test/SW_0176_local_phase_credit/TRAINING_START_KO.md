# SW176: fresh source에서 64-update 학습 시작

GPU1에서 실제 부모213909·worker214126과 CUDA 메모리 사용을 확인했다. 검증을 통과한 local-phase 정책을 학습에 적용한다. Kuramoto의 각 전이 입력 theta만 detach하며 full1024 구간을 계산하고 마지막512 구간의 실제 component spike로 기존 critical-component 손실을 계산한다. D/M/event/reset 연산과 forward는 유지한다.

seed1, 고유 TRAIN512장×2회, B16/micro4, fresh Adam64 updates다. SW175 critical-component 완료 모델은 재사용 대조군으로만 보관한다. 후보는 동일한 원본 source+zero-final context에서 새로 시작한다. 학습 목표에 TRAIN feasible target을 사용하는 supervised pilot이며, 70,000장 추가 학습이나 세 seed 평가가 아니다.

실제 control의 checkpoint·optimizer·history·boundary SHA를 별도 receipt에 묶었다. fresh 초기 state, base encoder, 전체512 ID 순서, optimizer의 parameter 이름·순서·LR·betas·기본 옵션이 일치한다. tuple/list 표현은 실제 Adam save/load 계약 검사로 정규화했다. 이미 학습한 control의 가중치나 Adam 상태를 후보에 로드하지 않는다. 같은 per-update seed, clip1, integration projection을 사용한다.

로컬·서버 각8개 CPU 검사를 통과했다. 서버에서 실제 full 모델과 새 Adam으로 public CPU factory를 실행한 뒤 boundary0을 저장하고 parameter를 변경한 후 복원해 초기 state와 빈 Adam 상태가 정확히 복원되는지 확인했다. 첫 두 committed updates에서는 실제 component spike와 context output/hidden layer에 학습 신호·parameter 변화가 생기는지 확인하며, 별도 disposable update를 추가하지 않는다.

64 updates 완료 후 같은 native32 고정320 이미지에서 conditional/standalone 예측을 확정하고 GT-last 채점한다. source SW164 seed1과 같은 loss의 SW175 tail64 대조군 대비 FG의 paired CI 및 두 IoU 조건을 모두 통과할 때만 seed0/2·더 큰 데이터로 확대한다. 최고 모델은 최종 검증 전까지 유지한다. 시작 및 실제 CPU 복원 증거는 `results_archive/training_started_20261016/`에 보존했다.

실제 boundary payload SHA와 등록된 update cursor를 검증하고, 첫 두 committed updates가 실제 component-spike gradient 및 context output/hidden layer의 credit 조건을 통과했음을 확인했다. 추가 disposable update는 없다. 원본 gradient·parameter 변화는 `results_archive/first_two_committed_updates_20261016/`에 보존했다. 이 검증은 segmentation 성능 향상을 입증한 결과가 아니다.
