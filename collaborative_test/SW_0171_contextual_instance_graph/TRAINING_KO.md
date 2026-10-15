# SW0171 matched prefix — 2026-10-15

동일한 original SourceT 초기 상태와 zero-final RF31 context로 Q-only와 Q+instance 두 조건의 학습을 시작했다. 실제 worker PID 3741306/GPU1 및 3741314/GPU2의 NVML 점유와 각 durable boundary 생성을 확인했다. 다른 사용자의 GPU0 작업은 건드리지 않았다.

두 조건 모두 첫 TRAIN64 RGB/정답 mask를 동일 순서로 두 번 학습한다. 각 조건은 logical B16/micro4의 8 Adam updates, 128 exposures다. Frozen base CNN과 preprocessing/T contrast는 유지하고 contextual encoder, graph와 native core, integration scalars를 학습한다. Q+instance만 같은 실제 graph projection의 leave-one-out class-balanced prototype CE를 추가하며, 사전 TRAIN-only calibration lambda=0.35786938902827736을 고정한다. 이 실험은 supervised TRAIN-mask 조건이다.

- 실제 full1024/settle512/live-tail64 rollout의 component spikes로 기존 production Q를 만든다. RGB-to-mask 우회 head는 없다.
- 첫 두 실제 업데이트에서 Q→S, graph/core 및 contextual output/hidden layers의 학습 credit을 검사한다. 실패하면 그대로 terminal scientific rejection으로 기록하며 추가 노출 없이 멈춘다.
- 학습 전후 고정 TRAIN positions256..263에서 actual-Q loss, instance loss와 own-QCC nonempty를 비교한다. 아직 VAL FG-ARI나 70,000장 개선을 입증한 결과는 아니다.
- Prefix가 등록된 비교 기준을 통과한 경우에만 나머지448 images 두 pass를 이어간다. incumbent는 아직 SW165다.
- 로컬·서버 CPU 검사 38개 및 실제 terminal admission→TRAIN cache 소비 경로가 통과했다. Adam/boundary 복원, initially-frozen source의 명시적 trainable-family 활성화, 중단 판정의 crash/resume 안전성을 확인했다.

현재 실행 증거는 `results_archive/started_prefix_20261015/actual_start.json`이며 완료 결과와 구분한다. Frozen 코드와 실행 근거를 공유하며, 데이터와 checkpoint 파일은 GitHub에 올리지 않는다.
