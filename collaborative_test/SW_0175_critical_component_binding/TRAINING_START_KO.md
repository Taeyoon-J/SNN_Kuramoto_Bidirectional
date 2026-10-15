# SW0175 비교 학습 시작

V1의 boolean-support 변환 누락을 새 V2 source에서 수정했다. 모델·loss 수식·
TRAIN IDs·학습 순서는 유지하며 실패 소스와 update0 checkpoint를 보존했다.

로컬과 서버에서 15개 CPU 검사를 통과했다. 추가로 실제 TRAIN cache의
8개 multiclass support와 feasible target을 새 caller에 전달해, 명시적 `labels > 0`
결과와 두 loss가 일치하고 gradient가 유한함을 확인했다. 이 검사의 Q/rho는 합성 입력으로,
성능 평가나 실제 source rollout 결과가 아니다. 검증 코드를 결과 폴더에 함께 보존한다.

두 arm은 동일 원본 SourceT와 zero-final context 초기 상태에서 시작했다.
원래 native-Q loss와 critical-component positive loss를 각각 512 TRAIN images,
2 passes, B16/micro4, 총64 updates로 비교한다. 실제 GPU 1/3 worker가 확인됐고,
두 arm의 첫 committed update도 확인했다. cache 우회 환경은 실제 worker에서 확인했다.
첫2 updates의 credit 검사는 학습에 포함하며 추가 GPU updates를 수행하지 않는다.

고정320 평가가 완료되기 전까지 성능 향상이나 새 최고 모델을 주장하지 않는다.
Supervised seed1 pilot이며 70k 신규 학습 또는 세 seed 결과가 아니다.
현재 최고는 SW165의 0.8043 / 0.6100 / 0.6511로 유지한다.

실행·검증 증거: `results_archive/started_v2_20261015/`.
