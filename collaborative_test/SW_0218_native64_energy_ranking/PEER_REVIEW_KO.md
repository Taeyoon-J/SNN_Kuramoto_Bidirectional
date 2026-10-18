# 이번 사이클의 동료 연구 확인

origin/patch_v2를2026-10-18에 다시 fetch했고, 이전a0a39df에서
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 갱신된 것을 확인했다.
해당 commit의 PV2_0102_grid32_seeds/README.md, metrics.csv와101 정정을 읽었다.

동료는8 epoch/native32 seed0 .7699/.7531/.7017이라는 결과가 세 seed에서
재현되지 않아 기존 headline을 철회했다. w0의 FG는 .7699/.2916/.7068,
평균 .5894, 표본 표준편차 .2599다. group-count w5도 .7803/.2118/.7617,
평균 .5846, 표본 표준편차 .3230으로 안정화되지 않았다. 비교 대상 .7239는
동료 자신의 published16 설정이며 Slot Attention 수치가 아니다.

동료는 foreground 과다 예측과 단일-group 동기화 실패를 구분했고, collapse
통계 하나로 seed 실패를 예측할 수 있다는 해석도 철회했다. 다음 graph-freeze
시험은 제안되어 있지만 이 commit에 완료 결과는 없다. 해당 contract의
고정 graph를 native64의 확정 개선으로 가져오지 않는다.

적용한 원칙은 동일 seed/source/학습 예산 비교, 세 seed 전체 평가 유지,
proxy 하나가 아닌 실제 mask/foreground/instance 오류 확인이다. 218은
완료된204와 같은 warm192 및 실제 Adam/RNG에서 시작하며 새 loss만 바꾼다.
실행 중 fingerprint를 바꾸거나 동료의 count term/8 epoch를 그대로 추가하지 않는다.
현재 우리의 full320 .848615와 동료300장/다른 계약의 .5894를 직접 순위화하지 않는다.

추가 주의: 동료의 seed0 8-vs40 epoch 비교는 +.1406의 근거가 있지만,
seed2 .7068을 seed0의40 epoch .6293과 비교한 사실만으로 seed-independent
overtraining이 증명되지는 않는다. 데이터 scaling과 epoch budget은 같은
seed, 데이터, readout을 맞춘 비교로 계속 구분한다.

근거: https://github.com/Taeyoon-J/SNN_Kuramoto_Bidirectional/tree/7a21632b93805f4f1da69a2e77a9ddd95c624d0a/collaborative_test/PV2_0102_grid32_seeds
