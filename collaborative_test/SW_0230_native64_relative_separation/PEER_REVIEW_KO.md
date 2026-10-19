# 동료 연구 확인

origin/patch_v2 commit 5c88a6faa3c7f3a63f2d2223303c2a74ae1d45db,
PV2_0103_seed_rescue의 README/commands/metrics 변경을 확인했다.
동료의 contract-v2 native32 seed1에서 graph 고정 8epoch는
.6777/.6895/.6357, 비고정16epoch는 .6867/.6008/.4887이었다.
graph 고정은 두 IoU와 비용에서 유리하지만 FG는 .0090 낮다.
따라서 세 지표 모두 우월하다고 해석하지 않는다. seed2 고정 측정 전의
예상 평균도 실측으로 취급하지 않는다. 이 계약에는 Slot 대조군이 없다.

우리 쪽은 이미 graph를 고정하며 실제 세 seed와 세 지표를 함께 확인한다.
공유 graph/seed 안정성 및 단일 지표 선택 오류의 교훈을 유지한다.
동료의 다른 데이터·해상도 점수를 우리 최고 기록으로 섞지 않으며,
이번 상대 주파수 손실은 우리223/225/228/229 진단에서 도출한 별도 가설이다.
