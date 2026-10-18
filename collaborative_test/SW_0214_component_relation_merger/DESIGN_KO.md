# SW0214: 기존 core component를 보존하는 관계 보정

211–213은 모든 patch의 instance assignment를 center head로 대체해 기존보다
나빴다. 기존 source205 seed2의 좋은 내부 partition은 보존하고, 같은 물체로
판단한 원래 component끼리만 합친다. 기존 component를 쪼개거나 전경을
삭제·추가하지 않는다. 이 방식은 기존 merge 오류를 고칠 수 없다는 한계도
명시한다. 기준은 동일 고정16/native64, source FG-ARI .788414다.

동일211 cache의 TRAIN32/DEV16/268-channel actual spike-rho/membrane 특징을
재사용한다. Core/encoder/gates 업데이트와 추가 core forward는0이다.
Component는 첫 occupied patch의 공간 순서로 정렬해 label ID 순서와 독립화한다.
TRAIN-only channel normalization 후 component 평균268과 위치·크기·채움률6을
계산한다. Pair 입력은 대칭 abs-difference/product: 특징536+geometry12=548.
RGB/GT count/DEV GT는 입력에 없다.

TRAIN target은 두 component의 정답 object histogram 내적이다. Histogram은
각 component의 전체 patch 수로 나누고 background0을 제외한다. Mixed
component는 soft target을 가지며 background끼리를 같은 물체로 학습하지
않는다. Image마다 전체 pair weight를1로 맞추고, positive/negative soft mass를
각각 normalize한 BCE를 사용한다.

MLP548→64 SiLU→1, 마지막 weight/bias0으로 시작한다. Strict logit>0만 merge
후보라 초기 readout은 기존 mask와 bitwise 같다. Adam1e-3, clip1,
full TRAIN-pair batch128 update 한 endpoint를 고정한다. Sparse center 검출이나
물체 개수 고정은 없다. 학습 및 Adam/RNG를 매 update 저장하고 재개한다.

Inference는 complete-link agglomeration이다. 모든 cross-component pair logit이
양수인 두 그룹만 합치며 그 최솟값이 가장 큰 후보부터 처리한다. Tie는 공간
canonical 순서로 결정한다. 따라서 약한 A–C 관계를 A–B–C chaining으로
우회하지 않는다. Validation threshold/count sweep은 없다.

Primary와 frozen geometry-only control을 먼저 고정한 뒤 GT를 읽는다.
Control은 첫536 입력을 TRAIN-pair mean으로 대체하고 geometry12는 유지한다.
이는 재학습한 물리적 gating ablation이 아니다. 모든 세 지표·count·merge 수와
paired CI를 보고하고, FG 우선 후보도 보존한다. 확대 gate는 FG+.01 이상,
paired95% CI lower>0, ObjectIoU 감소 없음이다. 실패 시 작은 계수 sweep 대신
component를 합치는 것만으로 충분한지 진단하고 방향을 판단한다.

Supervised readout pilot이며 새로운 전체320/세seed .90 또는70k 학습 결과로
부르지 않는다. 기존 전체평가 .848615 기준점은 따로 보존한다.
