# SW0181 — 실제 QCE loss를 낮춘 Adam proposal만 반영

SW0180 동일 TRAIN8 replay에서 seed 0·1은 QCE와 standalone FG가 개선됐지만
seed 2는 loss 0.526248→0.936206, FG 0.784225→0.636125로 악화됐습니다.
이번 가설은 실제 학습 목적을 악화시키는 optimizer proposal을 반영하지 않으면
세 seed의 학습 안정성과 최종 mask 성능이 개선되는지 확인하는 것입니다.
현재 minibatch descent가 다른 이미지의 loss나 FG 개선을 보장하지는 않습니다.

## 변경 범위와 고정 예산

세 seed 0/1/2 모두 자기 fresh SW0097 SourceT+zero-context에서 시작합니다.
공통 SW0171 TRAIN512 RGB/full modal GT, shared RandomState(171001)의 두
연속 permutation, logical B16/micro4, T1024/settle512/live-tail64,
product-Q LOO class-balanced CE(temp 0.1), 기존 네 optimizer family와 LR,
global clip 1 및 기존 parameter projection을 유지합니다. Factory/context/update
seed law는 SW0179와 동일합니다. Seed 1 초기 tensor는 SW0178의 f372…와 일치해야 합니다.
Core/gate/encoder 구조, mask readout, 배경 정책을 바꾸지 않습니다.

각 seed는 **64 proposals**만 처리합니다. 거절됐다고 추가 batch·LR·scale을
시도하지 않고 다음 등록 batch로 갑니다. 64 accepted updates가 될 때까지
늘리지 않습니다. Accepted/skipped, optimizer 실제 step, proposal 이미지 노출,
추가 no-grad 검사 연산량을 분리해 기록합니다. 이 실험은 supervised이며
native32/70,000장 신규 학습이나 unsupervised 성능으로 해석하지 않습니다.

## 반영 또는 복원

각 proposal에서 예정 RNG를 설정하고 gradient를 비운 뒤 모델 parameters/
buffers, Adam moments/step/groups와 CPU·CUDA·Python·NumPy RNG를 깊은 복사합니다.
기존 SW0179 gradient helper로 원래 Adam proposal을 정확히 한 번 만듭니다.
그 helper의 before loss는 gradient 선형 예측이 아닌 실제 전체 T1024 forward의
QCE 값입니다. Tail64는 backward 범위입니다. 처음 두 proposals에서는 동일
RNG의 별도 full no-grad before loss와 helper loss의 parity를 1e-6 절대+상대
오차 안에서 확인합니다. Parity 실패는 구현 검증 실패로 종료하며 loss를 바꾸지 않습니다.

매 proposal은 같은 batch와 시작 RNG로 full no-grad post loss를 계산합니다.
유한한 값이고 `post < before - max(1e-7, 1e-6*abs(before))`일 때만 반영합니다.
수치 guard는 사전 고정하며 sweep하지 않습니다. 반영하면 proposal의 모델·Adam을
유지하고 RNG를 gradient proposal 직후 상태로 되돌립니다. 거절하면 원래 모델·
Adam·시작 RNG를 완전히 복원합니다. 두 경우 모두 gradient를 비웁니다.
추가 no-grad 측정 자체가 모델 상태를 바꾸지 않았는지도 확인합니다.
비유한 post loss는 값을 null과 사유로 기록하고 복원하며 구현 예외를 숨기지 않습니다.

첫 두 proposals의 실제 S gradient와 네 family의 proposed gradient/delta를
검증합니다. Proposed 변경과 실제 committed 변경을 구분합니다. 거절된
proposal의 committed delta는 0이며, 이를 학습 성공이나 gate 기여로 해석하지 않습니다.
실제 Adam step은 accepted count와 같아야 합니다. Durable boundary는
proposal cursor와 accepted count를 모두 저장하며 rejected batch도 소비한 것으로
기록합니다. Resume은 둘을 섞거나 uncommitted gradient를 재사용하지 않습니다.

## 평가와 판정

완료된 SW0178/179 unconditional QCE의 같은 seed 초기 상태·데이터·loss·
batch/RNG 조건을 확인해 대조군으로 사용합니다. 새 후보의 모든 고정320
예측·protocol/hash를 먼저 확정한 뒤 평가 정답을 읽습니다. 실제 S→native Q→
production QCC→SW0165 standalone이 primary이며 conditional은 진단용입니다.
세 seed 평균 FG는 SW0165 0.8042989149474198와 matched Slot 0.7816519719609135를
엄격히 초과하고, 두 paired FG 95% CI lower가 0보다 커야 합니다. 두 IoU 평균도
Slot 0.20436012776655546 / 0.2031649861884501보다 엄격히 커야 합니다.
Bootstrap은 세 seed의 같은 이미지별 평균 후 320장을 10,000회 재표집합니다(seed 178).
FG 0.87 목표는 별도로 보고하며 실패를 숨기거나 사후 seed를 고르지 않습니다.
