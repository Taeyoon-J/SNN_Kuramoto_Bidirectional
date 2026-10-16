# SW0182 — 같은 proposal 예산에서 unique TRAIN512 → TRAIN1024

SW0181은 fixed320 native32 세 seed 평균 0.814836 / 0.589302 / 0.671681로
등록 primary gate를 통과했습니다. 정답 mask를 사용한 TRAIN512 추가 학습이며
unsupervised 또는 native32 fresh70k 성과로 해석하지 않습니다. 이 성공 recipe와
checkpoint는 변경하지 않습니다.

이번 가설은 동일한 1,024 proposal 이미지 노출에서 반복 이미지를 새 TRAIN
이미지로 대체하면 실제 mask 성능이 더 좋아지는지입니다. Architecture, loss,
LR, acceptance threshold, foreground/readout 정책을 탐색하지 않습니다.

## 데이터와 일정

SW0171의 기존 ordered512를 정확한 prefix로 보존하고, 같은 seed1 SourceT
training contract의 전체 ordered TRAIN IDs에서 바로 다음 512개를 추가합니다.
총 1,024개의 고유 TRAIN ID여야 하며 VAL1320..1639 및 reserve90640..90959는
포함하지 않습니다. 기존512 RGB와 full modal GT는 frozen cache와 byte-exact
이어야 합니다. 추가512도 같은 canonical TRAIN RGB cache·ID mapping·mask
preprocessing을 사용하며, gamma/S cache나 feasible projected GT를 만들지 않습니다.

SW0179/181의 shared RandomState(171001) 두 연속512 permutations를 그대로
사용합니다. 첫32 proposals는 기존과 같은 indices 및 images, 뒤32는 원래
둘째 pass의 index i를 new512의 i, 즉 cache index 512+i로 매핑합니다.
따라서 대조군의 512×2와 후보의 (기존512+새512)×1을 비교하며 총64 proposals,
B16/micro4, 1,024 proposal image exposures는 같습니다.

## 초기 상태·학습·비용

각 seed 0/1/2는 SW0181과 동일한 fresh SW0097 SourceT+zero-context 초기 상태와
빈 Adam에서 시작합니다. SW0181의 학습 완료 가중치를 이어받지 않습니다.
처음32 proposals는 source/RNG/batch/목적이 같으므로 SW0181 control의 대응
before/post loss, accept/reject 및 model 경계가 동일해야 합니다. 수치 차이는
사전에 고정한 guard보다 작아야 하며 과학적 실패를 기술적 수리로 구제하지 않습니다.

T1024/settle512/live-tail64 backward, product-Q LOO class-balanced CE(temp0.1),
네 optimizer family와 LR·clip·projection, finite strict actual-descent acceptance,
동일 batch/start-RNG post 검사 및 model/Adam/RNG 복원은 SW0181과 같습니다.
64 proposals 후 종료하며 accepted step을 맞추려고 batch나 retry를 추가하지 않습니다.
Accepted/rejected, 실제 Adam step, target 노출 및 추가 full-forward 검사 비용을
분리해 기록합니다. 모든 변경은 새 실험 폴더에 두며 frozen181 코드는 바꾸지 않습니다.

## 평가와 중단 기준

세 seed의 fixed320 예측·protocol을 모두 freeze한 뒤 GT를 읽습니다. 실제 S→Q→
production QCC→SW0165 standalone이 primary, conditional은 진단용입니다.
SW0181 대조군 원본 SHA는
`3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54`입니다.

Unique-data benefit은 FG 평균이 SW0181보다 엄격히 크고, 이미지별 세 seed 평균
차이의 paired bootstrap 95% lower가 0보다 클 때만 인정합니다(320 images,
RandomState178, 10,000 resamples). 기존 SW0165/Slot FG 평균·paired CI gate와
Slot 두 IoU strict gate도 유지하며 FG0.87은 별도로 보고합니다.

성공하지 못하면 이 1024 recipe를 닫고181을 보존합니다. 실패를 더 많은 데이터
전체의 무용성으로 해석하거나 LR/온도 rescue sweep을 하지 않습니다. 이 비교는
pretrained source에서의 supervised unique-data 효과이며 fresh native70k scaling
증거가 아닙니다. Gate 필요성·학습 기여는 다음 별도 ablation cycle에서 확인합니다.
