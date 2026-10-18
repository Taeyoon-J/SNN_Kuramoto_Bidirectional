# SW0201 normalized spike-Q group-count regularization

동료1fc3906의95/96은 여러 readout 조정 후에도 과분할을 보고했고 그룹 개수 loss를
다음 방향으로 제안했다. 원래 N²/||PLV||²는 그룹 크기가 같을 때 개수와 같으며,
우리 TRAIN64의 이상적인 정답 block에서도 실제 class 평균7.11에 대해1.58이다.
고정 raw PR 목표7.5를 그대로 옮기지 않는다. CPU 수학검토/테스트는197에 있다.

가설: 현재 class-balanced QCE에 크기를 보정한 spike-Q count proxy를 추가하면
과분할이 줄어들 수 있다. 단일 변경은 가중치1의 squared log-count penalty다.
production Q를 대칭화한 복사본의 diagonal만1로 두고 W, degree D에 대해
B=D^-1/2 W D^-1/2, P=tr(B)²/||B||²를 사용한다. QCE + (log P-log K)²,
K는 해당 TRAIN image의 GT class 수(background 포함)다. 정답 mask supervised
학습이며 추론에는 K/GT를 제공하지 않는다. normalization은 이상적인 불균형
block의 개수를 보정하지만 실제 connected count나 count classifier는 보장하지
않는다. proxy가 내려가도 실제 mask가 개선되지 않으면 성공으로 처리하지 않는다.
positive-product Q의 완전한 zero/dead correlation에는 gradient 제한이 남는다.

192의 실제 seed0 TRAIN8 model/Adam/RNG step2를 그대로 이어 indices8..15에서
B4 두 proposal만 추가한다. 총16 unique/logical proposal 노출(inherited8+new8),
기존5개 LR,clip1,T1024,settle512,live-tail64, 원본 geodesic, 원본 gating/CNN/readout.
200의 live512 변경을 섞지 않는다. 같은 출발점·데이터·64 credit의199 실제 QCE
control과193 warm model을 모두 비교한다. control의 두 proposal은 거절됐으며
실제 model/16 masks가 warm과 같은 것으로 검증됐다. 역사적194 prefix 대신 이
고정된 control을 사용한다. fixed allocator fraction.8을 적용한다.

같은 development offsets0,2,...30의16장과 RNG193000+offset. 모든 예측 protocol을
먼저 고정하고 canonical2×2 modal GT로 세 metric을 독립 검증한다. paired image
bootstrap10000(seed199). 확장 gate: candidate-control FG gain>=.01, CI lower>0,
candidate-warm FG mean>=0, 양쪽 objectIoU mean>=0, 실제 새 accepted update>=1 및
model/CNN committed change. mask 양성 label 수·정답 object 수·raw count exact도
집계하되 trained count classifier나 주 metric의 대체물로 부르지 않는다.
유망한 seed0 mini만 이후 seed1/2에서 재현한다. full320/three-seed.90은 별도 목표다.
실패하면 count weight/고정 group prior의 작은 sweep으로 연장하지 않는다.
