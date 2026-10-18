# SW0211: 전경을 유지하는 spike-anchor center assignment

209의 작은 loss 효과가 불확실하여 같은 계수 rescue를 중단했다. 207은
전경 제거가 holes/splits를 늘렸고,161/163의 raw/Q/A slot binder도 기존보다
나빴다. 별도의 center-voting classifier로 grouping을 학습한다.

실제205 warm seed2 checkpoint(dff381eb...53d0)를 고정한다. 210 prepare_cpu(2)
factory로 실제 전체 encoder/core를 복원한다. 원본197/B2를 쓰지 않는다.
동일16에서 실제FG .788414, GT-pure 전경고정 reference .976234, GT전경
누락3.17%였다는 posthoc 진단이 근거다. GT reference는 모델 성능이 아니다.

- TRAIN: 기존 registered64의 첫32 canonical128 RGB/2px modal mask.
  IDs1000..1639 제외. 모델 초기학습8장은 이32 안에 포함된다.
- DEV: 고정16 offsets0,2,..30, IDs1320..1350. DEV GT는 학습·cache에 없다.
- Core1024/settle512, gamma/graph/gating/encoder와 weights는 모두 고정.
- 입력268 channels: actual signed native rho의 spatial8×8 anchor columns64개
  ×4 components +spike rates4 +membrane mean4/std4. Anchors4,12,..60.
  RGB/gamma/GT count를 classifier 입력으로 사용하지 않는다.
- TRAIN inference RNG211000+index, DEV193000+offset. DEV baseline mask16개는
  기존205와 정확히 같아야 한다. float32 cache와 TRAIN-only mean/std.
- Head:268→32 conv1 +GN8/SiLU, conv3 dilation1/2/4/8 +GN8/SiLU,
  foreground1/centerHeat1/normalizedOffset2. 약46k parameters.
- Loss: object-class-balanced foreground BCE(보조 loss), visible-GT centroid에
  가장 가까운 같은-object patch의 Gaussian center heat(sigma1.5) focal loss,
  object-balanced SmoothL1 offset(beta1 patch). 세 계수 모두1.
- Inference: **원래 spike foreground를 반드시 보존**한다. FG head는 보조학습뿐,
  threshold로 patch를 삭제하지 않는다. Heat>=.3/localmax3×3/NMS radius1,
  최대10 centers(고정 capacity, GT count 아님). Center가 없으면 FG안 maxheat
  한 점을 쓴다. Patch offset vote를 가장 가까운 predicted center에 배정한다.
  공간적으로 떨어진 visible parts도 같은 object로 배정할 수 있다.
- TRAIN32/B4/Adam3e-4/clip1, primary128updates=16passes.32/64/128 snapshot을
  고정하고 DEV 예측을 모두 만든 뒤 GT를 읽는다. Primary128은 미리 고정하며
  이전 checkpoint가 좋아도 silent best-selection을 하지 않는다. Peer101의
  예산 결과를 반영해 불필요한128pass 학습 대신 budget curve도 기록한다.
- Primary128에는 TRAIN mean으로 전 입력을 대체한 mean_input과 rho256채널만
  TRAIN mean으로 대체한 moments_only frozen input ablation도 등록한다.
  동일 head와 원래 spike 전경을 사용한다. 이들은 재학습한 gating ablation이
  아니므로 전체 gating의 인과적 기여를 입증했다고 부르지 않는다.
- 매 head update마다 model/Adam/RNG boundary 저장. 행별 cache/prediction
  checkpoint와 shared exclusive GPU0..3 lease; 외부 사용자에게 양보한다.

Primary FG delta>=.01, paired95%CI lower>0, ObjectIoU delta>=0,
foreground bitwise identical, 실제 head Adam128과 baseline replay를 확인해야
확대를 검토한다. 실패하면 threshold/LR/width 미세 sweep을 하지 않는다.
GT가 있는 supervised classifier pilot이며 fresh70k, data scaling, 모든 gating의
인과 기여, full320 또는 세 seed .90의 증거라고 부르지 않는다.
