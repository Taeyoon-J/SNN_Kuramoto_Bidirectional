# SW_0211_spike_anchor_center_assignment

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `ef8dac18263eded15a2962f5c1a773ac78f99d233e720415c6d141cf0ed93e1e`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `4e981d5fc5fd029902092d44803d11560a667a919fea3acde049bb058bfef624`

````markdown
# SW0211 spike-anchor center assignment

Registered after17 local and17 Linux tests, AST/CLI checks and actual source/
TRAIN32/native64 head-and-decoder CPU admission. Supervisor2269832 is waiting
for an exclusive GPU0..3 lease. No improvement is claimed at registration.
See DESIGN_KO.md. Core stays frozen and the spike foreground is preserved;
only the new center-voting readout learns from TRAIN32 masks.

Tests cover real production correlation descriptor layout/time invariance,
disconnected-instance voting, exact foreground preservation, GT-free decoder,
actual Adam resume, changed-data rejection and GT-last scoring. A separate
synthetic TRAIN-only budget smoke passed; its results are not research-model
performance. Actual cache extraction and head training begin only after a lease.

The actual source/model seed is2, verified from the immutable admission and
checkpoint SHA. The generic dispatcher receipt's seed0 label is not passed to
the worker and is explained in results_archive/registration_provenance_v1.json;
original records are preserved and no job was restarted to correct that label.

Read-only waiter2272841 tracks the exact supervisor PID/start identity,
observes actual GPU start and independently audits completed head/Adam/RNG
boundaries, TRAIN normalization, frozen source and96 mask endpoints. This is
a supervised pilot, not a fresh70k or full320/three-seed .90 result.

Actual GPU3 worker2337998/start448637534 began after210 seed2 completed all
four prediction shards. NVIDIA ownership and actual argv/start identity are
recorded in started_v1/actual_GPU_start.json. This starts frozen-core feature
extraction; it does not yet establish that classifier Adam updates have begun.
The earlier GPU2 attempt yielded before extraction to respect another owner.
SW0212 is independently queued to compare learned foreground after completion.

Completed, gate failed: primary128 .034678/.445474/.044414 versus original
.788414/.445474/.674335. All DEV images use a single-center fallback. Actual128
Adam updates and96 endpoints independently audited. TRAIN-only diagnosis also
finds no effective center detections, despite offset error1.5185 pixels versus
zero-offset4.2434 and mean-input4.7278. These are TRAIN diagnostics, not gains.
See RESULT_KO.md and results_archive/head_diagnosis_v1/summary.json. Incumbent
full320 three-seed mean remains .848615/.592533/.672047.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `ded9c34960fdbd73abad8429a7fdf72a0335363790530cdb76bc0d4987082cd0`

````markdown
# SW0211: center head pilot 실패, 기존 기준점 유지

고정16·seed2·native64 결과다. 실제 head Adam128, source checkpoint 불변,
TRAIN-only normalization,32/64/128 snapshot과96개 mask endpoint를 독립
검증했다. Core/encoder/gates를 이 실험에서 학습하지 않았다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 warm205 | .788414 | .445474 | .674335 |
| center head32 | .034678 | .445474 | .044414 |
| center head64 | .034678 | .445474 | .044414 |
| primary128 | .034678 | .445474 | .044414 |

모든DEV 이미지에서 center가 검출되지 않아 단일 center fallback을 사용했다.
평균 정답 물체5.375개에 예측1개로 collapse했다. 전경은 원래 mask와 정확히
같으므로 Foreground IoU가 유지된 것이 grouping 성공을 의미하지 않는다.
Mean-input/moments-only 비교도 대부분 같은 fallback이라 최종 mask 점수로
입력 신호의 유용성이나 gating 기여를 판단할 수 없다. 확대 gate는 실패했다.

별도 CPU 진단에서 TRAIN32도 모두 fallback이었다. 정답 center가 고정
확률 기준 .3에 도달한 비율은0이고, 정답 center 확률 평균은 .06354였다.
즉 validation 성능만 나쁜 경우가 아니라 학습 데이터에서 center 검출기
자체가 아직 작동하지 않은 상태다. Head128의 full TRAIN loss는9.7335에서
4.2920으로 줄었고 모델·Adam은 실제 업데이트됐다.

한편 object-balanced offset 오차는 무변위4.2434 patch 대비 실제 입력에서
1.5185였다. 같은 frozen head에 TRAIN-mean 입력을 넣으면4.7278,
moments-only 입력은3.6219였다. 이 값은 TRAIN 진단이며 validation 성능이나
전체 물리적 gating의 인과 기여가 아니다. Offset 학습은 있지만 sparse
center 검출이 endpoint를 막는다는 후속 검증 근거로 사용한다.

211의 고정128 판정을 바꾸거나 threshold/LR/width를 sweep하지 않는다.
후속 실험에서는 동일 head/Adam을 이어 학습하는 단일 예산 확장을 별도로
등록해 TRAIN center 검출이 살아나는지와 DEV 성능이 함께 개선되는지 본다.
추가 core forward가 필요 없는 cached-feature 학습이다. 실패가 계속되면
학습량을 반복해서 늘리지 않고 readout 방향을 전환한다.

기존 전체320·세 seed 기준점 .848615/.592533/.672047은 그대로 유지한다.
증빙: completed_v1/completion_audit.json, score_v1/summary.json,
head_training/report.json 및 head_diagnosis_v1/summary.json.
마지막 진단의 source_model_sha256은 frozen classifier head SHA를 가리킨다.

````

</details>
