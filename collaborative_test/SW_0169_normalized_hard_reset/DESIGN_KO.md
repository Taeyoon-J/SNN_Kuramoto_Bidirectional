# SW0169 — 정규화된 slow membrane와 binary hard reset

2026-10-15, prospective DESIGN only. 구현·서버 실행 없음. **Supervised TRAIN-mask pilot**이며 unsupervised 성적표와 분리한다. SW168 fixed recipe는 실패로 닫고 그 trained weights/Adam을 이어받지 않는다. Sealed165/168 source를 수정하지 않는다. Incumbent는 SW165 3seed .804298915/.609975448/.651148975이다. .87, 모든 gate의 기여, fresh native32 70k 및 positive data scaling은 아직 미입증이다.

## 가설과 source

SW168 maximin은 local보다 FG가 좋았지만 두 출력 모두 SW164 baseline보다 낮았다. 두 trainer의 fixed TRAIN8에서 초기/최종 E_on=1, transition=0이었다. Membrane parameter 변화는 MG surrogate credit이며 이 표본에서 binary event selection의 유용성을 증명하지 않는다. 이름 필터가 `dendric_layer`를 놓쳤으므로 이전 raw dendritic norm이 측정됐다고 쓰지 않는다.

단일 새 가설은 **작은 weighted subtractive reset과 약 .5-step membrane retention을, 단위가 정규화된 느린 적분 및 raw binary-event hard reset으로 함께 바꾸면 실제 시간 누적이 assignment 학습에 쓰일 수 있다**는 것이다. Threshold만 움직였던159, DC를 제거한126, 작은 amplitude residual158과 다르다. 초기 Q/FG 손상 가능성이 크며 개선을 보장하지 않는다.

Seed1 원본 SW0097 checkpoint를 strict-load→native16-to32 map→SW143 T power2/contrast8로 시작한다. `SW0135_native32_spike_binding.foundation.load_native32_foundation(1)` 및 기존 T installer를 사용한다. Original source/state/encoder/stats/ordered512 hashes는 SW168과 동일하고 source model은 별도로 immutable 보관한다. Failed168/167 모델을 initialization으로 쓰지 않는다.

## TRAIN-only current 단위 calibration

같은 ordered512 TRAIN의 첫8 RGB, B4 두 batches, frozen original SourceT T1024/settle512에서 실제 dendritic `h_wave=sum(dendrite.h,branch)`를 캡처한다. **원본 source 방정식의 calibration 모드**에서 기존 SW134 late_rollout와 same-mode all-six tensors/Q exact parity를 먼저 확인한다. Membrane trace를 대수적으로 역산하거나 modified membrane의 h로 calibration하지 않는다.

Component d마다 r_d = sqrt(mean[(R_m*h_wave)^2 | g>0])를 8 images×1024 nodes×settled512의 **active-gate samples**에서 계산한다. h_wave는 branchwise **a_d integration 이후** dendrite.h의 sum이며, R_m를 곱한 뒤 a_m current conductance/normalization/membrane update 이전의 실제 current다. Frozen source는 a_d=0이지만 측정 위치를 이대로 고정한다. Active count도 component별로 저장한다. Accumulation CPUfloat64, 저장 r는float32 [4]. Active count0 또는 finite/strictly-positive r가 아니면 중지하며 epsilon/다른 sample scope로 구제하지 않는다. Event on/off로 current samples를 고르지 않는다. GT/VAL/FG 또는 원하는 event rate를 calibration에 쓰지 않는다. 정규화는 r를 단순히 고정하고 실제 h 값에는 detach를 걸지 않는다. 두 arm은 동일 calibration bytes를 쓴다. Calibration source와 modified-init state hashes는 구분한다.

## 정확한 동역학과 12-scalar 역할

기존 folded order는 b*4+d이다. State는 dendritic h, membrane V, **별도 previous binary E**, previous emitted S이다. V/E/S는0에서 시작한다. Float32 native act_fun_adp(MG backward)를 그대로 쓴다. Static gamma/graph, Kuramoto/coupling, delayed sinusoidal carrier/gate, dendrite projection/retention 및 a_d integration은 SW134와 같다. Previous **actual S**가 기존 `_dense_dendrite(...,previous_spike)`의 입력이다. E를 S 대신 dendrite에 넣지 않는다.

Membrane만 다음으로 바꾼다. 각 folded row에서 I=R_m*h_wave/r_d; lambda_i=sigmoid(tau_m[i]); theta_d=.06*exp(b_d); u=1+a_m,d*(integration_gate-1). Existing phase arm의 integration_gate=g이고 a_m은 기존처럼 [0,1] projection한다.

```
Vproposal = lambda_i*(1-Eprev)*Vprev + (1-lambda_i)*u*I
Vnext     = where(g == 0, Vprev, Vproposal)
Eraw      = act_fun_adp(Vnext-theta_d)
Snext     = Eraw*g
Ehistory  = where(g == 0, Eprev, Eraw)
```

Hard reset는 **다음 update의 carry에 Eprev=전 step의 Ehistory**로 적용한다. Weighted Sprev를 reset multiplier로 쓰지 않고, native `-theta*Sprev`를 중복으로 빼지 않는다. g==0이면 V와 binary-history는 hold, emitted S는0이다. Eraw는 threshold의 binary output으로 따로 저장하고 g==0이라는 이유로0을 대입하지 않는다. Ehistory는 recurrence용 hold-state이고 Eraw와 구분한다. 다음 dendritic update는 실제 Snext를 사용한다. a_m은 이제 current conductance를 조절하고 hard reset을 다시 Vprev 쪽으로 보간하지 않는다. a_d는 기존 dendrite integration, b는 existing threshold다. 이는 원본 a_m semantics도 포함한 명시적 coarse architecture change다.

Strict source load 뒤 tau_m만 source spatial offsets를 유지하여 한 번 re-center한다: tau_new_i=tau_source_i-median(tau_source)+logit(exp(-1/16)). 따라서 median lambda의 초기 e-fold는16 steps다. Native source tau_n은 바꾸지 않는다. b[4]는 log(.5/.06)로 설정하여 threshold=.5 normalized units에서 시작한다. 새 파라미터는 없다. These are fixed prospective physical constants, event-rate/GT calibration 값이 아니다. 값/초기화/hash를 저장하고 threshold·time constant sweep을 하지 않는다.

초기 theta/carrier/gate는 원본 SourceT와 같아야 하지만 새로운 S가 dendritic feedback을 바꾸므로 이후 dendrite/membrane/spike/Q의 source equality는 **요구하지 않는다**. Source와 modified coding의 초기 FG parity도 요구하지 않는다.

## 같은 forward·capacity의 두 학습 arm

* `history_live` primary: live tail에서 Vprev/Eprev의 위 recurrence credit를 유지한다.
* `history_detached` control: **membrane carry의 operands로 사용할 때만** Vprev.detach(), Eprev.detach()를 쓴다. Current I, lambda, b, a_m, theta/delayed-theta history, dendritic h와 emitted-S→dendrite feedback은 그대로 live다. Theta/dendrite state를 매 step detach하지 않는다. 모든 과거 경로를 끊는 control이라고 부르지 않는다.

같은 weights/input에서 두 forward traces/events/Q는 bitwise 같아야 한다. Parameter counts/shapes/init/optimizer는 동일하고 lambda는 control에서도 current coefficient(1-lambda)를 통해 active다. Control은 forward memory가 없는 모델이 아니라 **explicit membrane-history 학습 credit**의 대조다. 이 비교만으로 forward memory 필요성을 주장하지 않는다.

## rollout·훈련·checkpoint API 범위

별도 local opt-in rollout을 `SW0134_native_spike_binding/rollout.py:late_rollout`의 실제 static setup/state loop에 맞춰 구현한다. Native production files/global monkeypatch는 수정하지 않는다. Graph/gamma drive/factorized prepare_coupling은 grad enabled 상태에서 burn-in 전에 한 번 준비한다. Full1024를 수행하고960 boundary에서 theta와 delayed theta list, dendrite.h, V, Ehistory, S를 한 번 detach한다. 그 뒤64 steps는 candidate live/control operand-detach 규칙이다. 실제 settled512(448 detached+64 live)의 production positive-product Q를 loss에 쓴다. Native stack(T,B*4,N)→permute/reshape와 component mean 순서를 보존하고 기존 six fields 및 별도 binary events/current/reset diagnostics를 반환한다.

두 arm 모두 original actual RGB encoder8채널+learned graph+native core+기존 a_d/a_m/b12 scalars를 trainable로 한다. `SW0168/train.py`의 family partition/unique identity/checkpoint/batch 경로를 검토하여 재사용한다. Base/graph/native LR3e-5, integration LR1e-3, fresh Adam/global clip1, 같은512 TRAIN/두 passes=64 updates, B16 micro4, 새 fixed shuffle169001. Source support/feasible targets와 **constrained maximin loss 하나**는 completed168과 동일하며 local loss/phase prior/decoder/auxiliary activity loss를 더하지 않는다. B4 image means→logical mean→clip/Adam 한 번이다. Threshold .5 초기화를 strict load 후 실제 tensor에서 확인하고 checkpoint에도 저장한다.

완료 logical update별 optimizer/RNG/order/state/완료counter를 transactional 저장한다. Boundary는 별도 E state semantics를 포함한 runner fingerprint/runtime/source/r-calibration에 묶는다. Resume에 fresh Adam이나 unfinished microbatch gradients를 쓰지 않는다. 실제 두 disposable updates로 runtime/peak memory를 측정한 뒤64×2arm 예산을 기록한다.

## 실제 GPU admission — optimizer 이전부터 고정

1. Immutable source/calibration six-trace/Q parity, source unchanged/hash, eight ordered IDs/noGT, r/current fold component mapping, actual T graph/coupling을 확인한다. Modified live/detached의 초기 full1024 six traces/events/Q exact equality를 확인한다.
2. TRAIN8 settled512에서 eligible=(g>0)로 정의하여 **raw Eraw의 active-gate conditional occupancy**만 판정한다. 각 component aggregate가 0<event_on_given_active<1이어야 한다. Consecutive eligible=(g_t>0 AND g_{t-1}>0)인 step pair에서만 raw E flip을 센다. 각 component의 image×node units 중 적어도10%에 이러한 consecutive-active flip이 하나 이상 있어야 한다. 각 image/component도 eligible samples가 존재하고 eligible E가 전부0/전부1이면 안 된다. Raw 전체 E/active E/Ehistory/S 통계와 eligible denominator를 각각 보고한다. g==0에서 S가0이 되는 현상이나 hold 구간을 끼운 event 차이는 membrane selection/transition으로 세지 않는다. 이는 혼합·비상수 selection의 최소 feasibility 조건이지 desired firing density나 task 성능 목표가 아니다. 미통과 시 같은 recipe를 중지하며 time constant/threshold/r/sample을 바꾸어 재시도하지 않는다. Initial FG/GT separation은 admission에 쓰지 않는다.
3. Positive-product clamp에서 dL/dQ만 있고 actual dL/dS가 전부0이면 중지한다. Two disposable updates에서 finite actual losses/gradients/deltas, encoder/graph/native membrane tau_m 및 existing b의 actionable credit와 실제 update를 확인한다. Raw current I/actual earlier V/E를 참조해 live arm의 recurrence partial credit와 control의 해당 direct operand credit0를 검사한다; stacked output sibling gradient로 대신하지 않는다. 첫 step h feedback 등 다른 history 경로의 존재는 숨기지 않는다. CPU tiny recurrence에서 exact hard reset/hold/fold/boundary-detach를 검증한다.
4. Kuramoto·`dendric_layer`·membrane·encoder·graph·a_d/a_m/b를 실제 parameter-name set으로 따로 기록한다. Individual zero를 인위적 update로 통과시키지 않는다. 정상 finite/positive credit가 있어도 event usefulness나 FG 개선으로 해석하지 않는다. r를 current와 함께 detach하거나 E만 posthoc readout용으로 만드는 구현은 불허한다.

## fixed320 두 출력·판정·실제 necessity

최종64 checkpoint만 사용한다. 각 arm의 동일 learned B1/T1024/settle512/live0 rollout에서 아래 두 co-primary를 모두 만든다.

* Conditional-support 진단: frozen original SourceT QCC F + learned actual Q/A의 SW165 induced connected solver. Unseen deployment에는 old+new 두 rollout이 필요하므로 실용 endpoint로 대체하지 않는다. TRAIN F는 원래 CPU-B4 source cache에서 오며 energy의 GPU Q와 구분한다.
* Standalone: learned actual component spikes를 원래 CPU QCC(min8/largest-BG)로 얻은 own F + 동일 learned GPU Q/actual A/gain의 SW165 solver. 하나의 learned rollout이며 RGB/gamma/GT/count를 head로 우회하지 않는다.

양 arm×두 출력 및 SourceT/SW164/Slot references의 predictions/provenance를 먼저 freeze하고 검증한 뒤 IDs1320..1639/native32/modal4px의 기존 세 지표를 계산한다. History-live가 **각 출력에서** seed1 SW164 FG .770932099 및 해당 history-detached FG보다 높고 각 paired95% CI lower>0이어야 한다. Conditional objectIoU>=SW164 및 foregroundIoU per-image exact, standalone 두 IoU means>=SW164도 필요하다. Shared320 draws/bootstrap10000/RNG169. 두 출력 중 좋은 것만 고르지 않는다. 실패면 threshold/time-scale/LR/passes/seed sweep 및 native70k fanout 없이 닫는다. 다른 useful tradeoff는 별도 기록하되 통과로 바꾸지 않는다.

FG gate 통과 후에만 frozen live model의 full320 두 **미리 정한** causal interventions를 수행한다: `history_off`는 carry의 Vprev를0으로 대체하되 current/lambda/reset/E/S feedback을 그대로 유지; `reset_off`는 carry의 (1-Eprev)를1로 대체하되 나머지는 그대로 유지한다. Initial TRAIN8에서도 이 두 trace 변화를 기록한다. Interventions로 생긴 실제 S→dendrite feedback은 다시 rollout하며 readout arrays만 대체하지 않는다. Frozen full320 FG paired CI가 감소를 지지해야 해당 조건의 forward contribution을 말한다. Collapse alone은 기능적 유용성의 충분조건이 아니므로 prediction/occupancy/margin도 함께 보고한다. Retrained necessity/all-gate proof는 추가 matched retraining이 필요하다.

Source-preserving 초기화가 아닌 새 coding의 supervised pilot이다. 성공해도 세 seed/.87/Slot all3 superiority, unsupervised learning, fresh native32 70k/scaling을 자동 주장하지 않는다. Reserve90640..90959는 읽지 않는다. 새 생성물은 중앙 `collaborative_testing/prepare_output.ps1` output 정책으로 별도 namespace에 저장하고 이전 실패 기록은 보존한다.
