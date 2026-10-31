# SW_0169_normalized_hard_reset

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## ADMISSION_RESULT.md

<details>
<summary>ADMISSION_RESULT.md</summary>

Source SHA256: `a3bad1cb9a10db3b48aa9b788ccbef6e7b6f847d9bed253f05366fe40cdd7f41`

````markdown
# SW0169 실제 GPU admission 통과

2026-10-15. 별도 v3 parent3713598/worker3713642(GPU1)가 `complete_admitted`로 정상 종료했다. 보고서 SHA256은 `91662b392f6a7b927cdc1da2bde411520107665df404329802d13dab9fc39b30`이다. Root가 현재 실제 source/cache/코드로 queue/public validator를 CPU에서 다시 실행하고, 아래 물리·gradient·업데이트 증거까지 독립 검증했다. CUDA는 이 재검증에서 초기화하지 않았다.

고정 TRAIN8·B4·전체1024/settle512/tail64 범위에서 원본 source와 opt-in source-mode의 six traces/Q/A가 정확히 같았다. 모든8행의 저장 S/A 및 사전에 정한 historical-layout cache Q도 정확히 재현했다. Source는 변하지 않았다. Active-current RMS float32 r=[2.812951565,2.679559708,2.744066715,2.076063871]를 GT 없이 고정한 후에만 별도 CPU interpreter에서 TRAIN targets를 읽었다. 새로운 r·threshold·time constant 탐색은 하지 않았다.

원본 source의 이 표본에서 E가 항상1이었던 상태와 달리, 새 초기 모델의 **g>0 조건부 raw E-on 비율은 image/component에 따라 .0399971.. .0731432**였다. 시간 전환은 g가 연속으로 활성인 pair에 대해서만 집계했다. 각 component의 전체8×1024 node-units 중 한 번 이상 이러한 E 전환이 발생한 비율은 다음과 같다. Gate-off에서 S가0인 것은 전환 증거로 세지 않았다.

| Component | 연속 active-gate에서 E가 전환한 node-unit 비율 |
|---|---:|
| 0 | 1.000000000 |
| 1 | .998535156 |
| 2 | .984863281 |
| 3 | .852539063 |

Live/control의 초기 six traces/events/Q hash는8행 모두 정확히 같다. 같은 maximin loss를 사용하고 직접 membrane-history credit만 다르다. 각 arm에서 실제 spike-derived Q를 사용한 disposable B8 update2회가 완료됐고 모든 microbatch의 실제 dL/dS norm이 양수였다(최솟값11.4435551). Encoder·graph·native core·기존 integration family가 모두 실제로 업데이트됐으며, 정확한 `kuramoto`, `dendric_layer`, `membrane_layer` 이름별 gradient와 변화량도 기록했다. tau_m와 b도 두 update 모두 바뀌었다.

실제 마지막 step의 값을 고정해 재생한 **local carry partial**에서 live의 previous V/E 감도 norm은 .0057430824/.0024383642, detached control은0/0이었다. 이것은 직접 carry 경로의 국소 미분 검증이며, control의 emitted-S→dendrite feedback까지 끊겼다거나 전체 시간 경로의 total gradient가0이라는 뜻이 아니다. 이 진단은 원래 학습 rollout의 gradient를 바꾸지 않는다.

두 update와 이후 TRAIN 진단에 걸린 시간은 live27.8722초/control27.6628초, peak allocated GPU memory는 약3.03GB였다. 이 시간에는 source-calibration/초기 parity 준비가 포함되지 않으며 B16 full-training update의 직접 시간 측정도 아니다.

**이는 선택적 발화와 학습 가능성의 admission 성공이다. FG-ARI/IoU 개선이나 gate의 물체 구분 기여는 아직 입증하지 않았다.** VAL은 읽지 않았고,512장·두 pass·64 logical B16 updates의 본 pilot과 fixed320 두 출력 평가가 다음 단계다. Fresh original source/tau/b와 동일 r로 시작하고 disposable update weights/Adam을 이어받지 않는다. SW165 incumbent(.804298915/.609975448/.651148975), 세 seed/.87/70k scaling 요구는 그대로다. 초기 두 구현 실패는 ADMISSION_START.md와 각 원본 archive에 보존했다.

증거: `results_archive/admission_completed_20261015/server_records.json`, `independent_verification.json`. 로컬·서버 CPU30 tests와 실제 queue CPU 진입 검증도 해당 packet에 있다. 생성물의 실제 위치는 중앙 `collaborative_testing/generated/`이다.

````

</details>

## ADMISSION_START.md

<details>
<summary>ADMISSION_START.md</summary>

Source SHA256: `731d3482f92f3e93f2c8e786455ab57810a251404b5df5712ff0c26a1d434f01`

````markdown
# SW0169 initial GPU source admission

2026-10-15. Root independently reran all21 focused CPU tests locally and on Frontier: PASS, zero skips. Actual source CPU preparation and the queue's actual `_source_admission()` also passed with CUDA uninitialized, eight exact source-order TRAIN rows and no GT targets. Existing17 calibration dependencies matched local/server bytes before deployment. The ownership queue was then launched once: parent3711414, actual GPU1 worker3711465, both confirmed through `/proc` and NVML. Foreign GPU0 PID2653200 was left untouched.

This attempt is terminal **failed before optimizer updates** at the source calibration guard: `B4 source Q differs from immutable context cache at row4`. Native source versus opt-in source-copy six-field/Q/A exact checks had passed for the two B4 groups up to that guard; saved settled S also matched at row4. No modified-model training, FG evaluation, or positive admission is claimed. Full parent/worker logs and the original terminal state are preserved in `results_archive/initial_source_parity_failure_20261015/server_records.json`.

The actual historical producer `SW_0163_relation_context_assignment/context_cache.py` uses two fixed arithmetic layouts: rows0–3 derive Q from the live full trace's strided settled view; subsequent groups use contiguous `np.stack(saved settled S)` transferred to GPU. The failed SW169 guard incorrectly demanded bitwise equality between live full-trace Q and the latter cached-Q layout. Same spike values do not establish bitwise equality across those reduction layouts.

Sol6.1 reviewed the actual producer and approved a separately versioned calibration/preflight/queue correction: keep native versus source-copy six/Q/A exact; keep saved S/A exact for all8; replay each cached Q in its original producer layout, specified by batch before execution. Record both Q hashes, layouts/strides, maximum difference and0.5-edge differences. Do not select a passing layout after failure, change tolerances, replace actual training Q, change CPU-B4 support, change RMS calibration/data, or overwrite this failed namespace. No additional precision search is required.

The new membrane hypothesis remains untested in this attempt because the failure occurred before its admission stage. SW165 remains the incumbent and the research goal stays active. A version2 launch requires current code/tests and actual CPU checks first.

## Version2 target handoff and version3 continuation

Version2 passed the historical same-mode source/cache checks, then terminated before optimizer updates in the fresh CPU target worker: SW0155 `modal_train_targets32()` returns integer `[8,1024]`, whereas the new helper required `[8,32,32]`. The missing reshape is an implementation error, not evidence against the membrane hypothesis. Root subsequently executed the actual server TRAIN producer: raw masks `[8,128,128]`, modal output `[8,1024]` int64, exact reshape `[8,32,32]`, CUDA uninitialized. Original v2 terminal state/logs remain in `results_archive/v2_target_shape_failure_20261015/`.

Version3 adds that explicit shape conversion and saves successful source calibration immediately before the CPU target handoff. Calibration_v2 physics/layout checks are unchanged. Root reran30 meaningful CPU tests locally and on the server, and the actual queue CPU source admission passed before launch. Separate v3 parent3713598 and actual GPU1 worker3713642 were confirmed live through `/proc`/NVML. Queue fingerprint03b96fb3b2938cab08cedada002033f5c6e892434b8117fda29c76299c27bfe9. This is bounded8-image admission; the64-update512-image pilot has not started at this launch record. Its final result must be checked separately.

````

</details>

## CPU_PREPARATION.md

<details>
<summary>CPU_PREPARATION.md</summary>

Source SHA256: `b62e805e50c12ba7870c6cb9185dcd5f7d9d312e96f2832ef51dbf14143aa910`

````markdown
# SW0169 actual CPU preparation

2026-10-15. The normalized slow-membrane/binary-hard-reset opt-in implementation and source calibration leaf files were installed immutably on Frontier. The registered native source files and SW0168 were not changed.

Root independently executed all 11 leaf/rollout contract tests locally and on the server: PASS, no skips. Coverage includes binary reset versus emitted amplitude, zero-gate V/E-history hold with actual S=0, active-gate CPU-float64 RMS, fixed tau/b initialization, B2 source six-field/production-Q parity, candidate/control bitwise forward parity, component folding, actual 1024-step boundary960 and tail credit, and actual surrogate carry gradients versus detached control.

Actual `prepare_calibration_cpu()` passed on the server with CUDA uninitialized, no GT and no VAL read. Its source/cache/RGB bindings resolve exactly the first eight ordered TRAIN IDs: 68304,49032,70041,7829,16499,31675,67612,54671, in two contiguous B4 groups. Trace probe positions in the earlier SW0168 module are not image-row indices. Source cache and context digest families stay separate.

This preparation is not the full-size GPU source-calibration parity, mixed-event/gradient admission, training, evaluation, or a positive performance result. Those stages remain required before the pilot may expand. SW165 remains the incumbent.

Evidence: `results_archive/leaf_CPU_20261015/server_records.json` (full server test output, source proof, installed file hashes). Physical generated output is under `collaborative_testing/generated/`; canonical results access is a junction. No dataset/checkpoint/key was transferred or published.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `30809417207f043cbd843409072ad00d14eff0e1fab62a9fe49abf60f71408c9`

````markdown
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

````

</details>

## PROSPECTIVE_NEXT_STEP.md

<details>
<summary>PROSPECTIVE_NEXT_STEP.md</summary>

Source SHA256: `22afe67fc1bf9c1668c0c8d09675bc39031b601197a1dcc23d0382e972509aab`

````markdown
# Conditional next hypothesis — causal synaptic-current readout

2026-10-15. **조건부 아이디어이며 아직 새 실험을 등록하거나 실행하지 않았다.** SW169 full320 판정과 독립 검증 후 최종 설계를 검토한다. 기존 source/recipe/실행은 변경하지 않는다. 현재 최종 평가 점수나 후보의 성공 결과를 주장하지 않는다.

## 현재 증거의 범위

`results_archive/completed_training_20261015/server_records.json`의 fixed TRAIN8에서 두 arm initial event density 평균은 .06143558, final live .05846727 / detached .05785352이다. Q mean은 initial .00178529, final .00185593/.00171727이다. Initial standalone foreground patch 수 min/mean/max는0/2/8, final live0/5/16 및 detached0/4/16이다. 전 이미지가 정확히0이라고 쓰지 않는다. 이 기록의 fixed8 maximin loss는 .49625620→.49695632/.49814903이며 epoch minibatch loss와 다른 통계다. Dataset 전체 또는 VAL 성능의 결론은 아니다.

`spike_classifier.py:spike_synchrony_affinity`는 시간 평균을 뺀 component별 Pearson을 positive-clamp한 후 네 값을 곱한다. 동일 rho라면 Q>.5에는 각각 rho>.8409가 필요하다. Q mean에는 비퇴화한 diagonal의 약1/1024 기여도 포함된다. 따라서 sparse coding에서 off-diagonal 관계가 매우 약하다는 증거는 있지만, within-object timing jitter가 원인이라는 증명은 없다. Sparse code 자체의 정보 손실과 instant zero-lag readout mismatch를 구분해야 한다.

## 고정된 한 가지 구조 가설

원본 SourceT에서 시작한 SW169 normalized hard-reset 모델에 actual emitted S=E*g만 입력받는 causal state를 둔다:

`Z_0=0; Z_t=exp(-1/16)*Z_(t-1)+(1-exp(-1/16))*S_t`.

16-step time constant는 이 아이디어에서 고정하며 GT/활동률/VAL로 고르지 않는다. 다른 width·threshold·clamp 조합을 탐색하지 않는다. Z는 **actual-spike-derived synaptic current**이며 binary spike라고 부르지 않는다. RGB/gamma/theta 또는 직접 membrane 값을 readout에 추가하지 않는다. Raw S의 기존 dendrite feedback과 raw E의 hard reset은 그대로다.

Z는 full1024 동안 계산하고 settled512에서 기존 centered positive-product affinity Q_Z를 만든다. Maximin TRAIN loss와 SW165 signed energy/induced-connected solver 모두 같은 Q_Z를 사용한다. Actual A/gain의 경로는 그대로다. Control은 동일 모델의 identity synapse Z=S다. 두 arm 모두 encoder/graph/core/기존12 scalars, TRAIN512, B16 micro4,64updates, 기존 loss/optimizer/order,full1024/settle512/tail64를 공유한다. Z도960 boundary에서 detach하며 이후 live recurrence credit는 유지한다. 새 decoder/head/목표밀도/auxiliary loss는 없다.

Conditional-support SourceT-F 진단과 standalone own-Q_Z-QCC 출력은 모두 유지한다. Native32/modal4px/fixed320/세 지표/GT-last/금지 reserve 조건은 바꾸지 않는다. Source/base event-history의 변화는 SW169 구조에서 오고, 새 비교가 분리하는 것은 causal synaptic readout과 그 학습 credit다. Filtering이 연속 gate/rate만 복원하여 binary-event/history 기여가 여전히 없을 가능성을 명시한다.

## 판별·최소 admission·대조군 재사용

최종 학습 모델뿐 아니라 **동일 init, optimizer0의 filter-only endpoint**도 사전 명시해야 한다. Frozen initial identity/filter 양쪽을 동일 actual rollout에서 읽고 predictions를 먼저 고정한다. 이것으로 단순 readout transformation과 학습 후 추가 개선을 분리한다. 초기 endpoint를 보고 width/모델/checkpoint를 선택하지 않는다. 최종 설계에서 평가 순서와 train expansion gate를 먼저 확정한다.

TRAIN-only feasibility는 source/cache/actualS lineage, identity branch의 기존 forward/Q/loss parity, full1024 causal filtering/settle/boundary 처리, finite·비상수 Z/Q(상수/allzero 출력이 아닌지), actual Q_Z→Z→S→trainable families의 finite actionable gradients, 동일 init 두 disposable logical updates의 실제 optimizer changes를 확인한다. Direct state partial은 실제 state references로 검사한다. Fixed8의 positive/negative critical-edge distributions와 loss는 기술하되, 아직 보지 않은 candidate의 task score를 위한 숫자 cutoff를 여기서 만들지 않는다. Admission 성공은 object binding/gating efficacy의 증거가 아니다. Candidate를 본 뒤 cutoff를 만들어 통과시키지 않으며 최종 등록에서 필요 조건을 먼저 고정한다.

Identity 학습은 SW169 `history_live`를 재사용하면 중복64updates를 줄일 수 있다. **실제 init/weights, raw forward, identity Q/loss, data IDs/order, B4 grouping, optimizer states/steps/projection, RNG 및 truncation까지 동일함이 검증될 때만** 가능하다. 새로운 branch가 RNG/normalization/loss arithmetic을 바꾸거나 해당 증명이 없으면 fresh matched identity control이 필요하다. 실패한 control을 다른 seed나 checkpoint로 대체하지 않는다.

최종 설계에서 filter-only 대비 learned improvement와 identity 대비 효과, SW165 baseline 보존 및 두 출력의 paired FG/IoU gates를 명시한 후만 실행한다. 성공하더라도 E-forced1/history/reset 실제 동역학 개입 없이 발화/메모리의 필요성을 주장하지 않는다. .87/세 seed/fresh native32 70k/positive data scaling/unsupervised 성과는 각각 별도 증거가 필요하다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `e938a68052b572c7c670d2f780cbf907c24088a76a837c39c90e0fa0f020cbe6`

````markdown
# SW0169 final result: rejected

2026-10-15. Both registered 64-update pilots and all fixed320 predictions completed.
The public CPU validator independently recomputed every metric, prediction binding,
paired interval and decision from the frozen predictions and canonical modal32 GT.
Root separately rechecked all 320-value means, paired bootstrap intervals and code hashes.

| Seed1 fixed320/native32 endpoint | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| history_live_conditional | 0.03837044 | 0.61499343 | 0.05815510 |
| history_live_standalone | 0.00000000 | 0.00000000 | 0.00000000 |
| history_detached_conditional | 0.03858640 | 0.61499343 | 0.05905823 |
| history_detached_standalone | 0.00007265 | 0.00009091 | 0.00008681 |
| SW0164 | 0.77093210 | 0.61499343 | 0.63052348 |
| Source-T | 0.75527308 | 0.61499343 | 0.57951379 |
| Slot | 0.84181929 | 0.20341671 | 0.22211516 |

The registered co-primary gate failed. Live did not improve either output over
SW164 or its detached-history control. Conditional foreground is preserved exactly,
but object assignments collapse; the standalone live endpoint scores zero on all
three metrics. Nonconstant firing and nonzero gradients were insufficient for
object binding. This is a negative supervised pilot, not evidence that memory
or hard reset is generally useless. Sparse temporal coding/readout mismatch is
a hypothesis supported by TRAIN diagnostics, not an established causal explanation.

Close this recipe without threshold/time-scale/LR/seed sweeps, 70k expansion or
the positive-gate-only history/reset interventions. SW165 remains the verified
three-seed incumbent (.804298915 / .609975448 / .651148975); do not compare these
seed1 pilot results with its three-seed mean as if their scopes were identical.

The conditional synaptic-current idea in PROSPECTIVE_NEXT_STEP.md can now receive
a separate prospective design and cheap TRAIN-only feasibility test. It is not
a successful result and cannot change this rejection.

Evidence: results_archive/completed_score_20261015/server_records.json and
independent_verification.json. Score SHA256: `896562821faa67bb91acdc0612b2cf56a9bd594d63dcf944d002326b06d5875c`.

````

</details>

## TRAINING_AND_EVALUATION.md

<details>
<summary>TRAINING_AND_EVALUATION.md</summary>

Source SHA256: `449540cd428b976628dd56658f632eb2b20ea810caf1a65a9171163808bacc4d`

````markdown
# SW0169 training and automatic evaluation registration

2026-10-15. Two supervised TRAIN512 pilots were launched after the actual
512-row CPU admission and eight-row production-rollout diagnostic passed
(562.49 seconds, no validation data). Both use fresh source initialization,
the same frozen current normalization, maximin loss, B16/micro4, and 64 updates.
Only explicit membrane-history gradient credit differs; emitted-spike feedback
remains live in both arms. Disposable admission weights and Adam are not reused.

Training supervisors: history_live 3716549 / worker 3716709 (physical GPU2);
history_detached 3716550 / worker 3716710 (physical GPU1). At this archived
observation both actual workers were live in CPU preparation. Queue status
`running_training` alone is not evidence of CUDA learning or a committed update.

Automatic dependencies are registered and independently observed live:
prediction supervisors 3719599 and 3719615 wait for both final training artifacts;
CPU scorer 3719633 waits for both complete fixed320 prediction artifacts. Prediction
uses no validation masks. GT is opened only after all prediction bytes are frozen.
Original Source-T, SW164 and Slot reference predictions/protocols were checked
against the same 320 IDs and native32 schema on the real server, without GT.

Local evaluation/queue tests: 11 passed. Server training tests: 36 passed;
server evaluation/queue tests: 11 passed. Root rechecked deployed file hashes,
CPU receipts and actual /proc argv/liveness before archiving this evidence.

There is no SW169 performance result yet. SW165 remains the verified three-seed
incumbent (.804298915 / .609975448 / .651148975). This pilot does not prove
FG-ARI .87, fresh native32 70k scaling, or causal utility of every gate.

Evidence: results_archive/training_and_evaluation_registered_20261015/
server_records.json and independent_verification.json. Generated files physically
reside under collaborative_testing/generated via the existing archive junction.

Subsequent actual GPU start is confirmed in `actual_GPU_training_started.json`:
both same worker PIDs are present in NVML (about 4.1 GiB each), and each arm
has a validated committed boundary with eight completed optimizer updates.
This supersedes the preparation-only observation above; it is not a final score.

````

</details>

## TRAINING_RESULT.md

<details>
<summary>TRAINING_RESULT.md</summary>

Source SHA256: `42c1556d79a0e25363b0cb4c241fbdbfee70d4da7cdc792296736aad151ed987`

````markdown
# SW0169 completed training; fixed320 evaluation pending

Both registered TRAIN512 arms completed all 64 B16 updates (two passes).
The public CPU validator rechecked actual model, optimizer, complete exposure
order, diagnostics, final durable boundary and source fingerprints. Root then
independently checked the collected manifests, all update counters and current
local code hashes. No validation masks were opened for this collection.

| TRAIN8 diagnostic | Initial, both arms | Final history_live | Final history_detached |
|---|---:|---:|---:|
| Mean raw event-on fraction | .061436 | .058467 | .057854 |
| Mean event-transition fraction | .117920 | .112863 | .112115 |
| Mean all-pair Q | .001785 | .001856 | .001717 |
| Mean standalone foreground patches / 1024 | 2 | 5 | 4 |
| Standalone foreground patches min / max | 0 / 8 | 0 / 16 | 0 / 16 |
| Mean constrained maximin loss | .496256 | .496956 | .498149 |

These are the same fixed eight TRAIN examples, not validation scores. Q means
include the diagonal and are not evidence of same-object separation. Selection
is nonconstant, but this alone has not produced useful standalone foreground:
the readout is almost empty, not identically empty in every row. The fixed-eight
objective increased slightly; shuffled training-update loss is a different
quantity (first8/last8 updates: live .506217/.505480; detached .506529/.503736).

Full fixed320 prediction automatically started from the two final checkpoints.
No favorable endpoint, intermediate checkpoint or threshold is selected. The
registered co-primary gate is still pending; these TRAIN diagnostics do not
establish its final outcome. Existing SW165 remains the performance incumbent.

Final manifest SHA256: live
`678886c1f18602608a08dcdc144d9a2c42b3f11e3710a15967240ae51e2f64bb`;
detached `7856158558ea0a062701c89663d0f95bb3e886afbdcd9ff6a758883060d585e2`.
Evidence: `results_archive/completed_training_20261015/server_records.json`
and `independent_summary.json`. No dataset or checkpoint bytes are published.

````

</details>
