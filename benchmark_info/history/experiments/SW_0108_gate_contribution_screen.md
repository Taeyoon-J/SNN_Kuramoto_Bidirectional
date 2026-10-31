# SW_0108_gate_contribution_screen

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `ec5595031c2ca455b32e869a2677717f80823edfcfdccc321420dceaec8aebdb`

````markdown
# SW0108: fixed-checkpoint gate contribution screening

Use SW0097 seed0 positive_frozen and the original native validation gamma
cache on IDs1320-1639. Compare the baseline with five separate rollout
interventions: zero Kuramoto coupling, constant sinusoidal gate, no
dendritic retention, no membrane retention, and forced-on actual events.
The final classifier still consumes actual component spikes. These are
checkpoint-sensitivity tests, not matched retraining ablations.

## Invalid first attempt and correction

The first evaluator command omitted `--membrane-vth .06`, inheriting the
shared evaluator default2.0. Baseline and intervention scores from that
attempt cannot be used under the fixed contract. The queue was stopped;
all recorded PIDs were confirmed absent before the entire output tree,
including logs and lock, was preserved on Frontier at
`trained_models/SW0108_gate_contribution_screen_attempt1_vth2`.
The old preflight and halted queue state are archived within that tree.

Corrected local code supplies .06 explicitly and requires all320 baseline
per-image metrics and means to match the registered SW0097 seed0 record
within1e-10 before any intervention may start. The baseline record SHA is
`009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0`.
The existing server record in
`trained_models/SW0097_graph_adaptation/seed0_positive_frozen/evaluation.json`
was independently confirmed to have that exact SHA; no reference-file
transfer is necessary.

## Deployment status

2026-10-08: user explicitly authorized corrected source transfer and testing.
Deployment is complete with SHA-verified source files. Frontier Python
compilation, six SW0108 unit tests and thirteen SW0109 unit tests passed.
The coordinator was verified alive, using the original
native validation gamma cache. Its observed state is waiting_for_free_gpu:
seven tasks pending and no GPU worker started. All devices are occupied by
foreign processes. The coordinator will run actual-model preflight and the
full320 registered baseline before permitting intervention evaluations.

Historical blocker: corrected code was prepared locally but execution had not restarted.
Automatic approval review rejected transferring the four corrected source
files to Frontier, citing missing explicit source-transfer authorization
and prohibiting a different transfer route. An explicit user approval
request was subsequently approved. No transfer workaround was used while
the request was pending.
After authorization, deploy the frozen files, run server unit tests and
actual-model preflight, then start the baseline-first exclusive GPU queue.
Foreign GPU jobs must not be shared or stopped.

## Trained retention audit (2026-10-09)

Read-only CPU state inspection of all three SW0097 cores found mean dendritic sigmoid(tau_n) .1719/.1751/.1560 and membrane sigmoid(tau_m) .1517/.1676/.1389. Coefficient e-folding durations average about .55-.60 and .51-.58 active updates, respectively. These are recurrence coefficients, not measured whole-model memory: gate holds and nonlinear spike resets matter. Combined with the earlier neutral retention-removal intervention, this motivates a prospective temporal integration hypothesis; it does not prove that changing retention will improve segmentation. See `trained_retention_audit_20261009.json`.

````

</details>

## research_note_ko_20261008.md

<details>
<summary>research_note_ko_20261008.md</summary>

Source SHA256: `6f7e404344c0f7655ac4b65003ad349d9758fee4d8601e39fe61d25caef3395d`

````markdown
# SW0108: 게이트 기여도 스크리닝 (부분 결과)

이 분석은 SW0097 seed 0의 고정된 체크포인트에 등록된 단일 개입을 적용한 GT-free 추론 민감도 실험이다. 검증 이미지 320장(ID 1320–1639), batch 8, 1024 step 중 마지막 512 step을 분석했다. 여섯 개 arm 중 baseline과 세 개 개입만 완료됐고, `membrane_no_retention`과 `events_forced_on`은 미완료다.

구현 경로에서 막 전위 계층은 매 시점 `e = act_fun_adp(inputs_)`를 계산하고 실제 출력을 `spike = e * g_wave_t`로 만든다. 이벤트는 임계값 기반 이진 함수이고, gate는 펼쳐진 scalar `g_wave_t`다. 따라서 baseline의 마지막 절반 이벤트 평균 0.998744은 이벤트가 거의 항상 열려 있음을 시사한다. settled gate 평균/표준편차는 0.500/0.354; gate의 단위별 시간 표준편차 평균은 0.351이다. 이벤트 단위의 90.84%는 마지막 절반 내내 항상 켜져 있었다. 이는 이벤트 선택기가 자주 열린다는 관측이며 막 동역학 전체가 중요하지 않다는 뜻은 아니다.

`kuramoto_K0`는 FG-ARI가 크게 하락했고 paired-image 95% CI는 [-0.827, -0.788]였다. 이는 이 고정된 파라미터 상태에서 결합을 제거한 추론이 성능을 무너뜨렸다는 증거다. `constant_gate_half`에서는 실제 spike가 거의 상수 0.5가 되었고 세 지표 모두 0이었다. 이 개입은 막 gate와 함께 carrier를 `0.5*sin(theta)`로 바꾸므로 gate의 시간 변조와 carrier 크기 효과를 분리하지 못한다. `dendrite_no_retention` 결과의 FG-ARI 변화 CI는 [-0.0016, 0.0054]로 0을 포함한다. 이 단일 seed 자료에서 뚜렷한 성능 변화를 확인하지 못했다.

이 결과는 학습된 인과효과, 전체 membrane 기여도의 부재, 데이터 규모에 대한 설명을 확정하지 않는다. scalar hook은 객체 mask와의 상관을 측정하지 않는다. 특히 bootstrap은 이미지 쌍을 재표집한 단일 학습 seed 조건부 구간이다. 남은 두 arm 완료 후 해석을 갱신할 수 있지만, 이 결과만으로 새 하이퍼파라미터를 선택하지 않는다.

````

</details>
