# SW_0192_native64_incumbent_mini

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `d5a739cf6347398f675a7360ba16e6d0d2b85ac933207cc249e62d2a095934f6`

````markdown
# SW0192: native64 continuation from the verified SW0186 incumbent

User direction on 2026-10-17: move the primary development grid to64x64 and
target FG-ARI>=0.90. Keep the successful encoder/graph/gating/spike/QCE direction.
Use small TRAIN/development comparisons; reserve a joint70k run for a large,
replicated gain. Do not confuse the old SW0154 transferred-source score with a
newly trained64 result. SW0191 evaluation continues unchanged.

Start with predetermined seed0, exact completed SW0186 full learned weights.
Map native32 node-leading parameters by row-major2x2 parent replication and
pair parameters on BOTH axes. Copy shared learned tensors, the full learned
RGB/context CNN and integration scalars exactly. Regenerate only identity SC
and physical grid distance (0.25 units/cell). Graph top-k512 preserves the
native32 128/1024 neighbor fraction. Keep the SourceT squared-contrast8 operator;
reduce its exact row-chunk to1 for64 memory, without changing intermediates,
steps, radius, temperature or the mathematical operator. No core equations change.

Make actual native64 gamma from the same native128 RGB/feature normalization,
using64 adaptive pooling. TRAIN64 labels come directly from canonical128 pixel
labels through production2x2 modal pooling; never upsample32 targets. Keep the
same TRAIN512 ordered cohort and use its first64 rows. No validation GT is read
by construction, CPU admission or the initial probe.

Initial stage is TWO actual B4/micro1 native64 QCE proposals (8 TRAIN images),
T1024/settle512/live-tail64, unchanged five learning rates, clip1 and integration
projections. Match same-RNG full-forward loss before/gradient/post; accept only
strict full-forward descent, otherwise restore model/Adam/RNG transaction.
Measure emitted-spike gradient, all five parameter-family gradients/proposed
deltas, peak memory and actual forward/backward/proposal time. Persist each
complete boundary atomically, including model/Adam/RNG/history. Retain SW0191's
same-software/hardware/logical-CUDA0 portable resume policy and actual GPU ledger.
Scientific rejection if first-two credit fails; no pretend training success.

After resource/credit admission, the planned mini comparison is16 B4 proposals
on those64 TRAIN images, with paired before/after native64 predictions on32
fixed development images (IDs1320..1351). Freeze both masks before GT scoring;
report all three metrics against the matched Slot64 subset and the mapped
incumbent, plus time/memory. This later mini stage is not yet implemented by the
resource probe and must not be claimed queued or complete. Replicate a meaningful
gain across seeds0/1/2 on the full fixed320 grid before leaderboard promotion.
The goal remains native64 FG-ARI>=0.90 with both IoUs reported, plus eventual
70k confirmation when justified. A TRAIN-loss decrease is not mask improvement.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `2e95f8ee70d203da62037f1f9347d69038d1fb76a1bb7560b131095386459577`

````markdown
# SW0192 native64 incumbent continuation

New primary objective: native64 FG-ARI>=0.90. Start from the exact learned
SW0186 seed0 checkpoint, retaining its graph/encoder/gates/spikes/QCE direction.
Two TRAIN-only native64 proposals first measure actual backward memory, timing,
five-family credit and same-RNG strict-descent acceptance. This is not a new
performance result. See DESIGN_KO.md for the staged64-image mini comparison and
subsequent full320/three-seed confirmation requirements. Joint70k is deferred
until a large, replicated gain justifies its cost.

Local and server9 tests passed, including both-axis learned-state transfer,
native64 encoder gradient, exact geodesic value/gradient parity across memory
chunks, real Adam acceptance/rollback transactions and nested recurrent-device
allocation. Both actual CLI entrypoints passed. Server CPU admission verified
the exact trained SW0186 seed0 weights, preserved learned CNN, native64 mapping
and empty five-family Adam before CUDA. The exclusive-GPU probe supervisor is
registered and verified live (PID1828532). Its actual worker PID1831863 started
on physical GPU2 and saved the initial native64 boundary. GPU process and actual
argv evidence are in results_archive/start_v1/actual_GPU_start.json. Both proposals
subsequently passed actual-spike/five-family gradient credit, full-forward parity,
strict descent, CNN and Adam checks. Mean B4 proposal time was444.77 seconds;
peak allocated memory5.01GiB. See RESULT_KO.md and results_archive/completed_probe_v1.
Mask quality remains unproven here; SW0193 runs the32-image before/after screen.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `a1ee59f42664369d5c0726eceec86d7fd199b93d595a42a676b8aeaa1c89a60b`

````markdown
# SW0192: native64 실제 학습 검증 완료

SW0186 seed0의 learned RGB/context encoder, graph, core, integration을 native64로
옮겨 TRAIN8 이미지에 두 B4/micro1 QCE proposal을 수행했다. 모두 같은-RNG 실제
full-forward loss 감소 기준을 통과했다.

| proposal | 같은 batch 변경 전 loss | 변경 후 loss | 전체 시간 |
|---|---:|---:|---:|
| 1 | .288873 | .284535 | 445.05초 |
| 2 | .235884 | .227845 | 444.50초 |

두 행은 서로 다른 TRAIN batch다. 행 간 loss 차이는 학습곡선이 아니다.
실제 emitted-spike gradient와 다섯 parameter family의 gradient/proposed delta가
모두 유한하고 양수였다. 실제 Adam step2, 원본 CNN 변경, 최종 모델 hash와
완료 boundary/history도 검증했다.

- 평균 B4 proposal: 444.77초 (약7.41분)
- 두 proposal 합계: 약14.83분, 초기화/큐 대기 제외
- peak CUDA 할당 메모리: 5,374,827,008 bytes (약5.01GiB); driver 전체 점유와 다름
- gradient forward/backward: batch당 약286.8초
- 같은 조건 TRAIN64/16-proposal 단순 예상: 약1.98시간/seed

시간 예상은 두 proposal 실측을 일정하게 외삽한 값이다. 큐 대기·평가·연산 변동은
포함하지 않는다. 앞선 단일 추론 기반 예상보다 구체적인 학습 실측이며, 연산
최적화 후에는 다시 측정해야 한다.

이 결과는 학습 작동/자원 검증이며 mask 성능이나 gating 인과성 증명이 아니다.
70k 학습은 예약하지 않았다. SW0193이 같은32 development 이미지에서 추가 학습
전후 mask 성능을 측정한다.

최종 모델 SHA256:
`2dd2b0d64602b85d4082bcaddcd21dfe8c60d34af1ea78a2ed29bf03c8ba4b6e`

완료 boundary SHA256:
`14d0386db64d3890163574ac2421b1e80d3100536ec25c9940434e0689327e82`

보고서와 검증 증빙은 results_archive/completed_probe_v1/에 보관했다.

````

</details>
