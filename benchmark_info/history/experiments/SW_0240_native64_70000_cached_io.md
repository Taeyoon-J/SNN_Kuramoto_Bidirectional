# SW_0240_native64_70000_cached_io

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `1d7dc5c1c39a321f1b5544e14c8fc9a84a2a4b5bbdd1c30c198587ae644fc315`

````markdown
# SW0240: 동일한 70,000장 학습의 입력 I/O 개선

239번은 실제 GPU gradient·native64 forward 및 첫 Adam checkpoint를 검증했다.
그러나 무작위 batch를 HDF5에서 읽을 때 이미지 축에 큰 chunk로 압축된 데이터를
반복 해제하는 CPU 병목이 확인됐다. 모델이나 loss를 바꾸는 실험이 아니다.

## 입력 캐시

cache.py는 기존 239번 data.py의 canonical RGB와 modal patch label 함수를 그대로
사용한다. source chunk에 맞춰 **TRAIN 구간 0..999와 1640..70639만** 순차적으로
읽는다. 예약 평가 구간의 mask는 캐시에 넣기 위해서도 읽지 않는다.

RGB는 uint8 그대로, patch label은 손실 없이 uint8로 저장한다. batch 반환 시
label을 기존 int64로 복원한다. 원래 TRAIN ID에서 cache 위치로만 매핑하며,
seed별 shuffle과 batch 구성 및 이미지·mask tensor는 바꾸지 않는다. 모든 block의
tensor를 저장 후 확인하고 파일 hash와 완전 사용 범위를 완료 manifest에 남긴다.
캐시는 생성 artifact이며 Git에 올리지 않는다.

로컬 검사 2개가 전체 TRAIN ID 완전 포함·평가 구간 배제·chunk 정렬·무작위 순서의
batch identity를 확인했다. 실제 전체 캐시 생성과 원본 batch 대비 bitwise 비교 및
읽기 시간 측정은 별도로 수행해야 한다.

## 이어서 학습할 때 필요한 조건

239번의 등록된 코드·selection·checkpoint를 몰래 바꾸거나 reader를 숨겨서
교체하지 않는다. 동일 학습의 명시적 I/O successor로 출처를 기록한다.
최고 확정 모델 227번에서 시작한 기존 239번 초기 모델·teacher·loss·learning rates·
seed 순서·70,000장 10 passes·총 43,750 logical updates·readout·평가 320장도 유지한다.

239번은 캐시와 successor가 검증될 때까지 실행을 유지한다. 전환할 때는 본인 239번
supervisor/worker만 중단하고 마지막 원자적 checkpoint를 동결한다. 그 model·Adam·
RNG·이미지 사용 이력을 실제로 복원해, 학습을 처음부터 다시 하지 않고 이어간다.
수행되지 않았던 seed1/2는 기존 선정 모델에서 같은 계획을 시작한다.
동시에 이 70,000장 작업에 GPU 두 개를 쓰지 않도록 이전 worker 종료를 확인한다.

새 binding은 원래 239번 admission 및 보존 checkpoint의 SHA를 명시하고,
inherited update와 successor update를 구분한다. 합친 이력으로 각 pass 70,000장
완전 사용과 최종 Adam step 43,750을 검증해야 한다. 실제 checkpoint 재개 동등성과
단일 GPU 큐·서버 preflight 전에는 successor 학습을 시작하지 않는다.

Implementation status: the complete cache and file hashes were validated. Two actual TRAIN batches matched original RGB and int64 labels bitwise. Input reads were 3424-4666 times faster; this is not an end-to-end training speedup measurement. All seven local tests passed, including actual Adam checkpoint interruption/rebind matching uninterrupted model, encoder and history hashes.

Server Linux tests must pass before stopping the verified owned239 processes. Preserve the final committed model, Adam, RNG and history boundary, then atomically register migration provenance. All three actual CPU admissions must pass before launching the single-GPU successor queue. The original239 source and artifacts remain unchanged.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `42c4ec1ac15a2313117fe46e485cf0ab282557c04edf6520a456d5bf3b68355f`

````markdown
# SW0240: exact input-cache successor to SW0239

This continues the requested best-model training on 70,000 unique TRAIN images, followed by the fixed 320-image native64 evaluation and predetermined mask figures. It is an I/O correction, not a new model or loss hypothesis.

The original gzip HDF5 chunks span 3,125 images and small spatial regions. Random batches repeatedly decompress large chunks. The complete canonical TRAIN cache preserves RGB and modal labels, split, seed shuffle, teacher, optimizer, update budget and readout. Reserved evaluation masks are excluded from cache construction. Two actual original/cache batches match bitwise. The measured 3,424–4,666-fold improvement applies to input reads only.

Seven local and seven Linux tests passed. A real Adam interruption/rebind test matches uninterrupted model, encoder and history hashes. After Linux tests passed, only the verified owned239 supervisor and worker were stopped. Their final committed seed0 boundary at update128 was preserved, including model, Adam, RNG and complete TRAIN history. Seeds1/2 have not started. All three actual CPU admissions must pass before a single successor GPU worker can start.

Each seed still completes 43,750 logical updates and ten full passes through the same 70,000 unique images; inherited updates count toward that total. Seeds run sequentially on one GPU. Other experiment queues remain independent. The migration manifest and checkpoint bytes are private generated artifacts; aggregate verification is recorded in `MIGRATION_VALIDATION_EVIDENCE.json`.

All three actual CPU admissions subsequently passed. On the assigned GPU, inherited Adam128 resumed and committed Adam192, confirming 64 new successor updates. The final CPU raw-checkpoint/mask audit is attached. No new performance result is available yet. SW0227 remains the audited full320 incumbent. This training uses mask supervision and should not be presented as unsupervised learning.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `0e652e86a3596a095819ffa861a181c577a78eb3cd94077339bf7650a1d52565`

````markdown
# SW0240: native64 supervised full70k result

All three seeds completed 70,000 distinct TRAIN images, ten passes and43,750
new optimizer updates per seed. Evaluation uses the same fixed320 validation
images and native64 hard masks as the previous reference.

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| SW0240 three-seed mean | **0.888130** | **0.743444** | **0.684382** |
| Previous SW0237 mean | 0.872680 | 0.626110 | 0.676331 |
| Comparable Slot Attention | 0.786902 | 0.198201 | 0.195203 |

| Seed | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| 0 | 0.895956 | 0.828796 | 0.700791 |
| 1 | 0.894247 | 0.746766 | 0.688654 |
| 2 | 0.874186 | 0.654772 | 0.663702 |

All three means exceed Slot Attention; seed2 individually remains below Slot
seed2 in FG-ARI. The three-seed FG-ARI0.90 target is not met.

This uses TRAIN mask supervision; Slot uses reconstruction training. It is not
an unsupervised comparison. The source selected at registration was SW0227;
the subsequently completed SW0237 did not silently replace the initialization.
Only the encoder and sensory projection are adapted; other core/readout remain
unchanged. Neither all-gating causality nor a controlled data-size effect is
established: data amount and optimization budget both changed. These repeatedly
used validation images are not an untouched final test.

The user-requested shared GPU successor changed resource admission only; the
training objective, source fingerprint, Adam/history and masks were preserved.
Raw score and independent audit receipts are retained privately; public
aggregate evidence is in PUBLIC_RESULT.json once the independent audit passes.

````

</details>

## SHARED_GPU_POLICY.md

<details>
<summary>SHARED_GPU_POLICY.md</summary>

Source SHA256: `c34549c86ef5b05941878947688c817ebba8f0703160d4bf6d0f4eb2da8f0dbf`

````markdown
# User-requested shared GPU execution

On 2026-10-20 the user requested that an occupied GPU be used when sufficient
memory remains, and that our worker no longer yield merely because another
user's process appears. `queue_shared_v2` implements that resource policy.

Admission requires free memory for the unchanged worker's 80% device-memory
allocation cap plus 512 MiB headroom. Other process presence and utilization
do not exclude a GPU. Our existing research GPU lease still prevents duplicate
workers from these cooperating queues. After admission, foreign process arrival
does not terminate our worker. No other user's process is signalled.

The original worker, training source fingerprint, loss, dataset, optimizer,
checkpoint history, seed order and single-worker limit are unchanged. The
successor adopts the existing queue state only after the predecessor and any
recorded child are confirmed absent, under the same exclusive queue lock.
Original shutdown records are preserved. The original acquire loop reports an
intentional SIGTERM as an exception; its exact exit is archived separately in
the controlled handoff receipt before the explicit stopped state is supplied.

Six local and Linux tests cover memory admission boundaries, no yield on other
processes, device identity, live predecessor/child rejection, rejection of
unexpected failures, and sequential single-worker execution. Original workers
perform their existing CPU, runtime and actual GPU preflight before resuming.

This changes resource admission, not model performance. GPU time is shared;
free GPU memory does not guarantee exclusive compute throughput.

````

</details>
