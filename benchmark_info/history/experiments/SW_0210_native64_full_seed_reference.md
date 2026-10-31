# SW_0210_native64_full_seed_reference

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `32405a2a6129c9730b9e5f92debd174dc13d35891eefb4a2bb51998a0ac60787`

````markdown
# SW0210: native64 기준 모델의 독립 세 seed 전체320 평가

208의 warm192 seed0 .877709/.791224/.696284만으로 세 seed 평균 승리를
주장할 수 없다. 같은 두 B4/native64 TRAIN8 적응 recipe의 독립 seed1/2,
즉 완료된205 checkpoint들을 전체320으로 평가한다. 205의 mini 개선 gate는
실패했으며 이 실험은 그 실패를 구제하거나 학습 개선으로 재분류하지 않는다.
현재 선택한 공통 baseline의 seed 편차와 절대 성능을 확인하는 평가다.
seed별로 더 좋은 checkpoint를 골라 섞지 않는다.

Seed0는208의 검증된 실제320개 warm 예측을 그대로 재사용한다.
seed1/2 각각은 canonical128 RGB/no crop/IDs1320..1639/native64/1024steps,
settle512, 기존 readout을 사용한다. RNG는 전 seed에서193000+image offset.
독립성은 학습된 모델 seed의 독립성이며 inference noise는 paired하다.
Source205 mini16과 겹치는 새 mask32개는 정확히 재현해야 한다.

추가 학습은 없다. 두 seed supervisor는 각각4개의80-image shard를 순차
실행하고 매 shard마다 shared exclusive GPU lease를 해제한다. GPU0..3에
대기하고 외부 사용자가 들어오면 같은 행별 checkpoint에서 재개한다.
긴 평가가 하나의 GPU lease를 수 시간 연속 차지하지 않도록 분할했다.

모든960개 우리 mask와 Slot 세 seed를 검증한 뒤 GT를 읽는다. 2px modal,
동률 작은ID, 같은 세 metric 및 image-paired bootstrap(10000회, seed210).
모든 세 지표의 우리3seed 평균이 Slot3seed 평균을 초과하는지와 FG평균≥.90을
별도로 기록한다. Supervised, inherited70k lineage/additional native64 TRAIN8,
validation-selected 모델이다. 새70k 학습, unsupervised 승리, scaling 또는
gating 인과 기여의 증거로 사용하지 않는다. 209 학습 모델은 섞지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `7f595f314cffea8deafda0638d440bd04bbca13582c9b1a37a72a11255d5a084`

````markdown
# SW0210 full native64 independent seed reference

Registered after8 local+8 Linux tests, AST/CLI and actual independent
model/full320 RGB/mini-reference CPU admission. Completes the missing seed1/2 fixed320
reference alongside actual208 warm192 seed0. No209 candidate, training,
checkpoint selection by seed or70k expansion is included. See DESIGN_KO.md.

Supervisors2252470(seed1)/2252471(seed2) verified live. Actual seed2
worker2252587 verified in /proc and nvidia-smi on GPU3; seed1 initially waits.
Each supervisor executes four80-image shards, releasing its exclusive GPU
lease between shards. GPU0..3 eligible; foreign owners retain priority.
Registration and actual process receipts are archived. Read-only waiter2253239
tracks these exact supervisor identities and automatically archives the final
three-seed summary and all eight immutable protocols after GT-last scoring.
No final result, three-seed win or .90 claim exists at registration.

After209 completed and released its lease, seed1 worker2258383 was independently
verified on GPU1 while seed2 worker2252587 continued on GPU3. Exact /proc commands
and NVIDIA observations: results_archive/observed_v1/actual_workers.json.

Completed and independently recomputed: full320/native64 three-seed mean
ours .8486147855/.5925328333/.6720472673 versus comparable Slot
.7869016920/.1982007892/.1952025237. All three strict mean comparisons pass;
the newer FG-ARI>=.90 target remains false. No new training or fresh70k run.
All1920 mask endpoints, bindings, dual metrics and paired CIs were revalidated.
The evaluated models include mask-supervised adaptation. See RESULT_KO.md,
results_archive/completed_v1/completion_audit.json and selected_reference_v1.json.

Posthoc full320 support attribution also completed without model forwards:
seed1 is the weakest full FG reference, while seed2 misses only3.70% of true
foreground but retains split/merge errors. GT-assisted pure-group references
are .956939/.941650/.976018; these are not predictions or learnable ceilings.
See SUPPORT_DIAGNOSIS_KO.md. No change to the real .848615 mean or .90 verdict.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `318ca4e748492a3dfdb41300b441674e188ae298ab1ff4ef3320abe12bf1310a`

````markdown
# SW0210: 전체320·64×64·독립 세 seed 평가 완료

동일 canonical128 RGB/no crop, IDs1320–1639, 2px modal GT와 기존 고정
readout에서 비교했다. 우리 모델 960개와 Slot960개 mask를 모두 검증한 뒤
GT를 읽었다. 별도의 읽기 전용 재검증도 모든 mask·평균·paired CI를 재계산해
일치했으며, 독립 metric 구현 간 최대 차이는2.22e-16이다.

| 모델 | seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|---:|
| 우리 warm192/205 | 0 | .877709 | .791224 | .696284 |
| 우리 warm192/205 | 1 | .825583 | .517168 | .660796 |
| 우리 warm192/205 | 2 | .842552 | .469206 | .659062 |
| **우리 평균** | **0/1/2** | **.848615** | **.592533** | **.672047** |
| 비교 Slot | 0 | .636265 | .205508 | .184986 |
| 비교 Slot | 1 | .846161 | .197193 | .212411 |
| 비교 Slot | 2 | .878278 | .191901 | .188211 |
| **Slot 평균** | **0/1/2** | **.786902** | **.198201** | **.195203** |

세 평균 모두 Slot보다 높다. 차이는 FG+.061713 / FIoU+.394332 /
ObjectIoU+.476845이다. Paired image bootstrap95% CI는 각각
[.044403,.078783], [.377264,.412516], [.461945,.491677]이다.
이는 세 고정 모델의 이미지별 평균을 resample한 구간이며 seed 모집단에
대한 불확실성 구간은 아니다.

이번 결과는 16장 mini의 .90과 달리 전체320장의 세 seed 결과다. 그러나
**FG-ARI 평균 .90 목표는 아직 미달**이다. seed0가 현재 세 모델 중 가장
높지만 각 seed의 최고 설정을 따로 골라 평균한 결과는 아니다. 공통 TRAIN8
적응 recipe를 사용했고 seed0는 완료된208 결과를 재사용했다.

이 모델들은 기존70k core 학습과32-grid 학습을 물려받고, mask GT를 사용한
supervised 적응을 포함한다. Native64 추가 고유 TRAIN은 seed당8장이다.
210에서는 학습 update0이며 새 native64 TRAIN70,000 학습을 수행하지 않았다.
비교 Slot은 기존154/208에 고정된 reference이며 공개 checkpoint의 과거
.8946 수치와 평가 조건을 섞지 않는다. 같은 평가 계약에서의 결과 비교이며
동일한 supervision과 학습 비용을 맞춘 우월성 실험이라고 주장하지 않는다.

320장은 반복 사용한 validation이다. 데이터 scaling, untouched test 일반화,
모든 gating의 인과 기여를 이번 평가로 입증하지 않는다. 기존205의 작은
적응 전후 비교가 실패했다는 결론도 바꾸지 않는다. 이번210은 선택된 공통
recipe의 절대 성능을 더 넓은 평가에서 확인한 것이다.

이 결과를 native64 세 seed 기준점으로 보존한다.211은 spike/membrane에서
center·offset을 학습해 grouping을 개선하며,212는 동일 head의 학습 전경을
결합하는 별도 비교다. .90 목표를 위해 goal은 활성 상태로 유지한다.

증빙: results_archive/completed_v1/score_v1/summary.json,
completion_audit.json,8개 shard protocol과 queue 기록.
Checkpoint 식별자는 results_archive/selected_reference_v1.json에 기록한다.

````

</details>

## SUPPORT_DIAGNOSIS_KO.md

<details>
<summary>SUPPORT_DIAGNOSIS_KO.md</summary>

Source SHA256: `4004fc97bb34589fea551009d0fc1deb6cb2c074fe60c39c1d24fddf97fa380a`

````markdown
# 전체320에서의 잔여 오류 진단

새 모델 점수는 아니다. 완료된210의 동일960개 예측에 정답을 사용해
묶음/전경 오류를 구분했다. 모델 forward나 학습을 추가하지 않았다.
기존208 seed0 진단을 정확히 재현하고 같은 분석을 전체 seed1/2로 확장했다.

| seed | 실제 FG-ARI | GT 전경 누락률 | 포착된 같은 물체 patch의 split률 | 포착된 다른 물체 patch의 merge률 |
|---|---:|---:|---:|---:|
| 0 | .877709 | 6.37% | 2.97% | 4.18% |
| 1 | .825583 | 8.12% | 9.05% | 4.04% |
| 2 | .842552 | 3.70% | 6.29% | 6.29% |

Split/merge는 예측 전경에 포함된 정답 foreground patch 쌍에서 계산한다.
이미지별 비율의 평균이며 전체FG-ARI 하락을 독립 성분으로 분해한 수치는
아니다. Foreground IoU는 실제210 결과와 같은 .791224/.517168/.469206이다.

정답을 알고 원래 예측 전경 안의 patch를 순수한 정답 그룹으로 다시 묶은
참고 FG-ARI는 seed0/1/2 각각 .956939/.941650/.976018이다. 이는 GT에
의존하는 구체적 partition 예시다. 모델 점수나 학습 가능한 상한이 아니다.
누락된 정답 전경 patch를 기존 그룹에 정답으로 연결하는 별도 참고값은
.906102/.863455/.856483이다. 두 진단의 개선량을 더할 수 없다.

전체320에서도 묶음 오류가 남아 있으므로211의 center/offset 학습을 계속
확인한다. Seed2는 전경 누락이 비교적 작아 기존 전경을 보존한 grouping
시험의 근거가 유지된다.212는 같은 head의 학습된 전경을 적용하는 독립
비교이며211 결과를 보고 threshold를 고르지 않는다.

전체 평가에서 FG-ARI가 가장 약한 seed는1이다. 이는 고정16 mini에서 가장
약했던 seed2와 다르다.211/212가 유효하면 seed1 독립 모델에서 재현되는지
확인해야 한다. Mini의 순위를 전체 seed 순위로 일반화하지 않는다.
현재 .848615 세 seed 평균 기준점과 .90 미달 판정은 그대로 유지한다.

증빙: results_archive/support_diagnosis_v1/summary.json.
CPU에서 두 독립 metric 구현의 최대 오차4.44e-16, 추가 학습0/forward0.

````

</details>
