# SW0251 모델/benchmark 계열

## readout_benchmark/DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `b05eaee4c2151b0336b11d6c66c903a60daad01ae1033ccd63e72784ebf32071`

````markdown
# SW0251: 배경 결정과 최종 mask 감독의 통제 비교

사용자 요청: 먼저 원래 CLEVR 데이터에서 기존 240 수준의 FG-ARI를 유지하는지 검증한다.
통과한 설정만 Multi-dSprites / Tetrominoes에서 추가 학습 없이, 추가 학습 후 모두 평가한다.
원본 240 checkpoint, 기존 실행 코드, benchmark/ 및 benchmark.md는 수정하지 않는다.

## 가설과 비교

- baseline: 검증된 기존 SW0249 matched35000 terminal SW240 seed0/1/2.
- keep_groups: spike synchrony threshold .5, min size32는 그대로 두고 가장 큰 그룹만 제거하던 규칙을 끈다.
- all_support: 모든 patch를 signed partition에 전달한다. 배경을 지운 것이 아니라 제거 효과를 분리하는 진단이다.
- learned_background: 실제 spike/membrane 특징에서 배경 확률을 학습하고 all_support의 물체 분할을 사용한다.
- residual_mask: 기존 240 물체 masks와 all_support masks를 GT 없이 후보로 만들고, 실제 spike/membrane 특징으로
  최종 soft mask 및 배경 출력을 학습한다. 초기 argmax는 기존 240과 정확히 같다.

Readout 입력은 settle512 이후 실제 네 component의 spike 32-bin 평균/평균/표준편차와
membrane 평균/표준편차뿐이다(144개 특징). RGB, gamma, 정답 object count는 readout 입력이 아니다.
기존 Kuramoto→sinusoidal→dendritic→membrane→spike core는 유지한다.
후보 masks는 기존 모델 출력에서만 생성되며 GT는 학습 loss 및 별도 scoring에만 사용한다.
Readout 후보 수는 모델 출력에 따르며 GT 물체 수로 정하지 않는다.

첫 학습은 core/encoder를 모두 고정하고 readout만 학습한다. 이는 **최종 mask 감독이 readout에 직접 연결되는 비교**이며,
전체 encoder/core까지 mask loss로 end-to-end 학습했다고 주장하지 않는다. 원래 240 보존 및 병목 분리를 우선한다.

Loss: Hungarian matching으로 object ID 순서에 무관한 최종 mask balanced CE + soft Dice,
별도 배경 BCE, foreground same/different pair loss, 기존 partition 내부 일관성 anchor(보존 항).
Hard grouping/seed 선택 자체에는 역전파하지 않지만 soft assignment와 학습형 readout에는 실제 gradient가 흐른다.

## 데이터와 승격

CLEVR의 원래 canonical TRAIN70000 cache를 사용하며 reserved IDs1000..1639는 학습에서 제외한다.
작은 screen TRAIN64 / validation16(1000..1015)에서 시작하고, 유망하면 TRAIN256 / validation64(1000..1063)로 확대한다.
TRAIN은 deterministic subset이며 모든 후보에서 동일하다. 초기 screen과 승격 확인 모두 seed0/1/2를 사용한다.
1024 rollout / settle512 / native64는 유지한다. 70000장 전체 재학습은 이번 초기 비교에 필요하지 않다.

최종 보존 확인은 기존 fixed320 IDs1320..1639, seed0/1/2와 동일한 patch metric이며
mean FG-ARI(new)-mean FG-ARI(original240) >= -0.005가 수치상 승격 조건이다.
이 fixed320은 과거에 반복 사용된 validation이지 untouched test가 아니다. 평균만으로 유의한 동등성을 주장하지 않고
seed별 값·paired image 차이·bootstrap CI와 나머지 두 지표도 함께 보고한다.
학습/설정 선택은 TRAIN/새 validation에만 의존하며 fixed320으로 loss sweep하지 않는다.

새 데이터는 SW0250과 동일한 원본 benchmark 평가 규칙, 기존 splits와 seeds0/1/2를 유지한다.
Multi-dSprites(colored_on_colored) grid64, Tetrominoes native64→원본 modal grid32→35pixel adapter.
Zero-shot은 해당 데이터 mask를 한 번도 학습하지 않은 CLEVR 개선 모델/readout을 사용한다.
Fine-tune은 같은 seed의 CLEVR 개선 모델에서 출발하며 SW0250의 encoder/projection TRAIN6000×10 적응 후
해당 adapted core에서 추출한 TRAIN spike 특징으로 readout도 mask 감독한다. TEST/validation mask는 학습에 넣지 않는다.
이렇게 추가된 readout 적응 비용은 별도로 명시하고 SW0250과 완전히 동일한 학습량이라고 주장하지 않는다.
TEST1000의 결과는 우선 고정 앞360장, 이후640장을 모두 완료해 보고한다.
모든 새 데이터 최종 평가 전에 원본 preflight.py와 보호 파일 SHA 검사를 실행한다.

## 실행/기록

생성물은 patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0251_background_mask/ 및
서버 collaborative_testing/generated/SW0251_background_mask/에 기록한다.
Data0의 여유 공간 부족 및 export_home의 stale file handle로 큰 활동 캐시는 본인
/work/USERS/kevinswk/collaborative_testing/generated/SW0251_background_mask_cache/에 저장하고 실험 archive/cache에서 연결한다.
4MB 파일의 fsync·rename·독립 재읽기 SHA 검사를 통과했다. 실패한 홈 저장소 시도의 기록/파일과 원본 데이터/기존 결과는 보존한다.
서버 파일 전송 전 CPU 기능 검사, checkpoint 복원/forward smoke, 실제 CUDA 학습 gradient 검사를 진행한다.
GPU 확보는 타인의 프로세스를 건드리지 않고 실제 여유 메모리에 따라 공유 실행한다.
큐는 서버에서 지속 실행하고 원격 연결 확인 자체만을 위한 사용자 메시지는 보내지 않는다.
연결 복구/실패만으로 학습을 중복 시작하지 않는다.

````

</details>

## readout_benchmark/ORIGINAL_FIXED320_RESULTS_KO.md

<details>
<summary>ORIGINAL_FIXED320_RESULTS_KO.md</summary>

Source SHA256: `cc82f0dbb8e4d26062d67e19f4bfbbebce0617dc5f68da9b5282c377eec54657`

````markdown
# SW0251: 기존 240 성능 유지 검증 통과

64×64 패치, 기존에 사용한 고정 평가 320장(ID 1320–1639), seed 0·1·2 평균이다.
이 평가 집합은 재사용한 validation이며, 새로운 미사용 test 결과가 아니다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원본 240 | 0.889187 | 0.754775 | 0.686438 |
| 최종 mask 감독 residual readout | 0.889629 | 0.784008 | 0.707530 |
| 차이 | +0.000442 | +0.029233 | +0.021092 |

사전 조건인 평균 FG-ARI 절대 감소 0.005 이내를 통과했다. FG-ARI 차이의
paired-image 95% CI는 [−0.000673, +0.001409]이므로 확실한 FG-ARI 향상은
주장하지 않는다. Foreground IoU 차이 CI는 [+0.022518, +0.036138],
Object IoU 차이 CI는 [+0.016573, +0.026087]이다.

| seed | 원본 FG-ARI | 수정 FG-ARI |
|---|---:|---:|
| 0 | 0.891040 | 0.893264 |
| 1 | 0.897328 | 0.896829 |
| 2 | 0.879194 | 0.878795 |

후보 선택에는 별도 TRAIN256/VAL64만 사용했다. frozen 240의 실제 spike·membrane
활동을 입력으로 받는 readout head에 Hungarian mask CE·Dice 및 보조 loss를 적용했다.
encoder·core의 end-to-end mask 학습 결과가 아니다. 기존 VAL64의 기전 분석에서는
물체 간 재배정보다 전경/배경 support 변경이 개선의 주된 원인이었다.

원시 예측 1,920건(원본·수정 각각 320×3)을 독립 NumPy contingency ARI 및
Hungarian IoU 계산으로 재검증했다. 원본 mask는 기존 생산 예측과 일치하고,
240 checkpoint 3개 SHA와 보호된 benchmark 파일의 불변성을 확인했다.

증거는 비공개 실행 기록 SW0251_background_mask의
`collected/original_confirmation/result.json` 및
`original_gate_independent_audit.json`에 있다.

Multi-dSprites grid64와 Tetrominoes grid32의 세 seed zero-shot/추가학습 평가는
후속 단계이다. 앞선 seed0 TEST8 미리보기에서는 Multi-dSprites FG-ARI가 하락했으므로,
이 기존 데이터 검증만으로 배경 문제가 데이터셋 전반에서 해결됐다고 주장하지 않는다.

````

</details>

## readout_benchmark/RESULTS_FINAL_KO.md

<details>
<summary>RESULTS_FINAL_KO.md</summary>

Source SHA256: `90b1508dd1db96985205600f004a26828746d69dd1148262dfe6b88ad16e48eb`

````markdown
# SW0251 ?? ?? ?? ??

?? ??? residual_mask ??? ?? ??? ?? ?? ? ? ????? 12? ?? TEST1,000? ??? ????. ?? ???? ??? ?? ?? ??? ???? ??252?? ??? ????.

## ?? CLEVR ??320?? seed ??

??240: 0.889187 / 0.754775 / 0.686438. 251 residual_mask: 0.889629 / 0.784008 / 0.707530. ???FG-ARI / Foreground IoU / Object IoU??. ??FG-ARI ??+0.000442? ?? ??0.005 ?? ??? ????. ????validation ???? ???test? ???.

## TEST1,000??? seed ??

| ???? | ?? | FG-ARI | Foreground IoU | Object IoU |
|---|---|---:|---:|---:|
| Multi-dSprites | ?? ?? ?? | 0.667286 | 0.385668 | 0.721929 |
| Multi-dSprites | ?? ?? ? | 0.769756 | 0.795591 | 0.779550 |
| Tetrominoes | ?? ?? ?? | 0.667404 | 0.344925 | 0.669972 |
| Tetrominoes | ?? ?? ? | 0.838224 | 0.728408 | 0.828759 |

?? ??? ?? ??? ????251?? ? ????? ?? ??? ????? ????? ???? ???. ?? Multi-dSprites zero-shot ??? ??240??FG-ARI?Foreground IoU? ??.

? ??? ??spike/membrane144?? ??? ???? ?? ??? ???? head??. Encoder/core? ??mask loss? end-to-end ??? ??? ???. ??240 ??? mask-supervised??.

?? ? ???? ??? ??TEST ID7000?7999? ????. Multi-dSprites?grid64/original64 pixel, Tetrominoes?modal grid32? original35 pixel? ????. ??benchmark ??? ???? ???. ????? ??SW0250 encoder TRAIN6000?10(3,750updates)? ???? ? ??readout?TRAIN256?20(5,120updates) ????.

???? ?? ??? ??? **252?? ?? ?? ??? ??** ??? [?? ???](../neutral_readout/RESULTS_FINAL_KO.md)? ??? ????.252?? ????? ??ID? ????Foreground IoU ??? ? ?? ???.

??: ???SW0251_background_mask? `collected/benchmark_results_1000.json`, `original_gate_independent_audit.json`, `completion_audit.json`. ????? ?? artifact? ??? ????.

````

</details>

## readout_benchmark/VALIDATION_RESULTS_KO.md

<details>
<summary>VALIDATION_RESULTS_KO.md</summary>

Source SHA256: `f82f4e87be91ca93eb042d0e4a9ee720e71319df6b25e348b42e674bdc2c57ca`

````markdown
# SW0251 배경 처리·최종 mask loss 통제 비교

아래는 **TRAIN256, 별도 validation64, seed0·1·2 평균**이다. 고정320 보존 검증이나 새 데이터셋 결과가 아니다. 동일한 실제 SW240 activity를 재사용하고, head마다20epoch/5120 Adam updates를 수행했다. 원래 encoder/core/graph/readout 파라미터와 source checkpoint는 보존했다. 학습한 새 head는 실제 spike·membrane의144채널 특징을 입력으로 사용한다.

| 설정 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---:|---:|---:|
| 원래240 | .881432 | .793802 | .695344 |
| 기존 큰 component도 유지 | .885387 | .198595 | .702056 |
| 전체 영역 partition | .885710 | .200268 | .732342 |
| 배경 분류만 학습 | .874705 | .751942 | .669335 |
| 직접 최종 mask loss를 학습한 residual head | **.885416** | **.815043** | **.712711** |

큰 component 유지/전체 영역 설정은 전경을 매우 넓게 예측한다. FG-ARI만 높다고 배경 인식이 좋아졌다고 해석하면 안 된다. 배경-only head는 이 validation에서 원래240 대비 FG-ARI가 .006727 낮다. 두 학습 설정 중 FG-ARI가 높은 residual_mask를 기존 사전 정의된 기준으로 선정했다.

| seed | 원래240 FG-ARI | residual_mask FG-ARI |
|---|---:|---:|
| 0 | .880402 | .882063 |
| 1 | .891204 | .901498 |
| 2 | .872689 | .872686 |

Residual head의 평균 변화는 FG-ARI +.003984, foreground IoU +.021240, object IoU +.017367이다. 직접 Hungarian mask CE·Dice, 배경 BCE, 전경 pair 및 partition anchor loss의 gradient는 **새 readout head에만** 전달한다. Encoder/core까지 end-to-end mask loss로 학습한 실험이 아니다.

## 실제 변경 원인 확인

선정된 checkpoint를 CPU에서 validation64 전체에 다시 적용해 위 점수를 재현하고, 기존 물체 prototype의 ID 대응을 보정한 뒤 변경 patch를 집계했다. 고정320 정답은 열지 않았다.

| seed | 배경에서 전경으로 | 전경에서 배경으로 | 기존 전경의 물체 prototype 변경 |
|---|---:|---:|---:|
| 0 | 1777 | 3 | 1 |
| 1 | 1539 | 0 | 0 |
| 2 | 331 | 10652 | 7 |

배경/전경 변경만 적용하고 기존 전경의 물체 assignment를 유지한 counterfactual의 FG-ARI는 .882068/.901498/.872686으로, 실제 선정 모델과 사실상 같다. 따라서 이번 개선은 주로 전경 support 보정이며, 물체 간 재분류 또는 gating의 인과적 기여가 향상됐다는 증거로 사용하지 않는다. 원본 근거는 selected_mask_change_audit.json이며 selected checkpoint SHA도 검증했다.

## 후속 검증 완료

- 251번의 고정320 원래 mask 재현과 성능 유지 검증을 통과했다. 평균FG-ARI는 원래240의 .889187에서 .889629로 유지됐다. 이320장은 재사용한validation으로 untouched test가 아니다.
- 이후 사용자의 요청에 따라 배경 역할 고정과 영역 삭제를 없앤252번을 별도로 구현했다. 기존 데이터 고정320 평균FG-ARI는 .892326이며 원래240보다 절대 .005 이상 감소하지 않는 조건을 통과했다.
- 252번의 Multi-dSprites grid64 및 Tetrominoes grid32 zero-shot/추가 학습 후 세 seed TEST360 및 동일TEST1000 평가와 독립 검증을 모두 완료했다.
- 최종252 결과와 Foreground IoU 해석상의 차이는 [최종 보고서](../neutral_readout/RESULTS_FINAL_KO.md)에 기록했다. 별도 비지도 최고165번의 두 데이터셋 전이 평가도 완료했다.

최초251 후보 선정은 validation64만 사용했고 고정320 activity는 정답을 열지 않고 미리 준비했다. 작은 TRAIN64/validation16 실험과 평가 이미지 수가 다르므로, 두 표의 평균 차이를 학습 데이터 증가 효과로 해석하지 않는다. 이 문서의 비교 표는 당시251 validation 단계의 기록이며,252의 배경 비고정 모델과 구분한다.

원본 근거: SW0251_background_mask/collected/validation_TRAIN256.json, validation_selection.json, h_cv_z_seed*_*.json 및 collected_manifest.json의 서버 SHA 검증 기록. Benchmark Python 파일과 source240 checkpoint는 변경하지 않았다.

````

</details>

## readout_refinement/DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `7eef9bc8fb165da0519d600e0bb19f34c9a169961ef05aa224f16e50216ad2e6`

````markdown
# SW0251 readout score calibration

실제 TRAIN16/validation16의 네 loss 비교에서 기존 positive group끼리 재배정된 patch는 0개였다.
보존 prior4와 cosine logit의 bound(2×scale)가 원인이다. 실제 best scale .56이면 최대 점수차1.12로 prior4를 넘지 못한다.
따라서 FG-ARI 유지가 물체 분할 학습 성공을 뜻하지 않는다.

기존 core, cache, 정답/예측 생성 분리, final mask loss, LR, 데이터 순서,20 epochs를 유지한다.
control(prior4,gain1)과 soft_prior(prior1,gain1), strong_metric(prior4,gain16)을 비교한다.
세 설정 모두 초기 hard mask는 기존240과 정확히 같지만 두 새 설정은 작은 scale에서도 재배정할 수 있다.
매 epoch validation FG-ARI와 실제 positive→다른 positive 재배정 patch 수를 함께 기록한다.
초기 작은 비교가 통과해도 원래 fixed320·seed0/1/2 평균 FG-ARI 감소 .005 이내라는 최종 보존 gate는 유지한다.
이 실험의 결과로 기존 240 또는 benchmark 파일을 덮어쓰지 않는다.

````

</details>

## readout_refinement/SCREEN_RESULTS.md

<details>
<summary>SCREEN_RESULTS.md</summary>

Source SHA256: `2bdba860145bc6c061cd94bead9653abb836d8fdf504da3c04cf00187b152b2d`

````markdown
# SW0251 controlled readout screen

This is a small validation experiment, not the fixed320 preservation gate or a new-dataset benchmark result.
Same actual SW240 activity, TRAIN64, validation IDs1000–1015, 20 epochs, 1280 Adam steps, identical final-mask loss.
Core and encoder remain frozen. Benchmark code and original checkpoints are preserved.

| Readout | Seed0 FG-ARI | Seed1 FG-ARI | Two-seed mean | Mean foreground IoU | Mean object IoU |
|---|---:|---:|---:|---:|---:|
| Original SW240 | .880281 | .882659 | .881470 | .815321 | .711661 |
| Soft prior (1), metric gain 1 | .860626 | .881055 | .870840 | .708322 | .682559 |
| Original prior (4), metric gain 16 | .882094 | .884013 | .883053 | .814160 | .723516 |

Best epochs were selected by validation FG-ARI independently for each seed and configuration.
Soft prior loses .010630 mean FG-ARI and is rejected at this screen; it is not being extended to seed2.
Strong metric changes mean FG-ARI by +.001583 on these two seeds. Its seed2 confirmation is queued after the exact source cache is complete and GPU memory is admitted.

The selected strong-metric checkpoints have **zero positive-group-to-other-positive-group reassignments** on validation.
Their improvement comes from changed background/foreground assignments. Other epochs can change object assignments, but this experiment has not demonstrated that these changes improve validation results.
Do not describe this as a successful instance reassignment learner or a completed original-data preservation gate.

The original three-seed v1 comparison continues independently. No configuration is promoted to the new datasets on the basis of this screen alone.
Authoritative reports are SHA-verified copies under SW0251_background_mask/collected/rf_seed*_*.json; the baseline and queue receipt are in refinement_seed2_queue_launch.json.

## Completed three-seed screen

Seed2 strong-metric training also completed all1280 steps; best epoch6, validation FG-ARI .826703, foreground IoU .795725, object IoU .656926. Its selected checkpoint likewise has zero positive-group reassignments. The full same-data v1 comparison is now complete:

| Readout | Three-seed mean FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Original SW240 | .863203 | .794869 | .695189 |
| Learned background only | .854945 | .623144 | .661761 |
| V1 final-mask supervised residual | .864096 | .814651 | .701184 |
| Strong metric gain16 | .864270 | .808015 | .701320 |

The gain16 head exceeds the ordinary residual by only .000174 mean FG-ARI, with lower foreground IoU. This small validation study does not justify adding it to production selection or extending its training. The unchanged planned v1 TRAIN256/VAL64 comparison continues across all three seeds. No new refinement model is promoted, no fixed320 gate is claimed, and no new-dataset result is inferred.

The background-only head loses .008259 mean FG-ARI at this screen. The direct final-mask residual retains FG-ARI and improves the other two metrics here. These observations support continuing that controlled comparison, without establishing generalization or an object-reassignment benefit.

````

</details>
