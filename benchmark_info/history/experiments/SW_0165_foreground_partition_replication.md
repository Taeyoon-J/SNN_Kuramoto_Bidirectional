# SW_0165_foreground_partition_replication

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `874d973f443fa9bedb3a03e025238a5104dc223aa4dc3bcd8808784b5cc17d37`

````markdown
# SW0165 prospective — SW164 frozen-recipe seed0/2 replication

2026-10-14. **Conditional GO.** Root의 SW164 seed1 independent `validate_score` 완료 receipt와 frozen prediction/protocol hashes가 먼저 확인되어야 한다. 현재 보고된 seed1 candidate .770932/.614993/.630523 vs sourceT .755273/.614993/.579514의 긍정 신호를 복제하는 실험이다. 이 문서는 설계이며 서버/구현/queue를 변경하지 않는다. Sealed SW164 및 SW163는 불변이다.

## 변경하지 않는 과학적 recipe

동일 seed의 SW0097 source를 native16→32 strict spatial mapping한 SourceT,unchanged encoder/statistics/patcher,SW143 T power2/contrast8,actual emitted-spike Q,native prepare_coupling tuple[0] A,실제 per-source learned graph gain을 사용한다. B1,T1024,settle512,live_tail0,optimizer0이다. Threshold/emitter/graph weights나 energy coefficient를 바꾸지 않는다.

SW164 `foreground_signed.foreground_induced_merge`를 **그대로** 호출한다. SourceT labels>0의 induced4-neighbor support에서 singleton 초기화,unchanged full1024 signed energy,all-cross-pair gain,strictly-positive maximum merge,canonical first-patch ties를 사용한다. 모든 source foreground를 size1까지 유지한다. 새 min8 filter,largest-FG removal,nearest-group attachment/nonadjacency fallback,count cap을 추가하지 않는다. SourceT 단계의 native min8/largest-background policy는 그대로다.

SourceT와 candidate foreground membership의 exact equality,complete foreground coverage,각 final group의 induced4-connectivity는 모든 seed/image에서 필수다. ForegroundIoU는 per-image SourceT와 정확히 같아야 한다. 이 조건이 깨지면 새로운 tradeoff라고 해석하지 않고 technical failure로 멈춘다.

## Seed별 source와 proof

새 leaf runner에서 explicit seed parameter `(0,2)`를 처리한다. `SW130.run.source_contract(seed)`와 `SW135.foundation.load_native32_foundation(seed,device)`가 실제 generic API다. SW164/SW162의 public prepare/predict functions는 seed1/global output path에 묶여 있으므로 `module.SEED` monkeypatch나 문자열 치환으로 전체 pipeline을 재사용하지 않는다. 공유 pure solver/capture/trace-check 함수는 코드 fingerprint를 고정하여 재사용할 수 있다.

각 seed의 own source contract 순서에서 positions `[0,1,2,3,2048,2049,2050,2051]`를 고른다. Seed1의 image IDs/pool rows/model state/gain을 복사하지 않는다. TRAIN masks 없이 same-RNG untouched/capture eight-source parity와 source state unchanged,actual Q/A validity 및 empty/singleton/disconnected support invariants를 확인한다. Archived SW143 T seed별 native32 proof/reference를 검증하고 full320의 fresh QCC labels가 해당 T reference와 전부 일치해야 한다. 불일치는 technical stop이며 candidate를 조정하지 않는다.

Native source forward당 prepare_coupling에서 반환한 **동일 tuple의 첫 항**만 A로 캡처한다. Capture는 original tuple을 수정하지 않고 instance callable을 finally 복원한다. Per-image Q/A/gain/source state hashes를 candidate/control row ledger에 묶는다. Source gain을8이나 seed1 값으로 hardcode하지 않는다. 같은 forward를 두 readout이 공유한다.

## Fixed320/Slot과 실행 범위

Seed0와2 모두 IDs1320..1639의 동일 native128 RGB에서 직접32 labels를 만든다. 각 seed의 candidate/control/Slot archive와 prediction provenance를 고정·검증한 뒤 GT를 연다. SourceT own reference와 Slot own70k epoch10 seed가 섞이지 않아야 한다. Native128 Slot hard labels는 등록된4px modal converter만 사용하고 saved perimeter-BG policy를 유지한다. GT object count/allowed-label oracle/scene metadata를 inference에 쓰지 않는다. Reserve90640..90959는 읽지 않는다.

CPU source/reference admission 후 각 seed의 one eight-image TRAIN preflight,then one fixed320 B1 source pass가 범위다. Exclusive owner-aware leases/per-image transactional resume를 사용한다. Foreign GPU job은 공유·종료하지 않는다. 한 seed의 과학적 나쁜 결과가 다른 seed 보고를 취소하지 않는다. Technical failure는 해당 branch와 dependent scoring만 막고 partial artifacts를 보존한다. 완료 seed1은 검증된 결과를 재사용하며 수정/재학습하지 않는다.

## 보고와 판정

Seed0/2 결과는 좋고 나쁨과 관계없이 모두 보고한다. Discovery seed1과 prospective replication seed0/2를 구분한다. 각 seed 및 all3의 FG/foregroundIoU/objectIoU와 SourceT·동일 seed Slot 대비 차이를 기록한다. Shared320 sampled image indices를 모든 seeds/arms에 함께 적용한10000 paired bootstrap,RNG165를 사용한다. CI는 fixed three seeds에 조건부이며 seed-population generalization proof가 아니다. Seed0/2-only contrast도 별도로 보고한다.

고정 recipe의 replicated improvement는 all3 FG mean>SourceT,paired95%CI lower>0,적어도2/3 seed FG gains,objectIoU mean>=SourceT,foregroundIoU per-image exact SourceT로 판정한다. Ultimate board는 all3 means가 comparable Slot의 **세 지표 모두 strict superiority**인지와 FG mean>=.87을 별도로 판정한다. Discovery selection과 반복 사용된 validation의 한계를 밝힌다. Gate 미통과 시 threshold/weight/gain/min-size/seed selection sweep 없이 닫는다.

성공해도 source16-trained70k lineage의 native32 **inference/readout** 결과다. Fresh native32 70k training,positive unique-data scaling,all-gate causality,retrained layer necessity를 입증하거나 해당 jobs를 자동 승인하는 문서가 아니다.

````

</details>

## FIXED_SUPPORT_ARI_BOUND_20261014.md

<details>
<summary>FIXED_SUPPORT_ARI_BOUND_20261014.md</summary>

Source SHA256: `bfa0cc7ec2936bad3702596a38f2000cbe5db7740e1025a1b798c09522e4df0d`

````markdown
# Fixed-support FG-ARI의 인증 상한

이 문서는 SW0164/165의 예측 foreground membership을 고정했을 때 가능한 FG-ARI의 상한을 증명한다. 실제 solver를 변경하거나 GT를 prediction에 사용하는 방법이 아니라, 모든 prediction을 먼저 고정한 뒤 수행하는 GT-last 진단이다.

## 범위와 기호

Production FG-ARI와 같이 **GT foreground patch만** 모집단으로 삼는다. 그 수가 `n>=2`일 때 `T=choose(n,2)`, GT가 같은 pair 수를 `C`, prediction이 같은 pair 수를 `B`, GT와 prediction 모두 같은 pair 수를 `A`, `c=C/T`라 두면

`ARI = (A-cB) / (0.5*C + (0.5-c)*B)`.

SourceT가 background로 정한 patch는 prediction label 0으로 고정하고, foreground로 정한 patch는 0이 아닌 label을 가져야 한다. 상한 계산의 relaxation은 positive label 수를 제한하지 않고 disconnected group도 허용한다. SW0164 실제 solver는 각 positive group에 4-connectivity를 요구하므로, relaxation의 최대값은 실제 solver가 달성 가능한 값의 유효한 상한이다. 실제 solver가 disconnected group을 허용한다고 주장하지 않는다.

## 최적화 증명

임의의 admissible positive partition에서 여러 GT row를 섞은 positive cluster를 GT row별로 나눈다. 이때 `A`는 그대로이고 `B`만 감소한다. 분모가 양수인 비퇴화 경우

`dARI/dB | A = [A(c-0.5) - 0.5*c*C] / D^2`.

`c<=0.5`이면 음수이고, `c>0.5`이면 `A<=C`를 써서 최대 `0.5*C*(c-1)<0`이다. 따라서 mixed positive cluster를 나누면 ARI가 감소하지 않는다.

이제 같은 GT row에 속한 positive cluster들을 합친다. `A`와 `B`가 같은 양만큼 증가할 때 도함수의 부호를 정하는 항은

`0.5*C*(1-c) + (0.5-c)*(B-A)`.

`c<=0.5`이면 음수가 아니다. `c>0.5`이면 `B-A<=T-C=T*(1-c)`이므로 이 항은 최소 `0.5*T*(1-c)^2>0`이다. 따라서 같은 GT row의 positive cluster를 합쳐도 ARI가 감소하지 않는다.

두 변환을 반복하면 fixed label-0 patch는 그대로 두고, covered GT foreground에는 GT object별로 하나의 positive label을 준 partition에 도달한다. 그러므로 이 `fixed-support GT-induced FG-ARI`가 위 relaxation에서의 인증 최대값이며, 4-connected 실제 solver에 대한 인증 상한이다.

## 퇴화 경우

- `c=1`: GT foreground가 한 object뿐이다. missed와 covered patch가 모두 있으면 모든 admissible partition의 최댓값은 0이고 GT-consistent covered partition이 이를 달성한다. 전부 covered이거나 전부 missed이면 GT-consistent partition과 GT가 각각 단일 cluster이므로 분모가 0이며 production 구현 규칙대로 1이다.
- `c=0`: 모든 GT cluster가 singleton이다. `B=0`이면 분모가 0이고 production 규칙대로 1이다. 고정 background에 둘 이상의 patch가 있어 `B>0`이면 ARI는 0이며 GT-consistent covered partition도 최대값을 달성한다.
- GT foreground patch가 두 개 미만이면 production metric은 NaN을 반환한다. 상한 helper도 NaN을 반환하며 mean에서는 production `valid_count` 처리와 정확히 같은 방식으로 제외해야 한다.

`test_ari_bound.py`는 2~6 patch의 모든 GT partition, 모든 fixed-support mask, covered patch의 모든 positive partition을 열거해 후보 점수가 이 상한을 넘지 않음을 확인한다. 별도 소수 예제만 sklearn과 대조한다.

````

</details>

## INPUT_COMPATIBILITY_CHECKLIST.md

<details>
<summary>INPUT_COMPATIBILITY_CHECKLIST.md</summary>

Source SHA256: `1a23487d4165d3235b9bd2b21ee7d3704773db7506a4d1f3b706f3181ba36fc4`

````markdown
# SW165 seed0/2 input/API checklist

Design/admission checklist only. Do not modify sealed SW164 or another running experiment.

- [ ] SW164 seed1 score has independently validated complete receipt; save exact score/prediction/protocol and frozen-solver hashes. Preserve seed1 discovery provenance.
- [ ] Call `SW130.run.source_contract(seed)` for each seed,which delegates to `SW0110.run.source_paths/train_indices`; require matching source model seed,source manifest,4096 ordered IDs and117+seed order. Source core paths are `trained_models/SW0097_graph_adaptation/seed{seed}_positive_frozen/core.pt` via that API,not a hand-built substitute.
- [ ] Expected core SHA seed0=`36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf`; seed2=`798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661`. Verify the actual source audit/proof assets required by `source_paths`; no seed1 proof substitutions.
- [ ] Use generic `SW135.foundation.load_native32_foundation(seed,device)` strict mapping and `SW143.geodesic_operator.install_operator(graph,'T')`. Verify1024 units,adaptive_pool32,source encoder/statistics hashes,source geometry/non-feedback factorized backend,no pulse/no new emitter/threshold.
- [ ] Resolve own source rows/IDs for eight TRAIN positions0–3/2048–2051. All receipt arrays must be converted to JSON-native lists before dict equality or serialization. Do not read TRAIN masks.
- [ ] Own native32 source reference: `SW137.run.validate_prediction(SW137.run.OUTPUT_ROOT/f'seed{seed}',seed=seed)`.
- [ ] Own T reference: `SW143.run.validate_arm_prediction(SW143.run._arm_dir('T',seed),seed=seed)`. Require source/encoder/stats/operator/preflight/R-reference lineage and IDs1320..1639. Never reuse T seed1 labels as seed0/2 controls.
- [ ] Capture native `prepare_coupling(...)[0]` from the same unchanged rollout. Preserve original tuple,one static preparation per image,and restore instance callable finally. Same-seed gain is `exp(found.wrapped.core.graph_generator.log_coupling_gain)`.
- [ ] Production Q uses actual four component spikes and settle512 exactly once. Run untouched/captured six-trace/Q/QCC parity at B1/T1024/live0; full320 fresh QCC must exactly replay archived own-seed T.
- [ ] Call sealed `SW164.foreground_signed.foreground_induced_merge(Q,A,gain,source_labels,grid_shape=(32,32))`; no seed argument or new parameter belongs inside this scientific kernel.
- [ ] Empty FG→all0;isolated FG→singleton;source-FG disconnected components cannot merge. All source FG is retained and source BG remains0. No second min8/largestBG pass.
- [ ] Slot paths: `trained_models/SW0092_slot_our70000_eval/seed0_epoch10/{predictions.npz,protocol.json}` and corresponding `seed2_epoch10`. Call `SW135.evaluation_contract32.validate_slot_protocol(...,seed=seed,prediction_path=...)` and `slot_native_pixels_to32`.
- [ ] Slot checkpoint SHA seed0=`0301a456a586da5e1132c703d095f4736c4351cb86388eabe2aff7f1c96ae091`; seed2=`c665b2b52cb59ee5044b91cebff039a4b5199fa3edc2b5fa769094fcadadb200`. Model/preprocessing/BG/IDs and prediction/protocol bytes are also bound by the actual validator; checkpoint SHA alone is insufficient.
- [ ] New runner paths/protocols/manifests/row ledgers include explicit seed and new experiment identity. Do not call seed1-only SW164 `_references/_out/_work` or SW162 `prepare_preflight_cpu` under a global seed monkeypatch.
- [ ] Source/candidate/Slot predictions and hashes are frozen before GT;standard modal4px/native32 scorer only. Require candidate-vs-source binary foreground equality and per-image foregroundIoU exact equality at scoring.
- [ ] Full all3 summary reuses validated seed1 plus both0/2 regardless outcomes;shared-image bootstrap and discovery/replication distinction are explicit. No new model training or reserve access.

````

</details>

## NEXT_DIRECTION_ANALYSIS_20261014.md

<details>
<summary>NEXT_DIRECTION_ANALYSIS_20261014.md</summary>

Source SHA256: `2d4413023b64a1873d8d5fae5f31843d2dad0b941cf04ab536c691c1079a47e7`

````markdown
# SW0164 이후의 제한된 판별 진단

2026-10-14 prospective analysis. 이 문서는 SW0165 재현 결과를 선택하거나 변경하지 않으며, SW0164/165의 저장된 예측을 모두 고정한 뒤 한 번 수행할 GT-last 진단만 등록한다.

## 현재 증거

SW0164 seed1 foreground-induced 결과는 `.770932/.614993/.630523`, SourceT는 `.755273/.614993/.579514`였다. SourceT 대비 FG-ARI 차이는 `+.015659`, paired 95% CI `[.009116,.022883]`; object IoU 차이는 `+.051010`, CI `[.041976,.060664]`이다. Foreground IoU와 foreground membership은 이미지마다 SourceT와 정확히 같다. 따라서 SW0162의 foreground 손실 없이도 signed relation이 instance partition을 개선할 수 있다는 점은 입증됐다.

그러나 Slot seed1 FG-ARI보다 `-.070887`, 95% CI `[-.094692,-.046903]`이고 `.87`보다 `-.099068` 낮다. 320장 중 200장이 FG-ARI `.87` 미만이다. SW0164와 SW0162의 이미지별 SourceT 대비 FG-ARI 변화는 상관 `r=.7994`이므로 두 readout은 대체로 같은 쉬운 장면에서 개선된다. 고정 support가 SW0162보다 FG IoU를 `+.016608` 회복했지만 FG-ARI는 오히려 `-.004453`이므로, support 보존만으로 남은 grouping 격차가 해결되지는 않는다.

다만 현재 score만으로 남은 격차를 foreground 누락 탓 또는 covered-foreground 내부 merge/split 탓이라고 단정할 수 없다. production FG-ARI는 GT foreground만 평가하지만 그 안의 predicted-background patch를 하나의 cluster로 남긴다. SourceT의 false-negative foreground는 여러 GT object를 background cluster로 합쳐 FG-ARI 상한을 낮출 수 있다.

## SW0165 뒤 한 번만 수행할 GT-last 진단

SW0164와, 재현에 성공하면 SW0165 seed0/2의 **이미 저장되고 hash가 고정된 prediction**에 대해서만 canonical modal32 GT를 연다. 학습, 재예측, coefficient/threshold 선택은 하지 않는다. 각 이미지와 seed에 다음을 기록한다.

1. `fixed-support GT-induced FG-ARI upper bound`: SourceT foreground 안의 patch에는 GT object ID를 주고, SourceT background 안의 GT-foreground patch에는 모두 label 0을 준다. [`FIXED_SUPPORT_ARI_BOUND_20261014.md`](FIXED_SUPPORT_ARI_BOUND_20261014.md)의 증명처럼, unlimited/disconnected positive group을 허용한 relaxation에서의 인증 최대값이다. SW0164 실제 solver는 4-connectivity를 요구하므로 이 값은 실제 solver의 인증 상한이기도 하다. 실제 solver가 disconnected group을 허용한다는 뜻은 아니다.
2. `support-idealized attachment FG-ARI`: candidate의 covered-patch group은 그대로 두고, missed GT-foreground patch를 같은 GT object와 overlap이 가장 큰 기존 group에 붙인다. overlap이 없으면 그 GT object용 새 label을 준다. 이것도 GT-assisted reference이며 실제 predictor 성능이나 support만의 인과 효과로 해석하지 않는다.
3. GT-foreground patch를 `covered`와 `missed`로 나누고, missed patch 수와 missed GT-object 수를 기록한다.
4. covered patch 쌍에서 (a) 같은 GT object인데 서로 다른 candidate group인 split pair와 (b) 다른 GT object인데 같은 candidate group인 merge pair를 각각 전체 가능한 same/different-object pair로 정규화한다. predicted-background 안의 GT-foreground 쌍은 별도 `background-collapse merge`로 기록한다.
5. seed별 평균과 320-image paired bootstrap CI를 보고한다. per-image 상관은 설명용이며 원인 추론이나 선택에 쓰지 않는다.

판정은 하나다. 세 seed에서 GT-induced reference가 각 seed의 Slot FG-ARI와 `.87` 중 더 높은 기준에 못 미치고 forced background-collapse/cross-support pair 오류가 남은 deficit의 대부분이면, 다음 모델 시험은 foreground estimator를 함께 바꾸되 SW0164 partition을 고정한 단일 실험이어야 한다. Reference가 기준을 넘고 covered-patch split/merge가 deficit의 대부분이면 foreground 변경은 보류하고, 둘 중 더 큰 정규화 오류 하나를 줄이는 learned assignment 설계만 주요 검토 대상으로 올린다. 어느 쪽도 지배적이지 않으면 새 모델 시험을 열지 않고 이 이분법이 불충분하다고 보고한다. 어느 경우에도 이 진단값으로 threshold, group count, merge weight를 고르지 않는다.

SW0165가 FG-ARI 개선을 세 seed에서 재현하지 못하면 이 readout의 새 학습 실험은 열지 않고 종료한다. 이 진단은 seed1의 성공 이유를 설명하는 사후 분석으로만 남긴다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `902eb1aec17dd331752e38d902b2b00ff3625cb5f1b80e2f3374404a06150d18`

````markdown
# SW0165 ? seed ?? ??

?? native32/320?, seed0?1?2 ?? ?? ????. Seed1? SW0164 discovery ??? ?? ?????, seed0?2? ?? recipe? ?? ???? ?? ????. 16?16?? ??? 70k lineage source? native32? ??? frozen readout?? ?? ??? optimizer update? 0??.

| ?? | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| SW0165 foreground partition | 0.804299 | 0.609975 | 0.651149 |
| ?? SourceT | 0.783044 | 0.609975 | 0.596197 |
| ?? seed? Slot70k epoch10 | 0.781652 | 0.204360 | 0.203165 |

| Seed | ?? FG-ARI | Slot FG-ARI | ?? Foreground IoU | ?? Object IoU |
|---|---:|---:|---:|---:|
| 0 | 0.846399 | 0.626661 | 0.765802 | 0.693917 |
| 1 | 0.770932 | 0.841819 | 0.614993 | 0.630523 |
| 2 | 0.795566 | 0.876476 | 0.449131 | 0.629006 |

FG-ARI? Slot ?? ? seed ?? ??? +0.022647, ??320-image paired95%CI [+0.003997, +0.041634]?. SourceT ?? ??? +0.021255, CI [+0.015861, +0.027028]?. Foreground membership? foreground IoU? ?? seed/image?? SourceT? ??? ??. ??? replication gate? ????.

? ?? ??? Slot? ?? ???, seed1?2? FG-ARI? ?? ?? seed? Slot?? ??. ? CI? ??? ? seed? ?? ??? validation images? ????? seed-population ??? ???? ???. FG-ARI .87, fresh native32 70k scaling, ?? gating? ??? ??? ?? ?? ??? ????.

??? public score validators? `validate_and_summarize`? ?? ?? ????? ?? ??? ??? ???? ????. ?? ???paired CI??? ?????? `results_archive/completed_all_seeds_20261015/server_records.json`? ??. Summary SHA256: `72b17ca1a07d312400ab74f6ee7533c89f0e9d9a6c304378dd8e61c16ba379c5`.

````

</details>
