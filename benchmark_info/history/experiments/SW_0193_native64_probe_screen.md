# SW_0193_native64_probe_screen

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `4ca064dce7f51cef31ffed1d6851437015dca7e45e71f7dc81d185f2f7ceb1d1`

````markdown
# SW0193: small native64 before/after generalization screen

The primary target remains native64 three-seed fixed320 FG-ARI>=.90, retaining
meaningful gates and reporting both IoUs. Routine development uses mini tests;
70k training requires a large replicated improvement first.

Before spending more native64 optimizer time, measure the actual transferred
incumbent and the already-trained SW0192 probe on predetermined32 development
images, canonical IDs1320..1351. This is an exploratory seed0 screen, not a
three-seed benchmark, fresh70k training, or proof of FG.90. The eight TRAIN
images and two B4 proposals are inherited from SW0192 and disclosed as such.
The larger TRAIN64/16-proposal mini stage remains a subsequent decision after
this screen and measured runtime; do not claim it already implemented.

Baseline is the exact unadapted SW0186 seed0 -> native64 mapping admitted in
SW0192. Candidate is SW0192's final, successfully admitted native64 model.
Require the SW0192 report, boundary hash, learned model hash, accepted steps,
first-two gradient credit, and current implementation fingerprint. No updates
in this experiment. Same RGB, same normalization, graph/T1024/settle512 and
production QCC plus foreground-induced merge at64x64; min_group32 preserves
the native32 min_group8 spatial area. No postprocessing or loss sweep.

Both endpoints use the SAME per-image inference seed193000+image_offset. Save
each direct native64 prediction atomically and bind model/RGB/source hashes.
Re-use only fully validated matching rows after yielding a foreign GPU owner.
Freeze both complete prediction archives and validate all three Slot64 source
archives before opening development GT. Pool canonical128 GT and Slot labels
directly by production2x2 modal. Never interpolate native32 masks or targets.

Report all three scores for baseline/candidate/Slot seed0 and Slot three-seed
mean on the same32 images. The latter is context, not a matched-seed superiority
claim. Primary screen is candidate-minus-baseline FG mean and paired-image95%CI;
all3 IoU changes are disclosed. Verify every metric independently from stored
masks using the SW0185 arithmetic, including independent modal pooling. Keep
imagewise GT-derived values private; publish aggregates and hashes only.

Positive screen: FG gain>=.01 and paired-image95%CI lower>0. This admits more
mini training or full320/three-seed confirmation; it is not final success. A
negative result closes the two-proposal adaptation as a useful performance
change; it does not prove that further native64 training cannot help. Diagnose
the initial resolution transfer and measured resource cost before more training.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `4cb825f0aea4e0bba5e7720c1b948bd2ae5f38b3f30d49a21cca017bd62799ee`

````markdown
# SW0193 native64 before/after mini screen

Compare the exact mapped SW0186 native64 model with the successfully admitted
SW0192 model (eight TRAIN images/two proposals) on predetermined32 development
images. Same inference RNG and production spike-derived readout. Score only
after both masks are frozen; independently verify actual metrics and publish
aggregates only. This is a seed0 mini screen, not a three-seed FG.90 result.

Completed and independently audited: baseline FG .909537 -> candidate .918586
on32 development images/seed0. The registered +.01 minimum gain was not met;
screen_gate_passed=false. This is not final full320/three-seed FG.90 achievement.
See RESULT_KO.md and results_archive/score_v1/summary.json.

The screen ran on GPU2 after SW0192 successfully completed. Local/server
seven tests include real native64 mask conversion and foreground merge, paired
prediction resume/hash checks, actual production/independent metric parity and
GT-access ordering. Exact parent checkpoint/model/Adam and baseline CPU admission
passed before GPU leasing. Completed parent PID1836851, worker PID1837070.

The original inference attempt failed because the mask conversion function was
imported from spike_classifier instead of evaluation. Original state/log/source
and CPU admission were preserved under failure_v1 on the server; the failed queue
state/log are also archived locally. The import and a real conversion regression
test were corrected before seven server tests passed and inference resumed.
No training was repeated and no prediction rows or validation GT were consumed
by that failed attempt. The initial recovery launcher observed the same parent
before its CPU admission wrote state; the same PID was subsequently verified,
with no duplicate relaunch. See start_v2 and cpu_v2 for evidence. The canonical
cpu_v1/admission.json contains the current exact model binding; its fingerprint
identifies the fixed source version. The earlier six-test receipt is retained
as the pre-fix record, not evidence for the corrected readout path.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `48bedb0a0e56840019c64b61e665539d0d9742b7ceb48a5622a258d18a92f045`

````markdown
# SW0193 완료: seed0 native64 미니 평가

고정 development32장(IDs1320..1351), native64 직접 spike-derived mask 기준이다.
이는 전체320장·세 seed 평균 결과가 아니며, 최종 FG-ARI .90 목표 달성이 아니다.

| 모델/같은32장 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0186 learned model의 native64 전환, 추가 학습 전 | .909537 | .827147 | .734442 |
| SW0192 native64 TRAIN8/2-proposal 추가 학습 후 | .918586 | .832867 | .733686 |
| Slot seed0 | .689062 | .180979 | .209370 |
| Slot 세 seed 평균 (참고) | .821571 | .173315 | .203988 |

우리 모델은 seed0 하나다. Slot 세 seed 평균과의 비교는 참고이며, matched
three-seed superiority claim으로 사용하지 않는다. 미니 subset의 Slot 평균도
기존 전체320장 Slot64 평균 .786902보다 높으므로, 이 subset 점수를 전체 평가
점수로 외삽하지 않는다.

추가 학습 후-minus-before paired-image 차이와95%CI:

- FG-ARI +.009049, CI[+.001647,+.019472]
- Foreground IoU +.005721, CI[-.004043,+.018492]
- Object IoU -.000756, CI[-.018451,+.013969]

FG 변화의 paired-image CI는 양수지만, 등록한 minimum gain .01 기준에 미달해
screen_gate_passed=false다. 기준을 사후 완화하거나 이 결과를 새 incumbent로
승격하지 않는다. 데이터양, 업데이트 수, gating 인과성, seed 일반화는 입증하지
않는다. 양의 미니 신호를 기록하되, 더 큰 TRAIN64 미니 continuation은 원래
SW0192에 등록한 별도 학습 예산 검증으로 다뤄야 한다. 70k 학습은 예약하지 않았다.

실제 원본 CNN/graph/core/integration이 학습된 것은 SW0192 checkpoint와 Adam
step2/gradient 증빙으로 확인했다. 본 실험의 optimizer update는0이다. 두 모델
모두 같은 per-image inference RNG, T1024/settle512, SourceT graph, production QCC
및 foreground-induced merge를 사용했다. 두 모델의32개 mask를 먼저 동결한 후
GT를 읽었고, Slot native128 및 GT는 직접2x2 modal로 변환했다.

분리된 read-only audit에서 실제 후보64 image rows와 Slot96 rows, 독립 modal,
ARI/IoU/assignment/mean/CI/gate를 다시 계산했다. 최대 metric 오차는
2.220446049250313e-16이다. 원본 이미지별 점수는 비공개로 보관한다.

유효한 추론 시간은 baseline32장1138.79초, candidate32장1152.98초로, 각각
이미지당35.59초/36.03초였다. 초기 import 오류의 첫 이미지 forward는 이 합계에
포함되지 않는다. 오류·기존 source·admission·로그를 보존하고 실제64 mask 변환
회귀검사를 추가해 일곱 server tests를 통과한 후 재개했다. 학습은 반복하지 않았다.

공개 집계 summary SHA256:
`6e8e036718c520eec04226fc75e6356644de73bcea74e338bb7c174be9cbd4eb`

증빙: results_archive/score_v1/summary.json, completion_audit.json,
frozen_v1 및 completed_v1.

````

</details>
