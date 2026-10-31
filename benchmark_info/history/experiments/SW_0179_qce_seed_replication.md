# SW_0179_qce_seed_replication

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `bcb1a6317abd0b92a3d17e2e1f99f86c0c7ee5d864ec6dc02639a581f282d255`

````markdown
# SW0179 — supervised QCE seed replication

## Status and purpose

The audited SW0178 seed-1 product-QCE result is supervised exploratory evidence. Its preregistered task gate failed, so it does not replace the prior incumbent or support an unsupervised claim. SW0179 repeats the same fixed product-QCE procedure from fresh SW0097 source seeds 0 and 2. It does not reuse learned weights or optimizer state from SW0178, SW0177, or earlier candidate experiments. The existing SW0178 seed-1 checkpoint, prediction, and score are the candidate's seed-1 result only.

## Fixed training protocol

- For each new seed, load the matching source with `foundation.load_native32_foundation(seed, device, verify_rgb_assets=False)` and apply `install_operator(graph, "T")`. Do not use the seed-1-only SW0169 source loader.
- All three seeds use the validated shared SW0171 TRAIN512 RGB and full modal instance labels. SW0171 producer lineage refers to source seed 1 and proves only common ordered data/assets; it is recorded as dataset provenance. Its seed-1 feasible targets, support masks, and spike cache are not model inputs for seeds 0 or 2.
- Construct a fresh zero-context wrapper with context seed `171000 + source_seed`. Draw the two consecutive 512-image permutations from `RandomState(171001)`. Use 2 passes, 64 logical updates, B16/micro4, T1024, settle512, live tail64, and per-update seed `171000 + source_seed + zero_based_update`.
- Keep the frozen SW0178 product-Q leave-one-out class competition loss and temperature 0.1, including full TRAIN modal labels and background class 0. Do not change architecture, optimizer, learning rates, clipping, integration projection, or loss. Each source seed starts a fresh Adam. The first two committed updates must show finite positive emitted-S gradient and finite positive gradient and parameter change for each of the four registered optimizer families.
- This is supervised training: `mask_supervised=true`, `unsupervised_leaderboard_eligible=false`. Do not read validation or holdout labels during training or selection.

## Evaluation and promotion gates

Freeze all new-seed predictions, protocols, IDs, and hashes for the fixed validation IDs 1320–1639 before opening validation ground truth. Use the existing SW0178 product-QCE prediction as candidate seed 1. The primary endpoint is standalone prediction from actual S → native component Q → native QCC → SW0165 solver. Report the conditional SourceT endpoint separately.

For each image, average metrics across the three candidate seeds first; bootstrap the 320 paired image rows, not 960 seed-image observations. Compare the three-seed standalone result to the three-seed SW0165 incumbent (FG-ARI 0.8042989149474198, foreground IoU 0.6099754484530633, matched object IoU 0.6511489750258587) and to the matched three-seed Slot Attention baseline (FG-ARI 0.7816519719609135, foreground IoU 0.20436012776655546, matched object IoU 0.2031649861884501). FG-ARI must strictly exceed both FG baselines, with paired 10,000-resample 95% lower bounds above zero for both comparisons (bootstrap seed 178). Foreground IoU and matched object IoU must strictly exceed the Slot Attention means. Record the separate 0.87 FG-ARI goal; it does not replace these gates. No unsupervised leaderboard promotion claim is allowed.

## Provenance

Bind each seed's source checkpoint and manifest, shared SW0171 dataset manifest/arrays and source-seed-1 producer provenance, per-seed source state, initial zero-context state, encoder/preprocessing asset hashes, optimizer groups, exposure order, runtime/device, durable boundaries, final model, and prediction hashes. Do not claim SW0161 seed-1 cached spike evidence for seeds 0 or 2. Do not read validation ground truth or reserve data before every candidate prediction is immutable.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `f1579d11e672a39223daa8d895f10db658a9f3986385bfdecc021bfc13926292`

````markdown
# SW0179: QCE의 seed 0·2 재현성 검증

완료·독립 검산된 standalone 세 seed 평균은 **FG-ARI 0.761620 /
Foreground IoU 0.469308 / Object IoU 0.584656**입니다. seed 0·1의 FG는
높았지만 seed 2가 0.616842로 낮아져 등록한 primary gate에 실패했습니다.
기존 최고 unsupervised SW0165를 유지합니다.

전체 비교·신뢰구간·한계는 [RESULT_KO.md](RESULT_KO.md)에 있습니다.
이하 학습·큐 등록 근거에 기록된 실행 단계는 모두 완료됐고 재시작하지
않았습니다. 최종 점수·독립 검산·예측 protocol은 `results_archive/score_v1/`입니다.

SW0178 seed 1에서 standalone FG-ARI 0.8180 / Foreground IoU 0.4375 /
Object IoU 0.6864였던 product-QCE 설정을 변경 없이 반복합니다. 각 seed의
원래 SW0097 checkpoint에서 시작하고, TRAIN512 이미지와 정답 mask로
2회 추가 학습합니다. supervised 실험이며, 70,000장 신규 학습 결과나
unsupervised 성능으로 해석하지 않습니다. seed 1 결과는 기존의 검증된
SW0178 결과를 사용합니다.

실제 서버 CPU 검증에서 세 seed의 checkpoint가 서로 다름을 확인했고,
seed 1 초기 모델 상태는 SW0178과 정확히 일치했습니다. 각 seed의
zero-context gamma, 빈 optimizer, checkpoint 저장·복원도 확인했습니다.
학습·예측·점수 계산과 각 큐의 테스트 23개가 서버에서 통과했습니다.

seed 0·2의 예측 큐도 서버에 등록했고 실제 부모 프로세스가 학습 완료를
기다리는 것을 확인한 뒤, 학습 완료 후 빈 GPU에서 자동 예측을 마쳤습니다.
다른 사용자의 GPU 작업과 겹치면 양보하도록 구성했습니다. 예측 코드와
동일 seed의 실제 SourceT 기준 파일도 CPU에서 검증했습니다.

CPU 점수 계산 큐도 실제 프로세스로 등록했습니다. 새 seed 0·2와 기존
SW0178 seed 1의 예측이 모두 확정된 후에만 평가 정답을 읽습니다.
비교할 SW0165·Slot Attention의 세 seed 기준 파일과 평균도 실제 서버에서
검증했습니다. 최종 성능 점수와 별도 검산까지 완료됐습니다.

고정 32×32 patch·320장 평가에서 세 seed 평균을 SW0165와 같은 조건의
Slot Attention에 비교합니다. 예측을 모두 확정한 뒤 평가 정답을 읽습니다.
FG-ARI 향상이 특정 seed에만 나타나는지와 두 IoU의 tradeoff를 함께 보고합니다.
사전 기준과 한계는 [DESIGN_KO.md](DESIGN_KO.md), 실제 시작 근거는
`results_archive/start_v1/`, 평가 준비와 큐 등록 근거는
`results_archive/eval_v1/`에 기록했습니다.
학습 완료 검증과 각 최종 manifest는 `results_archive/train_v1/`에 있습니다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `50b2c79fa341df5697dfdd230c90dbc99d86d8d101be6b9a1ce1fbb56bf34216`

````markdown
# SW0179 — QCE seed 재현성 검증 결과

완료·독립 검산된 고정 32×32 patch / 동일 320장 결과입니다.
이 supervised QCE 추가 학습은 세 seed 재현에 실패했습니다. 기존 최고
unsupervised SW0165를 유지하며, 이 설정 그대로 70,000장으로 확대하지 않습니다.

| Standalone 결과 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| QCE seed 0 | 0.849988 | 0.642330 | 0.692904 |
| QCE seed 1 — 기존 SW0178 | 0.818029 | 0.437500 | 0.686364 |
| QCE seed 2 | 0.616842 | 0.328093 | 0.374701 |
| **QCE 세 seed 평균** | **0.761620** | **0.469308** | **0.584656** |
| 기존 SW0165 세 seed 평균 | 0.804299 | 0.609975 | 0.651149 |
| 비교 가능한 Slot Attention 세 seed 평균 | 0.781652 | 0.204360 | 0.203165 |

QCE 평균 FG-ARI는 SW0165 대비 -0.042679이며, paired 95% CI는
[-0.052731, -0.033229]입니다. Slot 대비 -0.020032,
95% CI [-0.040078, -0.000144]입니다. 같은 이미지의 세 seed 점수를 먼저
평균하고, 공유 320개 image index를 10,000회 재표집했습니다(RandomState 178).
두 IoU는 Slot보다 높지만, 등록한 primary gate와 FG-ARI 0.87 목표는 실패했습니다.

Conditional SourceT foreground를 그대로 유지해도 평균은
0.767848 / 0.609975 / 0.598112입니다. Conditional FG-ARI는
seed 0/1/2가 각각 0.845145 / 0.777384 / 0.681013입니다.
따라서 seed 2의 문제를 standalone 배경 판정 하나로 설명할 수는 없습니다.
학습 불안정성, loss와 최종 mask 분류의 불일치, 일반화 중 어느 원인인지
이 평가만으로 확정하지 않습니다.

각 seed는 자기 SW0097 source checkpoint에서 시작합니다. seed 0·2는
새 optimizer로 공통 TRAIN512장을 두 pass(1,024 exposures, 64 updates)
추가 학습했고, seed 1은 변경하지 않은 SW0178 product-QCE 결과입니다.
TRAIN 정답 modal mask를 사용한 supervised 실험이며 native32/70,000장
신규 학습이나 unsupervised 개선으로 해석하지 않습니다. seed 0·1의 유용한
checkpoint는 탐색 결과로 보존하며 실패한 세-seed 후보를 승격하지 않습니다.

실제 학습·예측·점수 계산은 모두 완료됐습니다. 최종 모델과 boundary tensor,
Adam step, 두 pass의 데이터 노출 및 첫 두 update의 spike gradient와 네
optimizer 그룹 변경을 검증했습니다. 모든 후보 예측을 확정한 뒤 fixed320
정답만 읽었고, reserve data는 읽지 않았습니다. 별도 검산은 각 seed·endpoint
점수, 기준 파일, bootstrap, gate와 전후 artifact hash를 재확인했습니다.

최종 점수 SHA256: `758e48f83b44dc31efeae73154c13dbcd8ffa6132a85c7055279a0f4b2649c3d`.
원본 점수와 독립 검산, 예측 protocol·모델 불변 기록은
`results_archive/score_v1/`에 있습니다.

````

</details>
