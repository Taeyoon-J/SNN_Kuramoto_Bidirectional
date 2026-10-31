# SW_0236_native64_TRAIN512_exact_resume

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `0e3b039d29f8b79f1ef771cd6ab841be9ba297e670eb5406e1526ef902606868`

````markdown
# SW0236: 같은512회 학습에서 unique TRAIN64와 TRAIN512 비교

SW0226에서는128회 업데이트로 TRAIN64와512를 비교했으나 FG 차이 CI가0을
포함했다. SW0232에서 TRAIN64의 누적512회 평균이128회보다 높아졌다.
따라서 학습량 부족이 데이터 증가 효과를 가렸는지 한 번 비교한다.

실제 SW0226 TRAIN512의128회 encoder·projection·Adam·history·Torch RNG를
직접 복원하고 bitwise 검증한 뒤384회만 추가한다. 누적512회/B16/8192
exposures로 SW0232 TRAIN64와 동일하다. 512개의 고유 이미지는 기존
TRAIN pool에서 고정됐고 앞64장은 SW0232와 같다. 원본 teacher·loss·LR·
clip·나머지 core·graph·gates·mask readout은 유지한다. 반복 노출은
64장128회씩 대512장16회씩이며 신규70,000장 학습이 아니다.

actual accepted128 checkpoint의 CPU 복원/gradient 검사를 통과한 뒤 큐에
등록한다. 235번 세 seed가 먼저 GPU 실행 기회를 받은 후에만236번 GPU
작업을 시작한다. 외부 작업이 있으면 우리 worker만 양보하고 경계에서 재개한다.

고정16장·native64·세 seed를 원래 기준 및 SW0232 TRAIN64/512회와 비교한다.
데이터 증가 효과 확정은 해당 FG 차이의 paired image CI 하한>0일 때만
주장한다. FG 후보 gate는 원래 기준 대비+.01 및 CI 하한>0, TRAIN64보다
FG 평균이 높아야 한다. balanced gate는 두 비교군 모두에 대해 두 IoU
평균도 낮아지지 않아야 한다. 기존16장 GT barrier와 RNG/1024-step spike
mask 계약을 유지한다. 전체320장 기준점 SW0227은 이 미니로 바꾸지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `7727bc6dd34567b378849ac8c15a07a088082b3476d96222d571bf135c5b1f60`

````markdown
# SW0236: TRAIN512 exact continuation at matched512-update budget

Restore actual SW0226 TRAIN512/128 weights, Adam, history and Torch RNG exactly;
add384 updates to cumulative512. Compare with SW0232 TRAIN64/512 on fixed16,
three seeds. Same teacher objective and encoder+actual gamma projection; other
core states and spike/membrane readout frozen. CPU restore/gradient admissions
precede GPU queues. Wait until every SW0235 seed has had its first GPU attempt.
Completed and independently audited: [results](RESULT_KO.md).
Fixed16 three-seed mean: .885655/.657488/.724440. FG data-scaling CI includes0;
balanced gate failed. No fresh70000 training, full320 promotion or gating causal claim.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `0efd66f309bdef1f13f39e75284ec5553f0a7ce1fbfb9abf56c239efe64f4c57`

````markdown
# SW0236: 같은512회 학습에서 데이터 증가의 FG 효과는 아직 불확실

고정16장·native64·seed0/1/2 평균이다. 기존 pretrained 모델에서 추가 학습한
고유 TRAIN 이미지 수와 누적 업데이트 횟수를 구분한다. 처음부터64장/512장만으로
학습한 모델도, 신규70,000장 학습 결과도 아니다.

| 같은512회·batch16·8,192 exposures | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 기준 | .842891 | .600692 | .702161 |
| TRAIN64 / SW0232 | .884622 | **.677086** | .712312 |
| TRAIN512 / SW0236 | **.885655** | .657488 | **.724440** |

TRAIN512−TRAIN64의 FG 차이는 +.001034, paired image 95% CI
[-.017698,.021437]이다. 데이터 증가로 FG가 좋아졌다고 확정할 수 없다.
Foreground IoU 차이는 −.019598, CI[-.048114,.003658]로 평균은 낮지만
감소 자체도 확정하지 않는다. Object IoU는 +.012128,
CI[.000026,.025617]로 이 고정 cohort에서는 개선을 지지한다. 이는 고정된 세
모델에 조건부인 이미지 bootstrap이며 seed 모집단 CI가 아니다.

원래 기준 대비 FG는 +.042765, CI[.009233,.089741]로 등록된 FG 후보 gate는
통과했다. 그러나 TRAIN64 대비 Foreground IoU 평균이 낮아 balanced gate는
실패했다. 데이터 증가 효과 flag도 false다. 이를 세 지표 모두 개선된 새 최고
모델이나 전체320장 FG≥.90 달성으로 승격하지 않는다. 바로70,000장 학습이나
같은 방향의 계수·epoch sweep을 추가할 근거로 사용하지 않는다.

실제 SW0226 accepted128 checkpoint의 encoder·실제 gamma projection·Adam·history·
Torch RNG를 bitwise 복원하고 새384회를 더해 누적512회로 학습했다. 세 seed의
실제 Adam counter는512이며 나머지 core/graph/gates/readout은 유지했다.
모든 prediction protocol을 확인한 뒤 DEV GT를 읽고 실제144개 mask를 독립
재채점했다. Teacher와 TRAIN 순서, checkpoint/source SHA, 평균과 CI도 검증했다.
로컬 통계 재계산 오차는0이다. 원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.

세 seed 모두 TRAIN teacher loss와 DEV 물체 내부 frequency variance는 원래
기준보다 감소했다. 그러나 최종 within-object variance는 TRAIN64보다 조금씩
높았다(.002838/.008891/.003428 대 .002771/.008172/.003398). Teacher loss 감소가
데이터 증가에 따른 FG 향상을 보장하지 않았으며, 그 원인이나 모든 gating의
기여를 이 통계만으로 단정하지 않는다.

전체320장 최고는 SW0227 .858916/.605286/.674313으로 유지한다. SW0235의
고정 TRAIN64/512회 모델 확장64장 확인은 별개로 계속 진행된다.

근거: `results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}`.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `2910a925b64d5b10a7354d9912b38dc6f461ba1b4b3d62953a9367316fa5c1a9`

````markdown
# SW0236 후속 비교 준비 및 실제 CPU 검증 시작

같은512회/B16 업데이트에서 고유 TRAIN64와512의 차이를 확인한다. TRAIN512의 실제 SW0226 128회 checkpoint를 복원하고384회만 추가한다. 원본 teacher·loss·나머지 core/gates/readout은 유지하며, 기존512장 pool을 사용한다. 데이터셋을 교체하거나 새70,000장 학습을 시작하는 것이 아니다.

로컬18개·Linux18개 검사와 CLI3개 검사를 통과했다. 세 seed의 실제 accepted128 checkpoint CPU 복원·gradient 검사를 시작했으며, 통과 후에만 후속 큐가 등록된다. 235번 모든 seed의 첫 GPU 실행 기회가 확인되기 전에는 이 새 학습을 시작하지 않는다. GPU0–3 독점 idle lease와 외부 작업 양보 정책은 유지한다.

목표는 고정16장·세 seed의 실제 spike/membrane mask 비교다. 데이터 증가 효과는 TRAIN64/512회 대비 FG 차이 CI 하한이0보다 클 때만 확정한다. 미니 결과로 전체320장 최고나 .90 목표를 갱신하지 않는다. 실제 시작 단계와 소스 SHA는 START_EVIDENCE.json에 기록한다.

````

</details>
