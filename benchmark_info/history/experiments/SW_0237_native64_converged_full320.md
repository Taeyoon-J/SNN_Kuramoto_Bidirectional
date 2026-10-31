# SW_0237_native64_converged_full320

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `65df04ef27f170985e54f53683d7f3ed0dadc71955af8af300937912e37870e4`

````markdown
# SW0237: fixed converged model, full320 confirmation

SW0235 independently accepted the actual SW0232 cumulative512 checkpoints on
first64 reused validation images, after the original fixed16 FG gate failed.
Preserve that failure. Same seed0/1/2 checkpoint, no new training or selection.
Reuse all64 validated masks/frequencies, require offset0 exact replay, then infer
only256 additional images per seed. Canonical IDs1320..1639, native64, T1024,
settle512, original spike/membrane-derived readout and exclusive GPU queues.
All prediction protocols and comparable Slot predictions precede DEV GT.
Report baseline210, candidate232, comparable Slot means and current227 reference;
FG >= .90 plus strict superiority over Slot on all three metrics is the target.
No fresh70000 training, untouched-test or all-gating-causality claim.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b7446ca9c92f268ebc72eb545af2874bd5b13c0c574d4e2007377f08471ae6c5`

````markdown
# SW0237: converged native64 full320

Prepared from the existing full320 path; actual SW0235 accepted checkpoint gate
is mandatory. Reuse64, infer256 plus one exact replay, no optimizer updates.
No run or result is claimed until actual server admissions and queues exist.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `205933c1d1e28135abda72370367632b51b85df8e8bd5598ff339e4bb8fc0e8d`

````markdown
# SW0237: new audited native64 full320 reference

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| SW0237 three-seed mean | 0.872680 | 0.626110 | 0.676331 |
| Previous SW0227 mean | 0.858916 | 0.605286 | 0.674313 |
| Comparable Slot Attention mean | 0.786902 | 0.198201 | 0.195203 |

The new checkpoint improves the previous full320 reference means by0.013764 /0.020824 /0.002017. It strictly exceeds the comparable Slot Attention means on all three metrics. The native64 FG-ARI0.90 target remains unmet. This is a comparison of three-seed means; individual seeds do not all exceed Slot Attention on every metric.

The best full320 FG-ARI seed is seed0:0.885240 /0.798583 /0.697130. Seed1 is0.869394 /0.592512 /0.675732 and seed2 is0.863406 /0.487234 /0.656130.

The evaluated models are the exact accepted SW0232 checkpoints:64 unique TRAIN images per seed,512 cumulative Adam updates, whole encoder plus the actual8-to4 sensory projection trained. The remaining core and production spike readout are fixed. SW0237 adds no training; it confirms these checkpoints on the fixed320 native64 validation images. Longer optimization improves the mean relative to the128-update reference. This does not demonstrate a benefit from more unique training images.

All three prediction protocols and comparable Slot artifacts were validated before evaluation GT. Independent validation rescored2,880 frozen masks across the original baseline, candidate and Slot Attention and reproduced means/bootstrap statistics with maximum error5.55e-16. Each seed also passed exact mask/frequency replay on its registered replay image. The full private audit and model references are preserved; `PUBLIC_RESULT.json` omits raw arrays and private dataset/server metadata.

This lineage uses TRAIN mask supervision. Slot Attention uses its reconstruction-training objective, so this is not an unsupervised leaderboard claim. These are repeatedly used validation images, not an untouched final test. Cross-layer gating causality remains unproven. Spike-to-Kuramoto and synchrony-to-graph feedback remain disabled; local previous-spike dendritic input and membrane-reset paths remain present.

SW0237 is now the tracked full320 reference. The previously registered70k training continues as SW0240, preserving its initial SW0227 checkpoint selection and Adam history; its final scores are pending. SW0241 separately tests downstream event information with these same accepted SW0232 models.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `0efd85442cd34fada3c2a32712043a163526a8b8352ef8607df00c8c7779a3c9`

````markdown
# SW0237 전체320장 평가 시작

235번에서 검증한 실제 SW0232 누적512회 모델을 seed0/1/2 그대로 사용한다.
64장은 검증 후 재사용하고 나머지256장만 새로 추론한다. offset0의 mask와
frequency exact replay도 요구한다. 추가 학습은0회이며 기존 고정320장 계약과
comparable Slot Attention reference를 유지한다.

로컬15검사, 서버15검사와 CLI3개를 통과했다. 서버의 실제 CPU admission3개에서
checkpoint SHA와 source binding, CUDA 미초기화, 추가 optimizer update0을 확인했다.
세 독점 GPU 큐가 등록됐으며 관측 시 seed0/1은 GPU1/2에서 실제 worker가 살아 있고,
seed2 supervisor는 빈 GPU를 기다리고 있었다. 이미 새 mask 생성도 시작됐다.
이는 시작 증거이며 전체320장 결과가 아니다.

최초 서버 사전 검사에서는 incumbent 결과의 로컬 보관 경로만 가정한 문제가
검출돼 GPU를 시작하지 않았다. 서버 원본/로컬 archive 경로를 모두 검증하도록
고치고 실제 CPU admission에서도 확인했다. 실패 소스/상태는 서버
`results_archive/source_revision_v1/`에 보존했으며 수정 후15검사를 다시 통과했다.

`START_EVIDENCE.json`에 배포10개 파일 SHA, 실제 CPU복원3개와 checkpoint 식별자,
프로세스 생존·GPU 시도 기록을 보존했다. 원본 RGB·GT·checkpoint는 포함하지 않는다.

````

</details>
