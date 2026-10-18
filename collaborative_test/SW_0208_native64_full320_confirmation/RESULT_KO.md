# SW0208: 고정320장 확대 평가 완료

64×64 patch, 전체128 RGB/no crop, validation IDs1320–1639,
2px modal GT(동률 작은ID), 실제1024-step spike mask/settle512다.
두 모델의 모든320장 예측과 Slot 세 seed 예측을 검증한 뒤 평가 GT를 읽었다.
새 학습이나 threshold 변경은 없다. 등록된 원본193/204 mini와 겹치는
48개 mask는 정확히 재현됐다. 독립 metric 구현과 최대 오차2.22e-16이다.

| 모델 | seed 범위 | FG-ARI | Foreground IoU | Object IoU |
|---|---|---:|---:|---:|
| 기준 warm192 | 0 | **0.877709** | **0.791224** | **0.696284** |
| component-credit204 | 0 | 0.876242 | 0.790093 | 0.693262 |
| comparable Slot Attention | 0 | 0.636265 | 0.205508 | 0.184986 |
| comparable Slot Attention | 1 | 0.846161 | 0.197193 | 0.212411 |
| comparable Slot Attention | 2 | 0.878278 | 0.191901 | 0.188211 |
| comparable Slot Attention | 0/1/2 평균 | 0.786902 | 0.198201 | 0.195203 |

204 minus warm의 image-paired bootstrap95% CI(10000회, seed208):

- FG-ARI: −0.001467, CI[−0.006263,+0.003277].
- Foreground IoU: −0.001131, CI[−0.004649,+0.002560].
- Object IoU: −0.003022, CI[−0.008057,+0.001870].

모든 CI가0을 포함한다. warm의 세 평균이 조금 높으므로 등록한 선택 규칙에
따라 후속 학습의 seed0 기준 checkpoint는 warm192로 유지한다. 이 비교는
학습량과 objective가 다른 두 checkpoint의 확인이며, 특정 loss가 성능을
해친다는 독립적인 인과 실험이 아니다. 동일 학습량의202 control은 기존
204 mini 실험에 있으며 이번 전체320 비교에는 포함하지 않았다.

## 16장의 .90은 무엇을 뜻했나

같은 고정16장에서는204 .900539, warm .924401이었다. 전체320장에서는
각각 .876242/.877709다. **16장에서 .90을 넘은 것은 사실이지만 전체320
성능이 .90인 것은 아니다.** 평가 이미지 수를 늘린 결과이므로, 학습
데이터 증가가 성능을 해쳤다거나 train overfit이 입증됐다고 해석하면 안 된다.

우리 seed0는 Slot 세 seed 평균보다 높지만, Slot의 최고 FG seed2 .878278과
비슷하다. 우리 독립 seed1/2 전체320 결과가 없으므로 세 seed 평균 승리나
native64 .90 목표 달성을 주장하지 않는다. 이320장은 반복 선택에 사용한
validation이며 untouched test도 아니다.

## 학습 범위와 다음 기준점

두 모델은 기존70k core 학습 lineage와32-grid encoder/core 학습을 물려받은
**supervised** 모델이다. 그 뒤 native64 적응에 사용한 추가 고유 TRAIN
이미지는 warm8장/20416장이다. 이번208은0 update의 평가만 수행했다.
새로운 native64 70,000장 학습 완료 결과로 표기하지 않는다.

후속 기준은 실제 warm checkpoint의 .877709/.791224/.696284다. 새70k
학습은 아직 등록하지 않았다. 기존204 기록의 B4 proposal은 약441–445초,
gradient forward/backward만 약285–287초였다. 이 microbatch1 절차를
70k 전체에 단순 반복하면 약90 GPU-days/한 pass의 계산량이므로,
현재 네 GPU 큐가 단일 학습을 이미 네 GPU로 병렬화한다고 해석해서는 안 된다.
이 수치는 소수 batch에서 외삽한 비용이며 실제70k 소요 시간 측정이 아니다.
따라서 확대 학습은 geometry/최적화/비용을 명시한 별도 실행으로 기록해야 한다.

증빙: `results_archive/completed_v1/`의 summary, 여덟 frozen shard protocol,
네 queue 상태·GPU attempt ledger, archive receipt. 데이터·checkpoint·실제
mask 배열과 imagewise 점수는 GitHub에 포함하지 않았다. Goal은 active로
유지하며, gating 인과 기여·데이터 scaling·세 seed .90은 계속 미검증이다.
