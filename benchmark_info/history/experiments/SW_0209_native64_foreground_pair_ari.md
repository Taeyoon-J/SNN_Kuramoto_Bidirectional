# SW_0209_native64_foreground_pair_ari

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `a36e4bf64c062b5dedf8a4f19655245010e5e781b31493d29d05efa0c387edf8`

````markdown
# SW0209: FG pair-ARI surrogate를 실제 spike loss에 추가

208 전체320 seed0의 기준 warm192 .877709/.791224/.696284를 보존한다.
새 학습은 같은 실제192 model/Adam/CPU-CUDA RNG에서 분기한다. Core,
CNN, gamma ordering, graph,1024-step dynamics/settle512/native64 readout은
변경하지 않는다. 원본 QCE와 모든5개 family 학습을 유지한다.

QCE는 class 평균 affinity의 상대 순위를 학습한다. 이번에는 GT foreground
안에서 Q를 soft pair coassignment로 보고, chance-corrected pair agreement
loss를 weight1로 추가한다. GT background pair와 diagonal은 제외한다.
Q가 hard partition의 동치 행렬이면 이 값은 실제 FG-ARI와 정확히 같지만,
일반 Q는 partition이 아니므로 **실제 최종 mask ARI의 미분 가능한 동일식이
아니다.** 그 proxy가 최종 classifier에도 도움이 되는지 검증하는 실험이다.

M=foreground unordered pairs, T=GT same-object pairs, P=sum(Q_ij),
I=sum(GT_same*Q_ij), E=T*P/M 이다. loss=1−(I−E)/(.5*(T+P)−E).
상하 Q는 평균하고 float64로 reduce한다. 기존 QCE temperature.1과
계수1을 고정하며 validation threshold/LR/가중치 sweep은 하지 않는다.

추가 TRAIN은 기존204/202와 같은 고정8개(indices8..15), B4 두 proposal,
micro1/live-tail64다. 유한 Adam step은 유지하고 실제 descent 여부를 따로
기록한다. Source의 TRAIN8을 포함한 논리적 추가 adaptation은16장이다.
실제 masks를 만들 때 GT, GT object count, 이 surrogate를 사용하지 않는다.

비교는 완료된202 finite-Adam 원본 QCE control과 warm192 모두를 사용한다.
고정16 development offsets0,2,..30/seed0, inference RNG193000+offset,
실제 masks의 세 metric과 paired95%CI를 평가한다. Candidate−control FG≥.01,
CI lower>0, candidate−warm FG≥0, 두 ObjectIoU delta≥0, 실제 CNN update와
new accepted update가 있어야 확대한다. 실패하면 같은계수의 작은 rescue
sweep은 닫는다. 70k 신규 학습이나 세 seed native64 .90이라고 부르지 않는다.

기존191의 discrete error-mined squared penalties,199의 absolute calibration,
204의 component QCE와 다르다. 이번에는 foreground pair counts의 chance
correction/FP-FN 균형이 직접 loss를 결정한다. 과거 실패를 성공으로 재분류하지
않으며, 실제 readout과 proxy가 불일치할 가능성도 결과에 명시한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `a1b362f68c110416652e55f39e23fe65c3cd902bac188405afc9b89b55874cc9`

````markdown
# SW0209 foreground pair-ARI loss

**COMPLETE, expansion gate failed.** Fixed16/seed0: candidate
.901220/.862279/.736022; matched QCE202 .900148/.859586/.728138;
warm192 .924401/.859896/.727638. FG gain versus matched control+.001072,
CI[-.000057,+.003147]. The tiny mean improvements are not established gains.
No full320/seed expansion or coefficient rescue of this recipe. Warm192 remains
the full320 seed0 reference. Candidate checkpoint and all evidence are preserved.

Actual two new Adam updates applied, both descent; all five families and CNN
learned. The standalone new term reached emitted spikes with norms.005706/.006321.
Its TRAIN surrogate losses fell .249533->.237221 and .207382->.197430, but that
did not establish better hard-mask grouping. Independent boundary/Adam/RNG,
48 masks and metrics audit passed (maximum error2.22e-16).
See [RESULT_KO.md](RESULT_KO.md), results_archive/score_v1/summary.json and
results_archive/score_v1/completion_audit.json. No new70k training or .90 claim.

## Historical registration

RUNNING after23 local and23 Linux tests and actual model/Adam/RNG/native4096
CPU admission. Supervisor2249349 and actual GPU1 worker2249387 verified.
New QCE+foreground-pair-ARI surrogate,
unchanged actual native64 spike readout, warm192 source and matched202 control.
See DESIGN_KO.md. No new70k training or improvement is claimed.

The first Linux preflight found an optional sklearn import in the test only.
Before admission/training it was replaced with the existing independent
project ARI implementation. No package installation, model/loss/data change,
GPU training retry or gate relaxation occurred. The failed test log and the
unstarted failure summary remain under results_archive/preflight_v1_failure/.

Registration and actual process receipts are archived. A read-only waiter
PID2250255 waits for this exact supervisor, then independently checks the
completed model/Adam/RNG transaction, standalone new-loss spike credit,
48 actual masks and all metrics. Actual new-loss spike credit and performance
are not claimed until those completed records exist. All GPUs0..3 are eligible
through the shared exclusive leases; foreign owners are not interrupted.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `775750cf7c2138561adcd0003e32cb1eccf6454ca6e1031e761986f8b92328ba`

````markdown
# SW0209 완료: pair-ARI surrogate는 실제 hard-mask FG 개선을 입증하지 못함

고정16장·seed0·native64, 동일한 warm192 실제 model/Adam/CPU-CUDA RNG에서
같은 TRAIN8 추가·B4 두 proposal을 수행했다. 비교 control202도 같은 추가
학습량이다. 모델/gating/encoder/graph/readout 구조와 평가 데이터는 그대로다.
Supervised pilot이며 새로운70k 학습이나 전체320 평가가 아니다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| QCE +foreground pair-ARI surrogate | 0.901220 | 0.862279 | 0.736022 |
| 동일 학습량 원본 QCE202 | 0.900148 | 0.859586 | 0.728138 |
| 추가 학습 전 warm192 | 0.924401 | 0.859896 | 0.727638 |

Candidate−control FG 차이 +0.001072, paired95% CI[−0.000057,+0.003147].
Object IoU 차이 +0.007884, CI[−0.000653,+0.024115]. 세 지표 평균은 control보다
조금 높지만 모든 CI가0을 포함한다. Warm 대비 FG는−0.023182,
CI[−0.069988,+0.000940]다. 등록된 확대 gate는 실패했으며 이 계수의 loss
recipe를 닫는다. 작은 LR/계수 sweep이나 이 모델의70k 확대를 진행하지 않는다.
향후 다른 접근에서 재검토할 수 있도록 checkpoint는 보존한다.

## 실제 loss 전달은 확인됨

두 새 proposal이 모두 적용되고 실제 TRAIN objective가 감소했다.
Adam 누적 step은4(상속2+새2), 실제 CNN도 바뀌었다. 새 pair-ARI 항만의
emitted-spike gradient norm은0.005706/0.006321이며 전체 loss의 gradient를
새 항의 기여로 잘못 해석한 것이 아니다. Pair-ARI TRAIN loss는 각각
.249533→.237221, .207382→.197430으로 감소했다.

따라서 이번 결과를 gradient 단절이나 encoder 고정 문제라고 부를 수 없다.
그러나 gradient가 있다는 사실은 gating의 인과적 유용성이나 hard readout
성능 향상을 증명하지 않는다. Soft Q pair agreement와 최종 discrete
partition이 다르다는 한계는 그대로다. 다음 후보는 계수 조정보다 물체별
assignment 학습을 다루되 기존 foreground를 함부로 제거하지 않는 방식이다.

독립 검산은 실제 완료 checkpoint/model/Adam/history, 초기 tensor transaction,
상속 CPU-CUDA RNG,48개 실제 masks, 세 metric/CI/count 및 새 항의 spike
gradient 기록을 확인했다. 최대 metric 오차2.22e-16.
Model SHA `9b497900ca05e7a1bcab6e26f97eb527c936e6df44b99ccdcb487a048694075d`.
Boundary SHA `9fb38d06400acbce4210183d373ebbee21afac1cfbed0fa693acfe749d3b031f`.
첫 서버 검산-test 의존성 누락은 학습 시작 전에 수정됐고 원본 실패 로그를
보존했다. 이 결과를 얻기 위한 GPU 학습 재시도는 없었다.

현재 full320 기준은208 warm192의 .877709/.791224/.696284다. 위16장
.901220을 새 full320 .90 달성이나 새 최고 모델로 표기하지 않는다.

````

</details>
