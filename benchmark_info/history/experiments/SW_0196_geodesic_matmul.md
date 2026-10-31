# SW_0196_geodesic_matmul

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `4068a65438d6892a3b196bd10a87a93332890163353ddfeb182c8671ac78c30e`

````markdown
# SW0196: 동일 soft geodesic 수식의 행렬곱 계산 검증

194번 학습 및195번 진단은 변경하지 않는다. 성능 향상 모델 실험이 아니라,
native64 mini 학습을 더 빠르게 반복할 수 있는 계산 구현 검증이다.

기존 soft relaxation은 모든 i,k,j 조합을 행 단위로 만들고 logsumexp한다.
K=exp(-D/t)라 놓으면 같은 수식은 -t*log(K@K)+t*log(4)이다.
cosine squared contrast8, radius, min(old,relaxed), cap16, 세 반복은 유지한다.
근사 이웃 제거, sparse 경로 생략, gate/loss 변경은 하지 않는다.
행렬곱은 실수 수식에서는 동등하지만 부동소수점 계산 순서가 달라지므로
동일한 최종 mask라고 미리 주장하지 않는다.

안전 범위: 입력 float32/float64, N<=4096, steps<=3, 유한한 입력과 비음수
Euclidean 거리, 초기 D/t<=256. 지수·행렬곱·로그만 float64에서 계산하고 각
relaxation을 원래 dtype으로 반환한다. Source는 radius 안에서 cap보다 큰 edge를
만들 수 있으므로 처음부터 cap으로 자르지 않는다. Contrast8/radius1.5/t0.5에서는
최대 D/t=99로 float32의 정상 표현 범위를 벗어나는 지수가 가능하다.
초기 K는 [exp(-256),1]. D는 min 연산으로 증가하지 않는다. 세 번째 반복의 곱은
최대 N^7/4^6<=2^72, 최소 exp(-512)로 float64의 정상 표현 범위 안이다.
더 긴 반복, fp16, 더 큰 dynamic range를 묵시적으로 처리하지 않는다.

CPU에서 reference chunk 연산과 값 및 z/거리/contrast gradient를 비교한다.
사전 허용치: float64 값1e-10/gradient1e-9, float32 값 absolute2e-5 및
relative2e-5, gradient absolute2e-4 및 relative2e-4. 채택 조건을 사후에 느슨하게
바꾸지 않는다. 이는 실제 native64 GPU 및 학습 결과 동등성 검증을 대체하지 않는다.

다음 실제 GPU 검증은 독점 idle GPU에서만 실행한다. 고정 TRAIN 인덱스0,4,8,12의
완료된 TRAIN8 모델을 읽기 전용으로 사용한다. 같은 RNG로 두 계산의 geodesic,
adjacency 및 전체 T1024 spike/Q mask를 비교하고 시간/peak memory를 기록한다.
각 이미지의 최종 mask가 완전히 같아야 production 후보가 된다. 실제 gradient와
동일 minibatch optimizer 제안의 동등성도 별도로 통과하기 전에는 학습에 적용하지
않는다. FG-ARI 개선 주장은 하지 않는다. CPU 검사만으로 production을 변경하지 않는다.

실제 GPU 후보 판정은 네 mask의 완전 일치, geodesic/adjacency/Q의
absolute·relative tolerance2e-5 통과, 첫 이미지의 비영 z gradient의 relative
L2 error<=2e-4, maximum absolute error<=2e-4 및 cosine>=.9999, 전체 prediction
시간의 합 기준1.5배 이상 단축을 모두 요구한다. adjacency support의 변화 개수도
공개한다. 이 단계가 통과해도 optimizer 제안 동등성은 아직 미검증 상태이다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `624c110fdf3c53762ee90465d7fdc4c0d23ee7eba4713f3d0ccb6c283e2b0580`

````markdown
# SW0196 geodesic matrix products

Read-only numerical/throughput experiment; not a new segmentation result.
CPU reference-value and gradient tests precede deployment. GPU admission and
actual four-image results are recorded separately under results_archive.

The queue depends on SW0195 completion to preserve its priority. It then admits
the immutable completed TRAIN8 model on CPU and requests an exclusively idle GPU.
Native64 gradient comparison and four full production mask pairs run without any
optimizer updates. Completed image pairs resume after a foreign owner joins.

Float64 exponential/matrix/log products preserve source edges above cap16;
they are not clipped early. Final masks must match exactly and values must pass
the registered tolerance. Whole-model optimizer equivalence remains a separate
requirement before any production training can use the candidate implementation.

The current194/195 models, sources and checkpoints are unchanged.

2026-10-17 user scheduling update: the original unstarted waiter was stopped
gracefully and preserved. `queue --ready-now` uses `queue_all_gpus` and removes
the optional195 dependency. It scans all four GPUs for an exclusive idle lease.
The operator, model, numerical gates and benchmark cohort remain unchanged.

Completed four-image GPU comparison, independently checked against actual saved
mask pairs. Whole predictions133.789s ->80.800s, speedup1.65581; geodesic portion
72.573s ->3.434s, speedup21.1328. Dense z-gradient relative error2.063e-6,
cosine .999999999998 passed the registered local gradient gate.

Production gate FAILED: all four Q arrays failed the registered forward tolerance
(max errors .8748,.2788,.0332,.5916), although distance/adjacency passed and graph
support was identical. Two of four final masks differed; the mismatches also
changed foreground support, so they are not merely different instance IDs.
This is evidence that local operator/gradient agreement was insufficient to
preserve the full recurrent spike/readout computation. No optimizer-equivalence
claim, no GT performance improvement claim, and no production training adoption.
The original operator remains in199 matched training.

````

</details>
