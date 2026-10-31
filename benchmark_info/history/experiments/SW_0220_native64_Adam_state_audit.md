# SW_0220_native64_Adam_state_audit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `b1b1368fcf15412f4f1a4b2b284f41f5cca5c58bf18d76d3a7644b009243504b`

````markdown
# SW0220: 새 objective 전환 시 Adam 상태의 한 번 제한된 진단

218은 실제 spike/모든5개 family의 auxiliary gradient가 존재했지만 FG를 개선하지
못했다. 첫 새 proposal의 같은-RNG combined TRAIN loss는2.819465->2.947188로
상승했다. 이전 objective의 Adam 상태를 물려받은 영향인지 아직 모른다.
Loss weight/LR/margin/epoch sweep 대신 같은 gradient에서 optimizer 상태만
비교한다. Reset은 first/second moments와 step counter를 모두 초기화하므로
특히 momentum 하나만의 인과 효과라고 하지 않는다.

실제 source192 TRAIN8 seed0 model/Adam step2/원래 CPU-CUDA RNG를 복원한다.
218의 첫 새 TRAIN indices8..11, B4/micro1, 같은5개 learning rate/clip1,
원래1024/settle512/live64,218의 component+energy-ranking loss다. Before-step
production partition/reference/scale은218 helper로 한 번 mining하고 고정한다.
4장 중 마지막 값/GT object count를 inference에 넣지 않고 DEV GT도 열지 않는다.

한 번의 gradient forward/backward에서 clipped gradient를 저장한다. 두 arm은
같은 initial model/RNG, 같은 gradient/context/projection에서 시작한다.
Inherited arm은 실제 Adam moments/counter를 유지한다. Reset arm만 state를
비우며 LR/betas/epsilon/parameter groups는 유지한다. 각 임시 step 후 같은
RNG의 실제 full-forward loss를 측정한다. 새 gradient를 arm별로 구하지 않는다.
원래218의 before/inherited-after/context hashes 재현 여부도 따로 보고한다.

Surrogate gradient와 실제 projected parameter delta의 내적, family delta,
Adam counters, 실제 forward loss를 기록한다. Hard spike의 straight-through
surrogate 내적은 실제 binary forward의 미분/단조 감소 보장이 아니다.
두 arm 뒤 model/Adam/RNG를 원래 source 상태로 정확히 복원하고 digest를
검증한다. 원래 source/218 weights/checkpoint/masks는 바꾸지 않는다.
임시 optimizer proposal은2번, committed update는0이다. 고유 TRAIN4장,
native full-forward image exposures16(사전4+gradient4+두 post8), DEV 평가0.
새로운 학습 모델/FG score/70k 결과를 만들었다고 하지 않는다.

Local/server tests에서 actual nonempty Adam의 opposite-state 사례, 동일 gradient,
state/counter reset, 예외·nonfinite 시 exact transaction restoration, projection 뒤
delta, JSON serialization, 실제 control import와 GPU command를 검증한다.
실제 CPU source model/Adam/RNG/frozen-control admission 후 exclusive idle GPU0..3
queue로 실행한다. 외부 owner에게 양보하며 source fingerprint를 유지한다.

Fixed follow-up criterion은 reset의 실제 loss가 before와 inherited-after 모두보다
max(1e-7,1e-6*abs(before)) 이상 낮은 것이다. 이것은 TRAIN-only 후속 학습의
근거일 뿐 validation 개선이나 .90 달성이 아니다. 두 arm이 모두 나쁘면 이
optimizer-state recipe는 닫고 LR/beta/반복횟수 micro-rescue를 하지 않는다.
조건을 통과해도 실제 segmentation 개선은 새 matched mini와 full320 세 seed로
확인해야 한다. 현재 full320 .848615/.592533/.672047 incumbent은 유지한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `5288f9e4a350ee2234a0840824f5779d2fc4710d5fdcb05fd03bf943015938c3`

````markdown
# SW0220 native64 optimizer-state diagnosis

One bounded TRAIN4 comparison after218's first uphill update: keep inherited
Adam state versus reset its moments/counter, with the same model, clipped
gradient, frozen ranking context, learning rates and projection. Measure actual
same-RNG full-forward loss, then restore source model/Adam/RNG exactly.
Two temporary proposals, zero committed updates, no DEV scoring or new70k run.
See DESIGN_KO.md. This diagnosis is not a new segmentation performance result.

Actual start2026-10-18:8 local/8 Linux tests,2 local/2 Linux CLI checks and real source model/Adam/RNG/frozen-control CPU admission passed. Supervisor2394953, worker2397214 on exclusive GPU1. Source and actual NVIDIA/process bindings are archived. Peer patch_v2 rechecked at7a21632, unchanged. No new segmentation score or source update is claimed.

Completed: before2.819465, inherited-after2.951871, reset-after2.919585. Both proposals are uphill in actual forward loss; TRAIN-only follow-up gate failed. Same gradient/hyperparameters and exact source model/Adam/RNG restoration were verified. No reset-Adam training expansion and no new segmentation score. See RESULT_KO.md.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `3ca333b2829421dd4d2dafeb9881634ca100641b7d4b922e4979c36ed4efe3f7`

````markdown
# SW0220 완료: Adam 상태 초기화만으로 실제 loss 감소를 회복하지 못함

218과 같은 source192 native64 model/Adam step2에서 TRAIN indices8..11의
한 B4 gradient를 구하고 동일한 clipped gradient를 두 arm에 복사했다.
초기 모델/context/RNG/LR/betas/epsilon/projection은 같고, reset arm만 Adam
first/second moments와 counter를 초기화했다. 실제1024-step forward loss다.

| 상태 | 같은 TRAIN4 combined loss |
|---|---:|
| 업데이트 전 | 2.819465 |
| Adam 상태 유지 후 | 2.951871 |
| Adam 상태 초기화 후 | 2.919585 |

초기화 쪽이 유지 쪽보다 낮지만 두 경우 모두 업데이트 전보다 높다.
사전 고정 TRAIN follow-up gate는 false이므로 이 결과를 근거로 reset-Adam
추가 학습을 등록하지 않는다. Beta/LR/update-count micro-rescue도 하지 않는다.
이는 한 source/한 batch의 결과이며 Adam 초기화가 일반적으로 무효라는 증명은 아니다.

두 step의 surrogate-gradient/projected-delta 내적은 각각 -.0007370/-.0010866이었다.
즉 두 경우 모두 surrogate 방향으로는 내려가는 업데이트였지만 실제 binary
full-rollout loss는 증가했다. 이 결과만으로 temporal truncation, hard events,
step 크기 또는 수치 민감성 중 어느 하나가 원인이라고 정하지 않는다.

## 범위와 검증

실제 Adam counters는 유지 arm2->3, reset arm0->1이었다. 동일 gradient의
정확한 복사와 group hyperparameter 불변을 확인했다. 진단 뒤 실제 원래
model/Adam/RNG를 복원하고 transaction digest 및 원래 source SHA와 대조했다.
원래 checkpoint 파일도 변하지 않았다. 임시 step2개, committed update0,
native image-forward16, 고유 TRAIN4, DEV 평가0이다. 새 segmentation 점수나
새로운 trained model/data-scaling/70k 결과가 아니다.

218의 before loss와 네 이미지의 frozen context hashes는 같았다. 다만 유지
arm의 after loss는218의 기록보다 .00468322 높았으므로218 optimizer step의
bitwise 재현이라고 하지 않는다. 이 진단 안의 두 arm은 실제 동일 gradient와
동일 초기 상태를 사용한 비교다. 수치 차이의 원인을 추가로 확정하지 않았다.

Local/server 각8개 tests, 각2개 CLI, 실제 source model/Adam/RNG와 고정 control
CPU admission을 통과했다. 전체 보고서/예외 복원/비유한 step/실제 owner API를
로컬에서 검사한 뒤 전송했다. 실제 GPU1에서617.04초에 완료됐다. 사후 CPU
감사는 실제 saved source content와 보고된 복원 SHA, counters, 판정의 산술을
독립 검산했다. GPU probe를 별도로 다시 실행했다는 주장은 하지 않는다.

Full320 최고 .848615/.592533/.672047과 세 seed FG .90 목표 미달은 유지한다.
Optimizer-state branch는 닫고 다음 모델 시험은 representation/실제 물체 연결의
정보 보존에 근거해 고른다. Naive scalar-Q threshold가 false-merge와 false-split을
동시에 고친다고 가정하지 않는다. 217의 두 오류에서 Q 부호가 반대로 잘못된
경우가 많았고219에서도 pair AUC 향상이 grouping 향상을 보장하지 않았다.

근거: results_archive/completed_v1/report.json 및 completion_audit.json.

````

</details>
