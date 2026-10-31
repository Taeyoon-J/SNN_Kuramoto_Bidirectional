# SW_0200_native64_long_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `b644f9dbc24d4d8a40432acb9ee09b0a292637d285eb34a308dffab0b69a7ce0`

````markdown
# SW0200 native64 settled-window time credit

가설: production 통계는 T1024 중 안정화 이후512시점의 spike를 쓰지만, 현재
직접 시간 역전파는 마지막64시점에만 걸린다. graph/drive는 setup에서 gradient를
유지하되 앞960시점 recurrent state는 detach된다. 이 truncated BPTT 편향이
추가 데이터 학습의 유용성을 제한할 가능성을 시험한다. 원인 확정은 아니다.

변경은 live-tail64 ->512 하나다. QCE temperature.1, 원래 full1024 forward,
settle512, 모든 gating/graph/CNN 및 생산 readout은 유지한다. 첫512시점의
recurrent history detach는 여전히 남는다. 전체1024 BPTT라고 부르지 않는다.
원본 geodesic 사용;196 가속 버전은 도입하지 않는다.

TRAIN8 source192 실제 model/Adam/RNG boundary2를 복원하고 indices8..15에서
B4 두 proposal을 추가한다. 199 control과 같은 시작 상태, 새 TRAIN8, LR5개,
clip1, same-RNG before/gradient/after accept/restore 정책이다. 각 proposal의
실제 S/five-family gradient를 기록한다. 기존 loss 비교군199 control은 이미
학습과16 mask 생성을 완료했고 두 proposal 모두 거절되어 warm model과 mask가
그대로다. 이 사실을 숨기지 않으며 양쪽 control/warm 대비로 평가한다.

GPU 사전검사: TRAIN index8 한 장에서 no-grad/live512 loss parity 및 실제 S/
모든 family gradient를 확인한다. optimizer step은 없고 model/Adam/RNG를 정확히
복원한다. allocator 최대비율.8을 적용하며 메모리 부족이면 이 후보는 reject하고
학습하지 않는다. 이 B1 gradient 평가는 추가 read-only TRAIN 노출로 별도 기록한다.
총16 unique/logical-proposal exposure는 inherited8 + 새8이며, 모든 forward 또는
gradient 계산 횟수라는 뜻이 아니다. 사전검사는 새 unique image를 추가하지 않는다.

고정 development offsets0,2,...30의16장, RNG193000+offset, 동일 GT/production
mask. 후보/비교군 protocol을 고정한 뒤 GT를 열고 세 지표와 paired bootstrap10000
(seed199;199과 같은 draws) 기록. 확장 gate: 후보-control FG>=.01 및 CI lower>0,
후보-warm FG mean>=0, 양쪽 objectIoU mean>=0, 새 accepted update>=1, 실제 model과
CNN의 warm 대비 committed change. 검증된 후보에만 추가 seed/full320를 진행한다.
이것은 seed0 supervised mini이며 최종 native64 three-seed .90 달성이 아니다.

199 control은194의 같은 QCE 첫4-proposal historical observation과 다르게
proposal4도 거절됐다. 초기 model/Adam tensor digest는 같고 측정된 family gradient
norm은 거의 같지만 역사적 optimizer trajectory의 완전 재현은 입증되지 않았다.
반면 고정 warm weights의16 mask는 과거 warm mask와 모두 정확히 일치했다.
본 비교에서 historical194 model을 control로 바꿔 끼우지 않는다. 동일 protocol의
199 실제 control을 그대로 사용한다. 수치 민감성 원인은 아직 확정하지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `47f7d98d9515846e061cb829cb68a7ca8e6b33a5f4cdb2ffc8f65a8ef0708a34`

````markdown
# SW0200 native64 full settled-window credit

QCE and production inference stay fixed. Direct temporal credit grows from the
last64 to the last512 of1024 steps, covering the entire settled scoring window.
The first512 recurrent steps remain detached. This tests a longer learning
window, not a new gate or a claim of full-horizon exact gradients.

Matched reference is the actual completed199 control, with identical TRAIN8
model/Adam/RNG start and two new B4 proposals. Before training, a one-image
bounded-memory full512 gradient admission must pass and restore all state.
Read-only admission exposure is reported separately. See DESIGN_KO.md.

No performance result or full320/three-seed goal achievement is claimed here.
Raw imagewise GT-derived vectors, prediction arrays and checkpoints stay private.

2026-10-17 registration:14 local and14 Linux tests, all Python ASTs and2 local/
Linux CLI entrypoints passed. Actual CPU model/Adam/RNG matches the199 control's
start, Adam steps2, frozen control masks/protocol checked without opening GT.
Supervisor2000695 is registered for all four GPUs. Actual native64 GPU full512
memory/gradient admission is still pending; CPU tests do not establish GPU fit.

Actual GPU admission subsequently PASSED onGPU1: full512/no-grad forward losses
matched exactly, actual emitted-S and all five parameter-family gradients were
positive. Peak allocation6,431,305,216bytes (~5.99GiB), below the .8 allocator
budget. The read-only probe restored model/Adam/RNG before training. Worker2003084
then started the registered two-proposal training. This is learning admission,
not a segmentation improvement result or a gate-contribution ablation.
Public validation facts are in gradient_admission/public_receipt.json; the
single-TRAIN-image loss/gradient values remain private.

Completed and independently audited actual final model/Adam/history plus48 masks
(candidate/control/warm on16 fixed images), maximum metric error2.22e-16.

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| Full512 credit | .899738 | .857661 | .723357 |
| Matched64 control / warm | .924401 | .859896 | .727638 |

FG difference-.024664,95%CI[-.074037,.001665]. One new update committed (actual
Adam step3), model and CNN changed; healthy credit/committed learning did not
produce a mask improvement on this mini. The registered extension gate failed.
Close this512 recipe, retain the incumbent/live64 setting, and do not extend it
with small tail/LR sweeps. A population degradation or general claim that longer
credit can never help is not established by this one seed/16-image screen.

````

</details>
