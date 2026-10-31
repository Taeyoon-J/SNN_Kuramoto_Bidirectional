# SW_0199_native64_absolute_calibration

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `1a82c1c55a4525bd01f2162206897bb25c760bd887d6a8261e392ad3dbfe6f2f`

````markdown
# SW0199 절대 연결 확률 calibration: matched native64 mini

가설: QCE는 클래스 간 상대 점수만 제한하지만 생산용 signed partition은
Q=.5를 경계로 연결의 부호를 정한다. 상대 QCE가 같아도 실제 mask가 쪼개지는
CPU 예시가 확인됐다. 이것이194 하락의 실제 원인이라는 주장은 아직 하지 않는다.

control은 기존 QCE, calibrated는 QCE + 가중치1의 class/pair-balanced Bernoulli
loss다. 같은 GT 클래스의 off-diagonal Q는1, 다른 클래스는0을 목표로 한다.
배경도 하나의 클래스로 포함한다. positive는 클래스별 평균, negative는 ordered
class-pair별 평균 후, 양/음에 각각 .5의 질량을 둔다. singleton은 positive에서
제외하며 모두 singleton이면 negative만 쓴다. 클래스가 하나면 connected zero다.
diagonal 제외, clamp1e-6. 추가 작은 weight/LR 탐색은 이번 비교에 포함하지 않는다.

두 arm 모두192의 실제 TRAIN8 모델, Adam moments/steps2, CPU/CUDA RNG를 이어받는다.
각 arm은 TRAIN indices8..15에서 B4 두 proposal만 추가한다. 동일5개 LR, clipping1,
원본 geodesic 연산, native64/T1024/settle512/live-tail64를 유지한다.196의 가속 연산은
optimizer 동등성 확인 전에는 도입하지 않는다. control/candidate의 numeric loss는
목적함수가 다르므로 서로 비교하지 않는다. arm 내 same-RNG before/gradient/after
검사와 원자적 model/Adam/RNG 복구, 새 proposal의 실제 S/family gradient를 기록한다.
총16장은 inherited8 + 추가8의 logical gradient exposure다. 데이터 이력 전체가 아니다.

평가: 기존193 development offsets0,2,...30 고정16장. production 원본 readout1024,
원본 foreground/spike classifier, inference RNG193000+offset. GT를 열기 전에 두 arm의
실제 prediction protocol을 먼저 고정한다. frozen193 TRAIN8 warm baseline도 재사용한다.
세 지표와 paired10000 bootstrap(seed199)을 각각 기록한다. 확장 gate는 candidate-control
FG gain>=.01, 95%CI lower>0, candidate-warm FG mean>=0, 두 비교의 objectIoU mean>=0,
그리고 candidate 새 accepted update>=1, model과 CNN의 warm 대비 committed change다.
예측 실패/credit 실패는 performance 개선으로 처리하지 않는다.16장 seed0 mini는 최종
320장/3seed의 .90 달성이나 unsupervised 우승이 아니다. 정답 mask supervised 실험이다.

각 arm 독립 all-GPU queue, 공유 exclusive lease와 foreign owner 양보를 사용한다.
완료된 proposal/이미지부터 이어간다. source fingerprint가 달라지면 실행을 중단한다.
이미지별 GT-derived 수치는 private/ignored, 집계와 코드/감사 기록만 공유한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `3d1c6704e880ac66ab1e3feeed46eee28a2da67b3b6dd770680c2a1615438fbc`

````markdown
# SW0199 matched native64 absolute calibration

Two independent ready queues compare unchanged QCE and QCE plus absolute-Q
class/pair-balanced Bernoulli calibration. Both resume the same actual TRAIN8
model/Adam/RNG and attempt two new B4 proposals. Production gates and readout
stay fixed; this is a supervised mini, not a70k or three-seed final result.

The hypothesis and fixed extension criteria are registered in DESIGN_KO.md.
No performance result is claimed before actual frozen predictions are scored.
Only aggregate results are public; imagewise GT-derived values remain private.

2026-10-17:19 local tests passed with one Linux-only lock check skipped; all20
Linux tests and2 CLI entrypoints passed. Both arms completed actual CPU admission,
preserving source Adam steps2 and matching model/Adam/RNG start hashes. Separate
all-GPU exclusive-lease queues registered: control1973294, calibrated1973323.
At registration both supervisors were verified live and waiting for an idle GPU.
Receipts are in results_archive/source_tests_v1 and registered_v1.

Both arms subsequently started on actual GPUs and independently recorded matching
inherited model/Adam/RNG restore receipts under each arm's started_v1. Calibrated
worker1978134 startedGPU2, then yielded to a foreign owner. Control worker1979735
startedGPU1 after198 completion. The calibrated supervisor remains queued and
restores its last committed boundary when it next obtains a GPU; no source or
model/loss settings were changed to handle the resource transition.

Control completed: both new proposals were rejected, actual Adam remains step2,
final model exactly equals the inherited TRAIN8 model. Its16 saved masks replay
all16 frozen warm masks exactly, without opening GT. Actual S gradients remained
positive. Historical194 first4 proposals are not a substitute for this control:
its proposal4 acceptance differed even though the start model/Adam tensor digest
and measured gradient norms were almost equal. Cause is not established. Use the
actual199 control and preserve its original protocol. Calibrated arm is pending;
no matched performance comparison is available yet.

Both arms subsequently completed. Actual final model/Adam/RNG/history and48
fixed16 masks (control/calibrated/warm) were independently audited, max metric
error2.22e-16. Calibrated arm committed one new update, actual Adam step3 and
CNN changed. Its scores .9004848394/.8612262559/.7355866443 compare with
control/warm .9244014538/.8598964167/.7276384095.

FG delta-.023917,95%CI[-.071883,.001932]. ForegroundIoU delta+.001330 and
objectIoU delta+.007948 also have intervals including zero. The primary FG
extension gate failed. Close this unit-weight calibration recipe and retain the
incumbent; no weight/temperature micro-sweep or broad claim that calibration can
never help follows from this one-seed16-image screen.

````

</details>
