# SW_0225_native64_joint_projection_validation64

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `ffec97ecc9d94638cea3173a63f6747a0b1076245687fa1fbf52c972ae26f368`

````markdown
# SW0225: 고정223 모델의 평가 범위를64장으로 확인

223의 DEV16 세 seed 평균 .859634/.645611/.706340은 source 대비 세 지표가
올랐지만 FG 차이 CI[-.000361,.040940]가0을 포함해 사전 full320 gate는 실패했다.
encoder-only 대비 FG 차이는+.005362 CI[.000062,.012050]였다. 모델을 더
학습하거나 LR/loss를 바꾸지 않고 한 번만 더 넓은 validation으로 확인한다.
이 별도 확인은223의16장 gate 실패나224 결과를 변경하지 않는다.

223의 최종 seed0/1/2 checkpoint를 고정한다. IDs1320..1383의 첫64장,
canonical128/no crop/2px modal GT/1024 settle512 원래 spike readout,
RNG193000+offset이다. 기존16장 offset0,2,..30 mask는 hash 및 RGB를 대조해
재사용한다. offset0은 native forward로 재생해 bitwise mask 일치를 확인한다.
추가 native forward는 seed당49장(새48장+재생1장), 추가 학습은0이다.
baseline은 동일 source 모델의 기존 full320 mask에서 같은64장을 쓴다.

세 모델/64장 최종 protocol을 모두 동결한 뒤 GT를 읽는다. paired image
bootstrap10,000회 RNG225. FG source 대비+.01/CI하한>0이면 full320 FG-track
후보, 여기에 두 IoU 비감소면 balanced 후보로 기록한다. final 목표는 같은
full320 평가의 세 seed 평균으로 Slot의 세 지표를 엄격히 초과하고 FG .90도
검증하는 것이다. 이64장 확인만으로 final 성공이나70k scaling을 주장하지 않는다.
고정 validation을 반복 사용하는 연구 단계이며 untouched test가 아니다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `603cecaf06c84d29950bc85248842462d7c4d61b5348ee0ff4521c518b909096`

````markdown
# 동료 연구 확인 — 2026-10-18

225번 배포 후에도 origin/patch_v2를 fetch했다. HEAD는
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 변동이 없다.
이번에는 source/222/223 control의 동일16장 비교에서 나온 불확실성을 다룬다.
기존223 세 checkpoint를 고정하고 prefix64 validation을 한 번만 확인한다.
새 hyperparameter/model 선택, 원래16장 gate 수정, 단일 seed headline 승격은 없다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `75d0f883e66b70b5520584f6961c7d0f21549564d52a969396dc0b94e194d863`

````markdown
# SW0225

One secondary64 validation of fixed223 checkpoints. No extra training or new
loss selection. Reuse the exact old16 predictions after RGB/hash checks and
one native replay per seed; predict48 additional images. The original16 gate
failure and full320 reference remain unchanged. See DESIGN_KO.md.

Actual start2026-10-18:7 local/7 Linux tests,2 CLI checks and three actual fixed223 model restorations/canonical64 RGB checks passed. Original mini masks replay exactly on active GPU workers. Three exclusive idle GPU queues registered; all223 checkpoints are fixed. Expected49 native forwards/seed includes one fidelity replay and48 new images; original16 reused. No extra training or70k run. Original16 gate and full320 incumbent .848615/.592533/.672047 unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `639ac417bf5ffb728c049aad87448fc026d378251bb8c79004e79f8c55fa4858`

````markdown
# SW0225 완료: 고정 모델의 64장 확대 평가

223번 encoder/core 입력 projection 공동학습 checkpoint를 고정하고 canonical128,
native64, IDs1320..1383에서 세 seed를 평가했다. 추가 학습은 0회다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 같은 이미지의 원래 모델 | .864270 | .571015 | .669769 |
| 223번 공동학습 모델 | **.877164** | **.597239** | **.670309** |

FG 차이 +.012894의 paired95% CI는 [.004403,.022515], FIoU 차이 +.026224는
[.002066,.051886]이다. Object IoU 차이 +.000540의 CI[-.013955,.015579]는 0을
포함하므로 그 지표의 개선은 불확실하다. 세 평균은 모두 비감소이고, 사전 정의된
FG 및 balanced full320 후보 기준을 통과했다. 이는 고정된 세 모델에 대한 이미지
bootstrap이며 seed 모집단의 불확실성 구간은 아니다.

| seed | 원래 FG-ARI | 공동학습 FG-ARI | 공동학습 FIoU | 공동학습 Object IoU |
|---|---:|---:|---:|---:|
| 0 | .911894 | .911601 | .807920 | .694423 |
| 1 | .841229 | .860112 | .521367 | .666115 |
| 2 | .839686 | .859778 | .462432 | .650389 |

FG 개선은 주로 seed1/2에서 나타났다. seed0은 거의 같고 Object IoU는 소폭 낮아졌다.
이것만으로 모든 gating의 인과적 기여나 데이터 scaling을 입증하지 않는다.

원래 16장 gate 실패 기록은 유지한다. 이번 별도64 확인의 범위와 기준은 GT를 읽기
전에 정했으며, 기존 평가에서 좋았던 모델을 더 넓게 확인한 exploratory validation이다.
반복 사용한 validation이므로 untouched test라고 주장하지 않는다.

세 frozen protocol 및 실제 checkpoint SHA를 모두 확인한 후 GT를 읽었다.
candidate/baseline 총384개 mask를 독립 재계산했고 최대 지표 차이는 2.22e-16,
로컬 평균·CI 재계산 차이는 0이었다. 16개 원래 mask/seed의 재사용과 bitwise replay도
검증했다. seed2의 THP 복구 이력·원래 실패 상태·추가 replay는 보존했다.

**64장 평균 .8772를 전체320장 최고 .8486과 직접 비교해 최고 기록을 갱신하지 않는다.**
전체320장·세 seed의 FG .90 목표는 아직 검증되지 않았다. 같은 checkpoint의 나머지
256장/seed를 확인할 후보로 선정했다. 70,000장 재학습은 이번 평가에 포함하지 않았다.
226번은 별도로 같은 업데이트 예산에서 고유 TRAIN64→512 효과를 평가 중이다.

근거: results_archive/completed_v1/summary.json, completion_audit.json,
archive_receipt.json 및 best_model_tracking_20261008.json의 별도 secondary64 항목.

````

</details>

## RUNTIME_DIAGNOSIS_KO.md

<details>
<summary>RUNTIME_DIAGNOSIS_KO.md</summary>

Source SHA256: `4afd0c640c42430e4b384630079c4f3578b5a5ec4daa8c71edcdad7da58f0990`

````markdown
# SW0225 저장 지연 진단 — 평가 결과 아님

seed0/1의 64개 mask 생성은 완료됐고 seed2는 29개 저장 후 긴 계산 구간에 들어갔다.
이는 최종 점수나 성능 개선의 증거가 아니다. seed2 worker PID 2633054의 실제
실행 상태와 증가하는 CPU 누적 시간을 확인했다. GPU1에는 해당 worker만 있었고,
조회 시 GPU 사용률은 0%, worker CPU 사용률은 약 99%였다. 오류 로그는 없었다.

한 번의 제한된 native 스택 표본은 NumPy의 `_multiarray_umath` 내부를 가리켰다.
이 표본만으로 정확한 Python 호출 위치나 근본 원인을 단정할 수 없다. 따라서
GPU 연결 실패나 프로세스 종료로 간주하지 않으며, 미완료 mask를 임의로 건너뛰거나
기존 worker를 재시작하지 않았다. 64장·세 seed 전체가 완료되기 전 점수는 발표하지 않는다.

첫 debugger 조회는 제한 시간을 넘겨 해당 진단용 debugger만 종료했다. 이후
외부 debug symbol 조회를 끄고 강제 종료 제한을 추가한 표본 조회는 정상 종료했다.
조회 직후 원래 worker의 실행 상태와 `TracerPid=0`을 확인했으나, 뒤이은 재확인에서
worker에 정지 상태가 남아 있음을 발견했다. 같은 PID에 SIGCONT를 보내 재개했고,
CPU 누적 시간이 다시 증가하며 `TracerPid=0`인 것을 확인했다. 이후 gdb는 사용하지
않고 `py-spy dump --nonblocking`으로만 조회했다. 모델 코드,
checkpoint, 입력, RNG 규칙, readout 및 저장된 mask는 변경하지 않았다.

후속 실행에서 지연을 재현할 경우 native forward와 CPU partition 각각의 경과 시간을
분리해서 기록한다. 현재 표본만으로 readout 변경이나 수치적으로 다른 최적화를
도입하지 않는다. 기존 320장 incumbent 및 원래 16장 gate 실패 기록도 유지한다.

## 메모리 정책 복구

비정지 Python 스택은 `signed_pair_energy`의 `np.log(q_bar/(1-q_bar))`를 반복해서
가리켰다. 별도 합성 진단에서도 4096×4096 float64 배열의 첫 쓰기가 15초 제한을
넘겼다. 해당 진단 프로세스에만 `prctl(PR_SET_THP_DISABLE,1)`을 적용했을 때는
같은 크기 배열 쓰기가 0.097초, 로그 연산이 0.426초에 완료됐다. 작은 1024개
로그 연산과 가상 배열 할당 자체는 기본 환경에서도 즉시 완료됐다. 호스트 가용
메모리는 약 85GiB였고 세션 및 상위 cgroup 메모리 제한은 무제한이었다. 따라서
일반적인 모델 연산량이나 메모리 용량 부족으로 설명하지 않고, transparent huge
page와 관련된 큰 배열 쓰기 지연에 대응한다. 커널 내부의 유일한 근본 원인을
증명했다는 주장은 하지 않는다.

`recover_memory.py`는 기존 worker와 supervisor의 종료를 확인한 후에만 별도 복구
상태 경로를 만든다. 원래 실패 상태, 로그 및 29개 mask의 58개 파일 SHA는 보존했다.
서버 전역 설정을 변경하지 않고 복구 프로세스와 자식 프로세스에만 THP를 끈다.
원래 checkpoint, 입력, fingerprint, GPU 독점 정책, 연산식, RNG 및 저장 경로는
유지한다. 복구 worker는 원래 223번의 offset0 mask와 frequency를 bitwise 재현해야
나머지 평가로 진행한다. 이 추가 replay는 기존 225번의 1회 replay와 별도로 기록한다.

원래 큐는 진단 후 정상 종료 신호로 종료되어 failed 상태를 보존한다. 완료 audit는
복구 큐의 완료, 원래 종료 상태 SHA, protocol SHA 및 정확한 replay 증거를 추가로
검사한다. 로컬 테스트 4개 중 Linux 전용 1개는 skip됐고, Linux에서는 기존 전체
테스트를 포함한 11개가 통과했다. 복구는 추가 학습이나 새로운 성능 결과가 아니다.

첫 복구는 실제 mask/frequency 재현을 통과했으나 같은 interpreter에서 CUDA 초기화 후
원래 predictor의 CPU admission을 재호출하여 종료됐다. 이 실패 및 replay를 v1 경로에
보존했다. v2는 replay 후 같은 PID를 원래 worker의 새 interpreter로 exec해 CPU admission
순서를 지키며, GPU lease와 THP 정책은 유지한다. 해당 전체 실행 경로를 mock으로
검사하고 Linux에서는 exec 이후 실제 THP 정책 유지까지 확인했다. Linux 전체 테스트
12개가 통과했다. v2 등록 전 29개 결과 SHA를 다시 대조했고 이전 프로세스가 모두
종료됐음을 확인했다. v1 실패를 성공이나 새로운 학습 결과로 취급하지 않는다.

````

</details>
