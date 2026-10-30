# SW0251: 배경 결정과 최종 mask 감독의 통제 비교

사용자 요청: 먼저 원래 CLEVR 데이터에서 기존 240 수준의 FG-ARI를 유지하는지 검증한다.
통과한 설정만 Multi-dSprites / Tetrominoes에서 추가 학습 없이, 추가 학습 후 모두 평가한다.
원본 240 checkpoint, 기존 실행 코드, benchmark/ 및 benchmark.md는 수정하지 않는다.

## 가설과 비교

- baseline: 검증된 기존 SW0249 matched35000 terminal SW240 seed0/1/2.
- keep_groups: spike synchrony threshold .5, min size32는 그대로 두고 가장 큰 그룹만 제거하던 규칙을 끈다.
- all_support: 모든 patch를 signed partition에 전달한다. 배경을 지운 것이 아니라 제거 효과를 분리하는 진단이다.
- learned_background: 실제 spike/membrane 특징에서 배경 확률을 학습하고 all_support의 물체 분할을 사용한다.
- residual_mask: 기존 240 물체 masks와 all_support masks를 GT 없이 후보로 만들고, 실제 spike/membrane 특징으로
  최종 soft mask 및 배경 출력을 학습한다. 초기 argmax는 기존 240과 정확히 같다.

Readout 입력은 settle512 이후 실제 네 component의 spike 32-bin 평균/평균/표준편차와
membrane 평균/표준편차뿐이다(144개 특징). RGB, gamma, 정답 object count는 readout 입력이 아니다.
기존 Kuramoto→sinusoidal→dendritic→membrane→spike core는 유지한다.
후보 masks는 기존 모델 출력에서만 생성되며 GT는 학습 loss 및 별도 scoring에만 사용한다.
Readout 후보 수는 모델 출력에 따르며 GT 물체 수로 정하지 않는다.

첫 학습은 core/encoder를 모두 고정하고 readout만 학습한다. 이는 **최종 mask 감독이 readout에 직접 연결되는 비교**이며,
전체 encoder/core까지 mask loss로 end-to-end 학습했다고 주장하지 않는다. 원래 240 보존 및 병목 분리를 우선한다.

Loss: Hungarian matching으로 object ID 순서에 무관한 최종 mask balanced CE + soft Dice,
별도 배경 BCE, foreground same/different pair loss, 기존 partition 내부 일관성 anchor(보존 항).
Hard grouping/seed 선택 자체에는 역전파하지 않지만 soft assignment와 학습형 readout에는 실제 gradient가 흐른다.

## 데이터와 승격

CLEVR의 원래 canonical TRAIN70000 cache를 사용하며 reserved IDs1000..1639는 학습에서 제외한다.
작은 screen TRAIN64 / validation16(1000..1015)에서 시작하고, 유망하면 TRAIN256 / validation64(1000..1063)로 확대한다.
TRAIN은 deterministic subset이며 모든 후보에서 동일하다. 초기 screen seed0, 승격 확인 seed0/1/2.
1024 rollout / settle512 / native64는 유지한다. 70000장 전체 재학습은 이번 초기 비교에 필요하지 않다.

최종 보존 확인은 기존 fixed320 IDs1320..1639, seed0/1/2와 동일한 patch metric이며
mean FG-ARI(new)-mean FG-ARI(original240) >= -0.005가 수치상 승격 조건이다.
이 fixed320은 과거에 반복 사용된 validation이지 untouched test가 아니다. 평균만으로 유의한 동등성을 주장하지 않고
seed별 값·paired image 차이·bootstrap CI와 나머지 두 지표도 함께 보고한다.
학습/설정 선택은 TRAIN/새 validation에만 의존하며 fixed320으로 loss sweep하지 않는다.

새 데이터는 SW0250과 동일한 원본 benchmark 평가 규칙, 기존 splits와 seeds0/1/2를 유지한다.
Multi-dSprites(colored_on_colored) grid64, Tetrominoes native64→원본 modal grid32→35pixel adapter.
Zero-shot은 해당 데이터 mask를 한 번도 학습하지 않은 CLEVR 개선 모델/readout을 사용한다.
Fine-tune은 같은 seed의 CLEVR 개선 모델에서 출발하며 SW0250의 encoder/projection TRAIN6000×10 적응 후
해당 adapted core에서 추출한 TRAIN spike 특징으로 readout도 mask 감독한다. TEST/validation mask는 학습에 넣지 않는다.
이렇게 추가된 readout 적응 비용은 별도로 명시하고 SW0250과 완전히 동일한 학습량이라고 주장하지 않는다.
TEST1000의 결과는 우선 고정 앞360장, 이후640장을 모두 완료해 보고한다.
모든 새 데이터 최종 평가 전에 원본 preflight.py와 보호 파일 SHA 검사를 실행한다.

## 실행/기록

생성물은 patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0251_background_mask/ 및
서버 collaborative_testing/generated/SW0251_background_mask/에 기록한다.
Data0의 여유 공간 부족으로 큰 활동 캐시는 본인 /export_home/kevinswk/collaborative_testing/generated/
SW0251_background_mask_cache/에 저장하고 실험 archive/cache에서 연결한다. 원본 데이터/기존 결과를 삭제하지 않는다.
서버 파일 전송 전 CPU 기능 검사, checkpoint 복원/forward smoke, 실제 CUDA 학습 gradient 검사를 진행한다.
GPU 확보는 타인의 프로세스를 건드리지 않고 실제 여유 메모리에 따라 공유 실행한다.
큐는 서버에서 지속 실행하고 원격 연결 확인 자체만을 위한 사용자 메시지는 보내지 않는다.
연결 복구/실패만으로 학습을 중복 시작하지 않는다.
