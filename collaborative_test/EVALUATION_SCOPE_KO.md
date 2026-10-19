# 최근 native64 결과의 학습·평가 범위 구분

아래 학습 장수는 **기존 checkpoint 이후 해당 teacher 학습 단계에 사용한 고유
이미지 수**다. 모델을 처음부터 그 장수만으로 학습했다는 뜻이 아니다.
업데이트 수는 optimizer.step 횟수이고, 평가 장수와도 다르다.

| 실험 | 추가 단계 고유 학습 이미지 | 해당 단계 누적 업데이트 | 고정 평가 이미지 | 세 seed 평균 FG-ARI |
|---|---:|---:|---:|---:|
| SW0227: 고정 SW0223 모델 평가 | 64 | 128 | 320 | .858916 |
| SW0232: 같은 모델 이어 학습 | 64 | 512 | 16 | .884622 |
| SW0235: 고정 SW0232 모델 평가 | 64 | 512 | 64 | 미완료 |
| SW0236: 데이터 수 증가 비교 | 512 | 512 목표 | 16 목표 | 미완료 |

227번과235번 자체는 새 학습 없이 해당 checkpoint를 평가한다. 232번과236번은
실제128회 checkpoint를 복원해384회 더 학습하므로 누적512회다. batch16,
누적8,192 exposures는 같은 이미지를 반복 본 횟수까지 포함하며 고유 이미지
8,192장이 아니다. 고유64장은 각128회, 고유512장은 각16회 노출에 해당한다.

현재 검증된 최고는 **고정 평가군 전체320장**에서 .858916이다. 여기서 전체는
저장소의 고정 평가군 전체를 뜻하며, 데이터셋100,000장 또는 학습70,000장을
모두 평가했다는 뜻이 아니다. .884622는 실제 측정치지만16장 미니 결과여서
전체320장 결과로 대신 내세우지 않는다. 두 평가 범위의 숫자를 직접 빼서
모델 개선량이라고 해석하지 않는다.

평가 데이터는 동일 HDF5의 고정 validation이며 반복 사용했다. untouched test
성능이나 새로운 native64 70,000장 학습 결과로 부르지 않는다. 현재 모델은
mask-supervised lineage이고 Slot Attention 비교와 동일 supervision을 보장하는
unsupervised leaderboard 주장이 아니다. FG .90 전체 평가 목표는 아직 미달성이다.

근거: [227번 결과](SW_0227_native64_joint_projection_full320/RESULT_KO.md),
[232번 결과](SW_0232_native64_exact_teacher_resume/RESULT_KO.md),
[235번 범위](SW_0235_native64_convergence_validation64/DESIGN_KO.md),
[236번 범위](SW_0236_native64_TRAIN512_exact_resume/DESIGN_KO.md).

## Bidirectional loop의 현재 범위

SW0227 최고 모델과 SW0232·235·236은 공통 native rollout을 사용한다.
이 경로는 spike pulse → Kuramoto feedback과 synchrony → graph feedback이
활성화되면 설정 검증에서 거부한다. 실제 Kuramoto 호출은 `spike=None`이다.
따라서 현재 점수를 완전한 bidirectional loop의 성과로 해석하지 않는다.

이전 spike를 dendritic 입력과 membrane reset에 쓰는 지역 recurrent 경로는
계산에 포함된다. 경로 존재와 성능 기여는 구분하며 기여는 별도 ablation으로
확인해야 한다. 최근 teacher 추가 학습은 encoder와 실제 gamma projection을
업데이트하고, 해당 추가 학습 중 나머지 core/gates 파라미터는 고정이다.

근거 실행 경로:
[native 입력 검증과 rollout](SW_0134_native_spike_binding/rollout.py),
[native forward 진입점](SW_0175_critical_component_binding/train_v2.py),
[최근 추가 학습](SW_0236_native64_TRAIN512_exact_resume/train.py).
