# SW0214: component 보정 classifier의 primary 개선 실패

Native64·고정16·seed2, 기존211 cache/TRAIN32를 사용했다. 새 core forward와
core/encoder update는0이다. Local8/Linux8 테스트, 실제6648 TRAIN pair의
finite gradient와 zero-edge readout의 정확한 baseline 재현을 사전검증했다.
GPU1에서 full-pair Adam128을 완료했고48개 mask·TRAIN normalization·Adam과
평가값을 독립 재검증했다. 기존 component 내부와 전경은 그대로 보존됐다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| primary 관계 특징+geometry | .720607 | .445474 | .580461 |
| frozen geometry-only control | .814008 | .445474 | .632410 |

Primary FG 차이는 -.067807, paired95% CI[-.139363,-.007868]이다.
Geometry control의 FG 차이+.025594는 CI[-.013878,+.085607]로0을 포함한다.
ObjectIoU 차이는 -.041925, CI[-.084533,-.002544]다. 따라서 .8140을 확정된
개선이나 새로운 전체평가 최고점으로 부르지 않는다. FG 후보 기록은 보존한다.
Control은 같은 trained head의 입력만 가린 OOD 비교이며, geometry-only로
재학습한 모델이나 물리적 gating ablation이 아니다.

TRAIN relation loss는 .693147→.144341로 감소했다. Primary는 평균9.3125개
component를 합쳐12.5개를 예측했고, control은15개를 합쳐6.8125개를 예측했다.
정답 평균은5.375개다. 단순히 component 수를 줄이는 것이 정확한 grouping을
보장하지 않는다.

후속 점검에서 positive/negative mass를 각각 normalize한 loss와 raw logit>0
판정의 의미가 다름을 확인했다.215는 기존214 판정을 보존한 채 TRAIN에서만
도출한 log-odds shift를 적용하는 별도 비교다. DEV threshold sweep은 하지
않는다. 많은 anchor 특징의 위치 의존성/일반화 문제도 가설로 남지만 frozen
control 결과만으로 원인이라고 확정하지 않는다.

기존 전체320·세seed .848615/.592533/.672047 기준점을 유지한다.
증빙: results_archive/completed_v1/score_v1/summary.json과 completion_audit.json,
deployment_v1/registration_receipt.json과 GPU 실행 ledger. 짧은 GPU 실행은
live NVIDIA 관찰 전에 완료됐으며 실제 CUDA checkpoint/Adam 기록을 검증했다.
