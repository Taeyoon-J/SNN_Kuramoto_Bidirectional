# SW0230 완료: 상대 주파수 간격 유지도 FG 개선에 실패

228번의 절대 중심 유지 대신, 223번의 원래 teacher denoising/gamma anchor를
유지하면서 TRAIN 물체별 중심 간 거리가 source보다 줄어들 때만 상대 패널티를
추가했다. 같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·실제
encoder+4x8 projection 공동학습을 사용했다. graph/나머지 core/gates/readout은
고정했다. mask supervision을 사용하는 native64 고정16장·세 seed 비교다.

| 조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 모델 | .842891 | .600692 | .702161 |
| 원래 공동학습(223번) | **.859634** | **.645611** | .706340 |
| 상대 간격 패널티 추가(230번) | .852402 | .624603 | **.711310** |

230−223의 FG 차이는 −.007232, paired95% CI[−.014287,−.001225]로 감소했다.
FIoU 차이는 −.021008, CI[−.064246,.011209], Object IoU 차이는 +.004969,
CI[−.007752,.018735]다. IoU 변화의 불확실성을 유지하며 FG 감소를 분명히 기록한다.
원래 모델 대비 FG +.009511도 CI[−.007517,.034865]가 0을 포함한다.
사전 등록 FG/balanced 후보 기준 모두 실패했다. 이 설정의 전체320장/70,000장
확대 및 계수·학습률·epoch sweep은 진행하지 않는다.

주파수 진단상 인접 class 중심 간 평균 제곱 거리는 세 seed에서
.087482/.156513/.050628로 원래 source의 .087986/.162614/.053982에 가깝게
유지됐다. 그러나 물체 내부 분산은 .003571/.015140/.006245로, 223번의
.003360/.013505/.005320보다 모두 컸다. 중심 거리 수축을 덜어도 내부 정렬과
최종 FG 개선을 동시에 얻지는 못했다. 228/230번 결과는 class 평균의 이동이나
수축만을 억제하는 접근을 우선순위에서 내릴 근거이며, source 주파수 간격이
실제 spike assignment에 최적이라는 가설을 지지하지 않는다. 개별 gating의
인과성이나 다른 구조에서의 가능성을 부정하는 결과는 아니다.

세 실제 encoder/projection/Adam checkpoint의128회 업데이트, 나머지 core의
불변성, source teacher/대조군/모든 mask protocol과 loss 분해를 확인했다.
모든 protocol 확인 후 DEV GT를 읽고144개 mask를 독립 재채점했다.
독립 metric/평균/CI 오차는 1e-12 미만, 로컬 재계산 오차는0이었다.
추가 학습이나 GPU forward는 없었다. 고정 모델에 조건부인 이미지 bootstrap이며
seed 모집단 CI나 untouched test 결과가 아니다.

전체320장 최고 기록 .848615/.592533/.672047은 유지한다. 223/225번 고정 후보의
전체320장 확인은 별도227번에서 계속 진행한다.

근거: results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}.
원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.
