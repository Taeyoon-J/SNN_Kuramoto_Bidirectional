# SW0236: 같은512회 학습에서 데이터 증가의 FG 효과는 아직 불확실

고정16장·native64·seed0/1/2 평균이다. 기존 pretrained 모델에서 추가 학습한
고유 TRAIN 이미지 수와 누적 업데이트 횟수를 구분한다. 처음부터64장/512장만으로
학습한 모델도, 신규70,000장 학습 결과도 아니다.

| 같은512회·batch16·8,192 exposures | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 기준 | .842891 | .600692 | .702161 |
| TRAIN64 / SW0232 | .884622 | **.677086** | .712312 |
| TRAIN512 / SW0236 | **.885655** | .657488 | **.724440** |

TRAIN512−TRAIN64의 FG 차이는 +.001034, paired image 95% CI
[-.017698,.021437]이다. 데이터 증가로 FG가 좋아졌다고 확정할 수 없다.
Foreground IoU 차이는 −.019598, CI[-.048114,.003658]로 평균은 낮지만
감소 자체도 확정하지 않는다. Object IoU는 +.012128,
CI[.000026,.025617]로 이 고정 cohort에서는 개선을 지지한다. 이는 고정된 세
모델에 조건부인 이미지 bootstrap이며 seed 모집단 CI가 아니다.

원래 기준 대비 FG는 +.042765, CI[.009233,.089741]로 등록된 FG 후보 gate는
통과했다. 그러나 TRAIN64 대비 Foreground IoU 평균이 낮아 balanced gate는
실패했다. 데이터 증가 효과 flag도 false다. 이를 세 지표 모두 개선된 새 최고
모델이나 전체320장 FG≥.90 달성으로 승격하지 않는다. 바로70,000장 학습이나
같은 방향의 계수·epoch sweep을 추가할 근거로 사용하지 않는다.

실제 SW0226 accepted128 checkpoint의 encoder·실제 gamma projection·Adam·history·
Torch RNG를 bitwise 복원하고 새384회를 더해 누적512회로 학습했다. 세 seed의
실제 Adam counter는512이며 나머지 core/graph/gates/readout은 유지했다.
모든 prediction protocol을 확인한 뒤 DEV GT를 읽고 실제144개 mask를 독립
재채점했다. Teacher와 TRAIN 순서, checkpoint/source SHA, 평균과 CI도 검증했다.
로컬 통계 재계산 오차는0이다. 원본 데이터·checkpoint·이미지별 벡터는 공개하지 않는다.

세 seed 모두 TRAIN teacher loss와 DEV 물체 내부 frequency variance는 원래
기준보다 감소했다. 그러나 최종 within-object variance는 TRAIN64보다 조금씩
높았다(.002838/.008891/.003428 대 .002771/.008172/.003398). Teacher loss 감소가
데이터 증가에 따른 FG 향상을 보장하지 않았으며, 그 원인이나 모든 gating의
기여를 이 통계만으로 단정하지 않는다.

전체320장 최고는 SW0227 .858916/.605286/.674313으로 유지한다. SW0235의
고정 TRAIN64/512회 모델 확장64장 확인은 별개로 계속 진행된다.

근거: `results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}`.
