# SW0178: temporal class competition

완료·독립 검증된 seed1 결과입니다. **QCE standalone FG-ARI 0.8180 / FIoU 0.4375 /
Object IoU 0.6864**. FG와 Object IoU는 기준 대비 개선됐지만 foreground false
positives가 늘었습니다. 원래 사전 gate는 실패로 종료하며 기존 unsupervised
세-seed 최고 모델은 유지합니다. 후속 SW0179에서 변경 없는 QCE 후보의
source seed0/2 재현성을 확인합니다.

전체 수치·비교·원인 진단·해석 범위는 [RESULT_KO.md](RESULT_KO.md), 사전 설계는
[DESIGN_KO.md](DESIGN_KO.md)에 있습니다. 이 실험은 TRAIN modal mask를 사용하는
supervised 추가 학습(512장×2)이며 native32 70,000장 신규 학습이나 gate 인과
기여 증명으로 해석하지 않습니다.

기록: `results_archive/cpu_v1`(실제 입력/모델/복원), `gpu_v1`(첫 두 실제 updates),
`train_v1`(최종64updates와 loss 검증), `eval_v1`(평가 준비/등록),
`pred_v1`(GT-free 예측 검사/audit 예약), `score_v1`(최종 점수/독립 검산/진단).
모든 학습·예측·점수·audit 작업이 완료됐으며 재시작하지 않았습니다.
