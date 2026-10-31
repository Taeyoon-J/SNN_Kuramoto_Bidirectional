# SW0255 actual scientific results

기록일:2026-10-30. **완료된 단일 seed 결과이며 세 seed 평균이 아닙니다.**
현재 예산은 고유 TRAIN3,500장×10epochs=35,000exposures/seed입니다.
다른 결과가 아직 없는 것은 미완료/미검증 상태이며0점이라는 뜻이 아닙니다.

| Model | Dataset | Seed | Native model grid | Evaluation | FG-ARI | Foreground IoU | Object IoU |
|---|---|---:|---|---|---:|---:|---:|
|253 unsupervised|Our CLEVR HDF5|0|32×32|Fixed320, native128pixels|0.428118|0.277787|0.240651|

동일 예측을 기존 fixed64 patch 기준으로 평가하면
**0.457315 / 0.273522 / 0.248168**입니다. 이것은 모델을64×64로 학습했다는 뜻이 아닙니다.
이 모델은 무작위 초기화부터 학습했고 TRAIN mask를 학습에 사용하지 않았습니다.

[전체 검증 기록](model253_clevr_seed0.json)에서2190Adam steps,35000exposures,
10회 전체 데이터 순서, checkpoint SHA, 모든 고정 예측 SHA,
보호 evaluator 독립 재계산을 확인할 수 있습니다. 세 지표 모두320장 valid입니다.
평가는 GT를 보지 않고 예측을 먼저 고정한 뒤 수행했습니다.

완료 checkpoint (Git에는 포함하지 않음):
`/work/USERS/kevinswk/experiments/SW0255_full3500/jobs/model253/clevr/seed0/resume.pt`

SHA256:`539e35b31ce5cb410ed3e0a0cf6293701d578976707643686e04f89701a1ca8e`

Frontier에서 이 checkpoint를 재평가하려면 이 폴더의 상위benchmark_codes에서:

```bash
CUDA_VISIBLE_DEVICES=0 python -B run.py --config configs/frontier.json --output /work/USERS/$USER/sw0255_seed0_reeval evaluate --model 253 --dataset clevr --seed 0 --checkpoint /work/USERS/kevinswk/experiments/SW0255_full3500/jobs/model253/clevr/seed0/resume.pt
```

다른 컴퓨터에서는 README의 데이터 가져오기 절차와 함께 해당 checkpoint를SSH로
별도 복사하고 SHA를 검증하면 됩니다. 학습 재현에는 이 checkpoint가 필요하지 않습니다.
