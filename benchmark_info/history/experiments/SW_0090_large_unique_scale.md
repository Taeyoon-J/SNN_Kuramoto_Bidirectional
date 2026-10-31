# SW_0090_large_unique_scale

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `242ccd5ef6cff52bedf2dff521149074e8711618df81624cd9390127afb44ba6`

````markdown
# SW0090 — large unique-scene scaling and qualitative masks

This experiment keeps the strongest SW0072 architecture and readout contract
fixed, while scaling the native frozen-encoder training set from 2,500 to
70,000 unique scenes, matching the official Slot Attention CLEVR6 scene count.
IDs 1320–1639 remain untouched validation data.

The same run saves epochs 1, 3, and 10 to measure whether additional passes over
the larger set help. Three seeds are evaluated on all 320 held-out
scenes with the fixed 1024-step spike connected-component readout at threshold
0.50. Qualitative output uses three pre-registered validation IDs (1320–1322)
and exports originals, ground-truth masks, predicted masks, and mask grids.

## Final three-seed result

All three seeds completed. Mean FG-ARI / foreground IoU / matched-object IoU:
epoch1 `.774323/.548028/.603848`; epoch3 `.752242/.548378/.614339`;
epoch10 `.677605/.469147/.561948`. Relative to SW0072's 2,500-image mean,
only foreground IoU improves. The early two-seed object-IoU improvement is
not preserved after including seed2. Both unique data and optimizer updates
increase in this comparison, so it does not isolate data diversity alone.

The registered normalized-sum selection chooses epoch1, then seed0 for the
qualitative exports. Three IDs1320-1322 were fixed before prediction. All
original images, individual ground-truth/predicted masks, instance maps, and
mask grids are under `results/full320/visualizations`. Prediction masks remain
16x16 patch labels enlarged by nearest-neighbour; colors are instance IDs,
not semantic categories or matching IDs across ground truth and prediction.

## Historical seed0/1 snapshot (superseded)

Before seed2 evaluation completed, the fixed full-320 evaluation covered seeds 0
and 1. Their epoch-1 mean was `.782584/.615775/.623642`. Against the same-seed
SW0072 mean `.785606/.553558/.607371`, 70,000 unique scenes change the metrics
by `-.003023/+.062217/+.016271`: foreground and matched-object IoU improve,
while FG-ARI is essentially flat but slightly lower. Epoch 3 trades more FG-ARI
for IoU (`.764199/.651509/.624770`); epoch 10 degrades to
`.708914/.541804/.583084`. Longer training degraded these measured checkpoints.
The final three-seed rows above supersede this two-seed snapshot.


````

</details>
