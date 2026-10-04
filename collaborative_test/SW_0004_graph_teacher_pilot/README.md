# SW_0004 — graph-to-membrane synchrony pilot

Status: 10-epoch pilot and fixed-split validation complete. Gradient path verified. No all-metric improvement.

## 쉽게 설명하면

모델 안에는 이미지마다 patch 사이의 연결 강도를 만드는 그래프가 있습니다.
진단해 보니 같은 물체의 patch 사이 연결이 다른 물체 사이보다 약 12배
강했습니다. 이번 후보는 **그 그래프를 정답 대신 힌트로 삼아 membrane의
시간 패턴도 비슷하게 만들도록 학습**하는 것입니다. 원래의 위상 loss는
그대로 유지하고 새 loss만 약하게 더합니다. 정답 CLEVR mask는 학습에
쓰지 않습니다. 같은 검증 이미지 320장으로 비교한 결과, 물체를 서로
구별하는 점수는 올랐지만 물체가 차지한 위치를 맞추는 두 점수는
내렸습니다. 따라서 이 설정을 최종 모델로 채택하지 않습니다.

| 같은 검증 이미지 320장, spike 분류 | FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ | spike 비율 |
|---|---:|---:|---:|---:|
| 기존 SW_0001, 40 epoch | 0.1363 | 0.2720 | 0.1456 | 0.4849 |
| 새 graph 힌트, 10 epoch | 0.2029 | 0.2459 | 0.1421 | 0.4257 |

**해석:** graph 힌트가 membrane 경로를 학습시키는 것은 확인됐지만,
세 평가 점수를 동시에 올리지는 못했습니다. 학습 길이도 40 대 10
epoch로 다르므로 loss 자체의 효과를 단정하지 않습니다.

## Motivation

The phase-only baseline leaves all dendritic and membrane parameters without
gradients. A diagnostic on validation IDs 1320–1351 found the learned graph's
mean edge weight to be 0.254288 for pairs in the same true object, versus
0.021720 for pairs in different foreground objects. Ground truth was used only
to inspect the graph, not to train it. Direct spectral clustering of the graph
gave FG-ARI 0.305600 / foreground IoU 0.227293 / object IoU 0.160476 on
these 32 images, but this **graph-only readout is not a valid goal result**.

## Change

Retain the baseline phase PLV loss, core computation, and 40-epoch baseline
settings. Add an optional loss that aligns *centered membrane-trace synchrony*
with the per-image learned graph. The graph is detached as a teacher; its
off-diagonal row-normalized edge weights form a distribution and each
oscillator's membrane synchrony forms a softmax distribution. Their KL
divergence is weighted by 0.1. This reaches the dendritic and membrane path
while the original phase PLV objective still trains theta. No CLEVR masks are
used during training. The pilot is 10 epochs and therefore not a matched-epoch
causal comparison to the 40-epoch baseline.

Source change: standalone `graph_teacher_synchrony_loss` in
`snn_kuramoto_bidirectional/loss_function.py`; optional
`--graph-teacher-weight` and `--graph-teacher-temperature` in
`training/train_s2net_core.py`, both leaving original behavior unchanged when
the weight is zero. `test_graph_teacher_loss.py` confirms finite nonzero
membrane gradients and no gradient into the detached teacher.

Exact server command is in `run.sh`. Evaluate both spike and original
spatial-components readouts on validation IDs 1320–1639 using the same
`evaluate_fixed_split.py` contract. Do not select on the reference test split.

Validation metrics and per-image scores: `results.json`. The original
spatial-components readout remained 0.0000 / 0.2168 / 0.0148.
`gradient_connectivity.json` shows nonzero gradients in all three dendritic
parameters and the membrane parameter under the new objective (previously
none were connected under the phase-only loss). This verifies signal flow,
not segmentation quality. The 10-epoch pilot should not be interpreted as
a matched-duration comparison to the 40-epoch baseline.
