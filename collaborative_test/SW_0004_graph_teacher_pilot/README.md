# SW_0004 — graph-to-membrane synchrony pilot

Status: code smoke-tested; 10-epoch training running on server GPU 1.

## 쉽게 설명하면

모델 안에는 이미지마다 patch 사이의 연결 강도를 만드는 그래프가 있습니다.
진단해 보니 같은 물체의 patch 사이 연결이 다른 물체 사이보다 약 12배
강했습니다. 이번 후보는 **그 그래프를 정답 대신 힌트로 삼아 membrane의
시간 패턴도 비슷하게 만들도록 학습**하는 것입니다. 원래의 위상 loss는
그대로 유지하고 새 loss만 약하게 더합니다. 정답 CLEVR mask는 학습에
쓰지 않습니다. 아직 점수는 없으며, 결과가 나오면 기존 모델과 같은
검증 이미지로 비교합니다.

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
