# SW0111: partition RGB credit with frozen encoder and graph

This test sends reconstruction credit to the actual-spike assignment through a straight-through production connected-component partition. It freezes the encoder and all legacy graph parameters, addressing a different mechanism from SW106 joint adaptation. Detached native features supply pooled decoder content; actual spikes alone supply assignment. The decoder never predicts evaluation masks.

`protocol.json` fixes the equations, fresh SW0097-derived shared decoder warmup, train-only shared lambda calibration, separate optimizer gradients and all-three-seed matched256/full320 endpoints. SW106 source95 warmup and lambda artifacts may not be reused. Implementation review and real preflight must pass before training.

`run.py` contains the checked preflight, paired arm training, prediction-before-mask evaluation and exact SW0097 B8 per-image reference guard. `coordinator.py` exposes the seed0-calibration dependency and the paired tasks; `summarize.py` computes the registered three-seed comparisons and shared-image bootstrap. The local runner currently has CPU contract tests only; no SW0111 cache, GPU preflight, training or endpoint has run.

Report both the new matched control and original SW0097 reference. No endpoint tuning, reserve access or replacement mask classifier is permitted. Technical failure stops this branch; unrelated queued experiments continue.
