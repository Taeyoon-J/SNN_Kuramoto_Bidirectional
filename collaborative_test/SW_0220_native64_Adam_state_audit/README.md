# SW0220 native64 optimizer-state diagnosis

One bounded TRAIN4 comparison after218's first uphill update: keep inherited
Adam state versus reset its moments/counter, with the same model, clipped
gradient, frozen ranking context, learning rates and projection. Measure actual
same-RNG full-forward loss, then restore source model/Adam/RNG exactly.
Two temporary proposals, zero committed updates, no DEV scoring or new70k run.
See DESIGN_KO.md. This diagnosis is not a new segmentation performance result.

Actual start2026-10-18:8 local/8 Linux tests,2 local/2 Linux CLI checks and real source model/Adam/RNG/frozen-control CPU admission passed. Supervisor2394953, worker2397214 on exclusive GPU1. Source and actual NVIDIA/process bindings are archived. Peer patch_v2 rechecked at7a21632, unchanged. No new segmentation score or source update is claimed.

Completed: before2.819465, inherited-after2.951871, reset-after2.919585. Both proposals are uphill in actual forward loss; TRAIN-only follow-up gate failed. Same gradient/hyperparameters and exact source model/Adam/RNG restoration were verified. No reset-Adam training expansion and no new segmentation score. See RESULT_KO.md.
