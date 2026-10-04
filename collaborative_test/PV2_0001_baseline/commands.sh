#!/bin/bash
# PV2_0001. Checkpoints already existed from the spatial-decay sweep; training
# shown for completeness, evaluation is what this test ran.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
cd /export_home/tkim1/snn_kuramoto_bidirectional
for SEED in 0 1 2; do
  SEED=$SEED GPU=$SEED RUN=/work/USERS/tkim1/runs/PV2_0001_s$SEED \
    bash configs/best_16x16.sh
done
cd /export_home/tkim1/collaborative_test
export PYTHONPATH=/export_home/tkim1/snn_kuramoto_bidirectional:.
for spec in "0:DEC055" "1:D055_s1" "2:D055_s2"; do
  s=${spec%%:*}; r=${spec#*:}
  CUDA_VISIBLE_DEVICES=1 $PY evaluate_model.py \
    --checkpoint /work/USERS/tkim1/runs/$r/core.pt \
    --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
    --targets /work/USERS/tkim1/clevr/with_masks/targets_v1.pt \
    --manifest /work/USERS/tkim1/clevr/with_masks/split_manifest.json \
    --split test --limit 300 --synchrony-threshold 0.40 --min-group-size 2 \
    --json-out /work/USERS/tkim1/runs/PV2_0001/seed$s.json
done
