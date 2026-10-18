# SW0221

Prospective six-arm mask-supervised encoder-frequency pretraining experiment.
See DESIGN_KO.md for the fixed protocol. No result is claimed before execution.

Actual execution2026-10-18:12 local/12 Linux tests,3 local CLI checks and all6 real source/TRAIN CPU admissions passed. Six exclusive idle GPU queues registered. START_EVIDENCE.json records actual NVIDIA-bound workers. All six final masks freeze before DEV GT scoring. No new performance claim; native64 full320 incumbent remains .848615/.592533/.672047.

A separate read-only diagnose_frequency.py waits for all six scored arms, revalidates their actual models and masks, then compares within-object frequency variance, adjacent-class separation and global variance with final spike mask scores. Independent centered NumPy statistics are checked against the actual Torch training objective. It adds no model forward or training and leaves the original six-arm fingerprints unchanged. Coherent and collapsed frequency fixtures passed locally and on Linux.
