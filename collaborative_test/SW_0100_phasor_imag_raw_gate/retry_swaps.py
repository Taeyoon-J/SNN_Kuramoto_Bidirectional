"""Retry only SW0100's fixed 16-image zero-update diagnostics after an output-schema fix."""
from coordinator import OUT, run_swaps, verify_candidate

# Never train or re-evaluate here. Require the already completed, audited
# candidates before touching only the separate ``swaps/`` diagnostic outputs.
for seed in range(3):
    verify_candidate(seed, OUT / f"seed{seed}")
run_swaps()