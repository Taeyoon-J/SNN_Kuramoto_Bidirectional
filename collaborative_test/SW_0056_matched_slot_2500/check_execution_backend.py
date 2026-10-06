"""Process-isolated check for the CPU-default/GPU-opt-in import contract."""
import os
import sys

import train_seed


expected_backend, expected_visible = sys.argv[1:3]
assert train_seed.EXECUTION_BACKEND == expected_backend
assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_visible
print(f"{expected_backend}:{expected_visible}:ok")
