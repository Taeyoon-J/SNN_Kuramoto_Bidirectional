"""Verify byte-identical RGB access while diagnosing HDF5 fancy-index overhead."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

DATA = "/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5"
OUT = Path("/Data0/kevinswk/patch_v2_sw/trained_models/SW0094_aligned_joint_pilot/rgb_io_benchmark.json")


def main():
    torch.set_num_threads(2)
    rows = torch.randperm(70000, generator=torch.Generator().manual_seed(17))[:16].numpy()
    ids = np.where(rows < 1000, rows, rows + 640)
    order = np.argsort(ids)
    with h5py.File(DATA, "r") as dataset:
        images = dataset["image"]
        start = time.perf_counter()
        fancy = images[ids[order].tolist()][np.argsort(order)]
        fancy_seconds = time.perf_counter() - start
        start = time.perf_counter()
        direct = np.stack([images[int(image_id)] for image_id in ids])
        direct_seconds = time.perf_counter() - start
        assert np.array_equal(fancy, direct)
        record = {"training_image_ids": ids.tolist(), "shape": list(fancy.shape),
                  "dtype": str(fancy.dtype), "byte_identical": True,
                  "fancy_seconds": fancy_seconds, "direct_seconds": direct_seconds,
                  "ratio": fancy_seconds / direct_seconds,
                  "caveat": "one batch; sequential reads can benefit from cache warming"}
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
