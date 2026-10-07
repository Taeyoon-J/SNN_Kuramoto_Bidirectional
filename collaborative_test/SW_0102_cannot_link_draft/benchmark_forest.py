"""Synthetic CPU runtime benchmark; opens no model/data/GT files."""

import json
import statistics
import time

import torch

from .forest_loss import maximin_forest_edge_map


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(102)
    raw = torch.rand((16, 256, 256), generator=generator)
    q = (raw + raw.transpose(1, 2)) * .5
    q.diagonal(dim1=1, dim2=2).fill_(1.)
    durations = []
    edge_map = None
    for _ in range(5):
        start = time.perf_counter()
        edge_map = maximin_forest_edge_map(q)
        durations.append(time.perf_counter() - start)
    result = {"status": "complete", "data": "synthetic only; no source/data/GT opened",
              "batch": 16, "nodes": 256, "repeats": len(durations),
              "edge_threshold_strict": .40,
              "median_seconds": statistics.median(durations),
              "all_seconds": durations,
              "mapped_fraction": float((edge_map >= 0).float().mean()),
              "torch_version": torch.__version__}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
