"""Registered 64-proposal mapping from SW0179's two permutations to TRAIN1024."""
from __future__ import annotations

from collaborative_test.SW_0179_qce_seed_replication import train as base

COUNT = 1024
PREFIX = 512


def expected_batches_1024():
    """Keep pass one identical; map pass two onto the unique appended half."""
    old = base._expected_batches()
    if len(old) != 64 or any(len(batch) != 16 for batch in old):
        raise ValueError("frozen SW0179 schedule is not the registered 64xB16 order")
    result = [list(map(int, batch)) for batch in old[:32]]
    result.extend([[int(index) + PREFIX for index in batch] for batch in old[32:]])
    if len(result) != 64 or any(index < 0 or index >= COUNT for row in result for index in row):
        raise ValueError("SW0182 mapped schedule has invalid dimensions or indices")
    flattened = [index for row in result for index in row]
    if len(flattened) != 1024 or len(set(flattened)) != 1024:
        raise ValueError("SW0182 schedule must expose each TRAIN1024 image exactly once")
    if result[:32] != old[:32]:
        raise AssertionError("SW0182 first32 proposals differ from frozen SW0181 control")
    return result


def validate_ordered_ids(ids):
    values = list(map(int, ids))
    if len(values) != COUNT or len(set(values)) != COUNT:
        raise ValueError("SW0182 requires exactly 1024 unique ordered TRAIN IDs")
    for value in values:
        if 1320 <= value <= 1639 or 90640 <= value <= 90959:
            raise ValueError("SW0182 TRAIN order includes validation/reserve IDs")
    return values
