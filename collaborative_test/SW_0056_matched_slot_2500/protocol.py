"""Exact exposure schedule for the SW0056 Slot Attention candidate."""
import re

import numpy as np


TRAIN_ID_SEGMENTS = ((0, 999), (1640, 3139))
VALIDATION_IDS = (1320, 1639)


def stable_weight_signature(weights):
    """Ordered names/shapes with Keras-generated numeric layer suffixes removed."""
    signature = []
    for variable in weights:
        name = str(getattr(variable, "path", getattr(variable, "name", ""))).split(":", 1)[0]
        parts = name.split("/")[-2:]
        normalized = tuple(re.sub(r"_\d+$", "", part) for part in parts)
        shape = tuple(int(d) for d in variable.shape)
        signature.append((normalized, shape))
    return tuple(signature)


def training_ids():
    return np.concatenate((np.arange(0, 1000), np.arange(1640, 3140))).astype(np.int64)


def exact_exposure_batches(ids=None, repeats=10, batch_size=16, seed=0):
    """Shuffle one full permutation per pass, then split without padding/drop."""
    ids = training_ids() if ids is None else np.asarray(ids, dtype=np.int64)
    repeats, batch_size, seed = int(repeats), int(batch_size), int(seed)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be a nonempty vector of unique global IDs")
    if repeats <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("repeats/batch_size must be positive and seed nonnegative")
    rng = np.random.RandomState(seed)
    stream = np.concatenate([ids[rng.permutation(len(ids))] for _ in range(repeats)])
    return [stream[start:start + batch_size] for start in range(0, len(stream), batch_size)]


def official_learning_rate(step, base_learning_rate=4e-4, warmup_steps=156,
                           decay_rate=0.5, decay_steps=1563):
    step = int(step)
    if step < 0 or warmup_steps <= 0 or decay_steps <= 0:
        raise ValueError("step must be non-negative and schedule steps positive")
    if base_learning_rate <= 0 or not 0 < decay_rate <= 1:
        raise ValueError("invalid learning-rate parameters")
    return float(base_learning_rate) * min(step / warmup_steps, 1.0) * decay_rate ** (step / decay_steps)


def training_protocol(seed=0):
    batches = exact_exposure_batches(seed=seed)
    unique, counts = np.unique(np.concatenate(batches), return_counts=True)
    expected = training_ids()
    if not np.array_equal(unique, expected) or not np.all(counts == 10):
        raise AssertionError("full protocol must present every train ID exactly ten times")
    return {
        "seed": int(seed),
        "training_id_segments_inclusive": [list(x) for x in TRAIN_ID_SEGMENTS],
        "validation_ids_inclusive": list(VALIDATION_IDS),
        "unique_training_images": int(len(expected)),
        "exposures_per_image": 10,
        "effective_image_exposures": int(len(expected) * 10),
        "batch_size": 16,
        "optimizer_updates": len(batches),
        "full_batch_updates": len(batches) - 1,
        "final_batch_size": int(len(batches[-1])),
        "warmup_steps": 156,
        "decay_steps": 1563,
        "decay_rate": 0.5,
        "learning_rate": 4e-4,
        "adam_epsilon": 1e-8,
        "resolution": [128, 128],
        "num_slots": 11,
        "iterations": 3,
        "loss": "mean squared reconstruction error",
        "preprocessing": "full HDF5 RGB, float32 /127.5 - 1; no crop",
        "batch_policy": "ten independently seeded permutations concatenated; 1562 batches of 16 plus one 8-example static-signature batch; no padding and no dropped exposure; final batch gradients update the primary 16-signature variables with shared optimizer state",
        "protocol_type": "matched-data unique-scene 10-pass budget candidate, not official 500k-step reproduction",
    }
