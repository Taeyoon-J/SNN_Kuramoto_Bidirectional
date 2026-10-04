"""Deterministic finite-budget training helpers for SW0048."""
import math

import numpy as np


def official_learning_rate(step, base_learning_rate=4e-4, warmup_steps=250,
                           decay_rate=0.5, decay_steps=2500):
    """Official Slot Attention warmup/exponential-decay rule, rescaled to budget."""
    step = int(step)
    if step < 0 or warmup_steps <= 0 or decay_steps <= 0:
        raise ValueError("step must be non-negative; schedule steps must be positive.")
    base_learning_rate = float(base_learning_rate)
    decay_rate = float(decay_rate)
    if base_learning_rate <= 0 or not 0 < decay_rate <= 1:
        raise ValueError("Require positive base learning rate and decay_rate in (0, 1].")
    warmup_factor = min(float(step) / float(warmup_steps), 1.0)
    return base_learning_rate * warmup_factor * (decay_rate ** (float(step) / decay_steps))


def make_epoch_balanced_batches(num_examples, batch_size=16, steps=2500, seed=0):
    """Shuffle each full data pass, concatenate passes, then form exact batches.

    Batch boundaries can cross pass boundaries, so no final partial batch is
    dropped. For 1000 examples, batch 16, and 2500 steps, every example appears
    exactly 40 times across the update stream.
    """
    num_examples, batch_size, steps, seed = map(
        int, (num_examples, batch_size, steps, seed))
    if num_examples <= 0 or batch_size <= 0 or steps <= 0 or seed < 0:
        raise ValueError("num_examples, batch_size, steps must be positive and seed non-negative.")
    needed = batch_size * steps
    passes = int(math.ceil(needed / num_examples))
    rng = np.random.RandomState(seed)
    stream = np.concatenate([rng.permutation(num_examples) for _ in range(passes)])[:needed]
    return stream.reshape(steps, batch_size)


def training_protocol(seed, num_examples=1000, batch_size=16, steps=2500,
                      warmup_steps=250, decay_steps=2500):
    batches = make_epoch_balanced_batches(num_examples, batch_size, steps, seed)
    counts = np.bincount(batches.reshape(-1), minlength=num_examples)
    return {
        "seed": int(seed),
        "train_ids": [0, int(num_examples) - 1],
        "num_train_images": int(num_examples),
        "batch_size": int(batch_size),
        "num_train_steps": int(steps),
        "effective_image_passes": float(batch_size * steps / num_examples),
        "warmup_steps": int(warmup_steps),
        "decay_rate": 0.5,
        "decay_steps": int(decay_steps),
        "per_image_occurrences": {
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean()),
        },
        "batch_policy": "independent seeded permutation per full pass, concatenated before batching; batches may cross pass boundaries and no examples are dropped",
        "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
        "resolution": [128, 128],
        "num_slots": 11,
        "num_iterations": 3,
        "loss": "mean squared reconstruction error",
        "optimizer": "Adam",
        "learning_rate": 4e-4,
        "adam_epsilon": 1e-8,
    }
