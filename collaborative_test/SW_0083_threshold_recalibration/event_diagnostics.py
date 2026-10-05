"""Validate/reshape hard membrane-threshold crossings captured by a hook."""
import numpy as np


def reshape_binary_crossings(captured_steps, batch, components, regions, steps):
    if len(captured_steps) != steps:
        raise ValueError(f"expected {steps} membrane hook calls, got {len(captured_steps)}")
    arrays = [np.asarray(value) for value in captured_steps]
    expected = (batch, components * regions)
    if any(array.shape != expected for array in arrays):
        raise ValueError(f"each captured crossing must have shape {expected}")
    stacked = np.stack(arrays, axis=-1).reshape(batch, components, regions, steps)
    if stacked.dtype != np.bool_ and not np.isin(stacked, (0, 1)).all():
        raise ValueError("membrane crossing diagnostic must contain only binary values")
    return stacked.astype(np.bool_, copy=False)
