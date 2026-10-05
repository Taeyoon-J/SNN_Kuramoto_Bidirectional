"""Guard target loading until all fixed GT-free readouts have been produced."""

REQUIRED_READOUTS = {"spike_cc_threshold_0p50", "membrane_spatial_sigma1p5_k10"}


def require_complete_predictions(predictions, expected_count):
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    if set(predictions) != REQUIRED_READOUTS:
        raise ValueError("Both registered prediction readouts must exist before loading targets")
    for name in REQUIRED_READOUTS:
        chunks = predictions[name]
        if not chunks:
            raise ValueError("Both prediction readouts must contain outputs before loading targets")
        try:
            count = sum(int(chunk.shape[0]) for chunk in chunks)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Prediction chunks for {name} must have a batch dimension") from exc
        if count != expected_count:
            raise ValueError(f"Readout {name} has {count} predictions; expected {expected_count}")
