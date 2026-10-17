"""Paired changes within cohorts; no overfit claim from raw train/dev gap."""
import numpy as np

TRAIN_INDICES = tuple(range(0, 64, 4))
METRICS = ('fg_ari', 'foreground_iou', 'matched_object_iou')


def paired_changes(train_before, train_after, dev_before, dev_after):
    arrays = [np.asarray(v, dtype=np.float64) for v in
              (train_before, train_after, dev_before, dev_after)]
    for value, expected in zip(arrays, (16, 16, 32, 32)):
        if value.shape != (expected, 3) or not np.isfinite(value).all():
            raise ValueError('fixed finite TRAIN16 / development32 vectors required')
    tb, ta, db, da = arrays
    rng = np.random.RandomState(195)
    train_delta, dev_delta = ta - tb, da - db
    train_boot = train_delta[rng.randint(0, 16, size=(10000, 16))].mean(1)
    dev_boot = dev_delta[rng.randint(0, 32, size=(10000, 32))].mean(1)
    result = {}
    for name, delta, samples in (
        ('TRAIN', train_delta, train_boot),
        ('development', dev_delta, dev_boot),
        ('TRAIN_minus_development_change', train_delta.mean(0) - dev_delta.mean(0),
         train_boot - dev_boot),
    ):
        mean = delta.mean(0) if delta.ndim == 2 else delta
        ci = np.percentile(samples, [2.5, 97.5], axis=0)
        result[name] = {metric: {'mean': float(mean[i]), 'ci95': ci[:, i].tolist()}
                        for i, metric in enumerate(METRICS)}
    result['overfit_consistent_FG_signal'] = (
        result['TRAIN']['fg_ari']['ci95'][0] > 0
        and result['development']['fg_ari']['ci95'][1] < 0
    )
    result['causal_overfitting_proved'] = False
    result['scope'] = 'seed0, TRAIN16/development32, TRAIN8-to-TRAIN64 adaptation only'
    return result
