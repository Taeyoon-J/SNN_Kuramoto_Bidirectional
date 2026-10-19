"""Stream the complete registered TRAIN split; never read reserved evaluation masks."""
import hashlib
import numpy as np

BATCH = 16
EPOCHS = 10
TRAIN_COUNT = 70000
STEPS_PER_EPOCH = TRAIN_COUNT // BATCH
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH


def train_ids():
    ids = np.concatenate((np.arange(1000), np.arange(1640, 70640))).astype(np.int64)
    if len(ids) != TRAIN_COUNT or len(np.unique(ids)) != TRAIN_COUNT:
        raise ValueError('exactly 70000 unique TRAIN images required')
    return ids


def validate_ids(ids):
    ids = np.asarray(ids)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu':
        raise ValueError('nonempty integer TRAIN IDs required')
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= 70640) or np.any((ids >= 1000) & (ids < 1640)):
        raise ValueError('duplicate, out-of-range or reserved evaluation ID')
    return ids.astype(np.int64)


def epoch_ids(seed, epoch):
    if type(seed) is not int or seed not in (0, 1, 2) or type(epoch) is not int or not 0 <= epoch < EPOCHS:
        raise ValueError('registered seed and complete pass required')
    return train_ids()[np.random.RandomState(239000 + seed + 10000 * epoch).permutation(TRAIN_COUNT)]


def batch_ids(seed, step):
    if type(step) is not int or not 0 <= step < TOTAL_STEPS:
        raise ValueError('registered new-stage update required')
    epoch, offset = divmod(step, STEPS_PER_EPOCH)
    return epoch_ids(seed, epoch)[offset * BATCH:(offset + 1) * BATCH]


def canonical_rgb(values):
    rgb = np.asarray(values)
    if rgb.ndim != 4 or rgb.shape[1:] != (128, 128, 3):
        raise ValueError('canonical128 RGB geometry required')
    if rgb.dtype != np.uint8:
        if rgb.dtype.kind != 'f' or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('RGB must be uint8 or normalized finite float')
        rgb = np.rint(rgb.astype(np.float32) * 255).astype(np.uint8)
    return rgb


def modal_labels(pixels):
    pixels = np.asarray(pixels)
    if pixels.ndim == 4 and pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    if pixels.ndim != 3 or pixels.shape[1:] != (128, 128) or pixels.dtype.kind not in 'iu' or pixels.min() < 0:
        raise ValueError('canonical128 nonnegative integer TRAIN masks required')
    blocks = pixels.reshape(len(pixels), 64, 2, 64, 2).transpose(0, 1, 3, 2, 4).reshape(len(pixels), 64, 64, 4)
    ordered = np.sort(blocks, axis=-1)
    # Sorted candidates and first argmax implement the established lowest-ID tie rule.
    counts = (ordered[..., :, None] == ordered[..., None, :]).sum(-1)
    return np.take_along_axis(ordered, counts.argmax(-1)[..., None], axis=-1)[..., 0].astype(np.int64)


def read_batch(stream, ids):
    ids = validate_ids(ids)
    order = np.argsort(ids); inverse = np.argsort(order)
    # h5py fancy indexing requires strictly increasing IDs.
    rgb = canonical_rgb(np.asarray(stream['image'][ids[order]]))[inverse]
    labels = modal_labels(np.asarray(stream['mask'][ids[order]]))[inverse]
    if rgb.shape[0] != len(ids) or labels.shape != (len(ids), 64, 64):
        raise ValueError('TRAIN batch geometry mismatch')
    return rgb, labels


def split_binding():
    ids = train_ids()
    return dict(TRAIN_count=TRAIN_COUNT, TRAIN_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        TRAIN_ranges=[[0, 999], [1640, 70639]], reserved_range=[1000, 1639],
        evaluation_ids=[1320, 1639], batch=BATCH, complete_passes=EPOCHS,
        additional_updates=TOTAL_STEPS, logical_exposures=TRAIN_COUNT * EPOCHS)
