"""Complete split coverage, mask tie semantics, and real HDF5 shuffled reads."""
import h5py
import numpy as np
from collaborative_test.SW_0239_native64_best_70000 import data


def test_complete_registered_split_and_passes():
    ids = data.train_ids()
    assert len(ids) == len(set(ids.tolist())) == 70000
    assert not np.any((ids >= 1000) & (ids < 1640))
    for seed in (0, 1, 2):
        for epoch in range(data.EPOCHS):
            sequence = data.epoch_ids(seed, epoch)
            assert np.array_equal(np.sort(sequence), ids)
            assert np.array_equal(data.batch_ids(seed, epoch * data.STEPS_PER_EPOCH), sequence[:16])
            assert np.array_equal(data.batch_ids(seed, (epoch + 1) * data.STEPS_PER_EPOCH - 1), sequence[-16:])
    assert data.TOTAL_STEPS == 43750


def test_reserved_ids_rejected_before_dataset_access():
    for ids in ([1000], [1320], [1639], [-1], [70640], [0, 0], [1.5]):
        try:
            data.read_batch({}, ids)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid TRAIN batch accepted')


def test_patch_majority_and_lowest_id_ties():
    rng = np.random.RandomState(239)
    pixels = rng.randint(0, 11, (2, 128, 128)).astype(np.uint8)
    pixels[0, :2, :2] = [[3, 7], [7, 3]]
    actual = data.modal_labels(pixels)
    expected = np.empty_like(actual)
    for b in range(2):
        for y in range(64):
            for x in range(64):
                values, counts = np.unique(pixels[b, 2*y:2*y+2, 2*x:2*x+2], return_counts=True)
                expected[b, y, x] = values[counts.argmax()]
    assert np.array_equal(actual, expected) and actual[0, 0, 0] == 3


def test_real_hdf5_unsorted_read_preserves_image_mask_identity():
    with h5py.File('sw0239_fixture.h5', 'w', driver='core', backing_store=False) as stream:
        images = stream.create_dataset('image', (20, 128, 128, 3), dtype='u1')
        masks = stream.create_dataset('mask', (20, 128, 128, 1), dtype='u1')
        for i in range(20):
            images[i] = i; masks[i] = i
        ids = np.array([19, 0, 8, 2])
        rgb, labels = data.read_batch(stream, ids)
        assert np.array_equal(rgb[:, 0, 0, 0], ids)
        assert np.array_equal(labels[:, 0, 0], ids)


def test_normalized_float_rgb_and_invalid_ranges():
    values = np.full((1, 128, 128, 3), .5, np.float32)
    assert (data.canonical_rgb(values) == 128).all()
    values[0, 0, 0, 0] = np.nan
    try:
        data.canonical_rgb(values)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite RGB accepted')
