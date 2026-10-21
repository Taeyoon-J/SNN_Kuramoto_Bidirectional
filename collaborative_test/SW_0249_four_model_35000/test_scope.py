import numpy as np
import pytest
from . import scope as s

def test_same_unique_subset_and_no_eval_overlap():
    ids=s.train_ids()
    assert ids.shape==(35000,) and np.unique(ids).size==35000
    assert not np.any((ids>=1000)&(ids<1640))
    assert np.array_equal(ids,s.train_ids())

def test_exact_epoch_coverage_including_short_final_batch():
    for seed in s.SEEDS:
        for epoch in (0,9):
            rows=[s.batch_ids(seed,epoch*s.BATCHES+i) for i in range(s.BATCHES)]
            assert len(rows[-1])==8 and sum(map(len,rows))==35000
            assert np.array_equal(np.sort(np.concatenate(rows)),s.train_ids())
    assert s.UPDATES==21880 and s.COUNT*s.EPOCHS==350000

def test_shuffle_changes_by_seed_and_epoch_not_model():
    assert not np.array_equal(s.epoch_ids(0,0),s.epoch_ids(1,0))
    assert not np.array_equal(s.epoch_ids(0,0),s.epoch_ids(0,1))
    assert len(s.EVAL_IDS)==320 and s.EVAL_IDS[0]==1320
    assert len([(m,v) for m in s.MODELS for v in s.SEEDS])==12
    with pytest.raises(ValueError):s.batch_ids(0,s.UPDATES)
