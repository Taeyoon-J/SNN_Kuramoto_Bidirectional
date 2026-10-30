import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mod_benchmark import common as c
from mod_benchmark.worker import mbo,digest_json

def test_protected_snapshot_and_syntax():
    assert c.immutable()
    for p in (ROOT/'mod_benchmark').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))

def test_exact_train_coverage_and_no_test_ids():
    for seed in range(3):
        for epoch in range(10):
            ids=c.epoch_order(seed,epoch)
            assert sorted(ids)==list(range(6000))
            assert len(ids.reshape(375,16))==375
    assert c.UPDATES==3750 and c.TEST_IDS==list(range(7000,8000))

def test_native64_preservation_and_tetrominoes_modal_adapter():
    source=np.zeros((64,64),np.int64);source[:2,:2]=[[3,2],[2,3]]
    before=source.copy()
    result=c.prediction_grid(source,'tetrominoes')
    assert result.shape==(32,32) and result[0,0]==2
    assert np.array_equal(before,source)
    assert np.array_equal(c.prediction_grid(source,'multi_dsprites'),source)

def test_actual_protected_pixel_evaluator_and_best_overlap():
    metric=c.benchmark_module('evaluation');cells=c.benchmark_module('make_targets_grid')
    truth=torch.tensor([[[0,1],[0,2]]]);pred=truth.clone()
    result=metric.evaluate_patch_masks(pred,truth)
    assert all(float(v)==1. for v in result['mean'].values())
    assert mbo(pred,truth).tolist()==[1.]
    row,edges=cells.cell_index(35,32,'cpu')
    assert len(row)==35 and len(edges)==33 and bool((edges[1:]>edges[:-1]).all())

def test_no_oracle_cluster_count_or_metric_reimplementation():
    code=(ROOT/'mod_benchmark/worker.py').read_text(encoding='utf-8')
    assert 'metric.evaluate_patch_masks(expanded,pixels)' in code
    assert 'fixed_terminal_epoch10' in code
    assert 'known_cluster_count_supplied=False' in code
    assert digest_json({'a':1,'b':2})==digest_json({'b':2,'a':1})
