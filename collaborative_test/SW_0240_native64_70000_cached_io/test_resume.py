"""A real239 Adam boundary must continue bitwise identically through the I/O successor."""
import json
import h5py
import numpy as np
import torch
from collaborative_test.SW_0239_native64_best_70000 import train as original
from collaborative_test.SW_0239_native64_best_70000 import data as original_data
from collaborative_test.SW_0239_native64_best_70000.test_training import fake_found
from collaborative_test.SW_0240_native64_70000_cached_io import train, data, bootstrap


def test_real239_Adam_model_RNG_history_rebind_matches_uninterrupted(tmp_path, monkeypatch):
    dataset=tmp_path/'input.h5'
    with h5py.File(dataset,'w') as stream:stream['image']=np.arange(12,dtype=np.float32).reshape(4,3)/10
    stat=dataset.stat()
    for module in (data,original_data):
        for name,value in (('BATCH',2),('EPOCHS',2),('TRAIN_COUNT',4),('STEPS_PER_EPOCH',2),('TOTAL_STEPS',4)):
            monkeypatch.setattr(module,name,value)
        monkeypatch.setattr(module,'epoch_ids',lambda seed,epoch:np.arange(4))
        monkeypatch.setattr(module,'read_batch',lambda stream,ids:(np.asarray(stream['image'][ids]),np.zeros((len(ids),1,1),dtype=np.int64)))
    monkeypatch.setattr(original.helpers,'effective_frequency',lambda core,gamma:core.gamma_channel_proj(gamma.transpose(1,2)))
    monkeypatch.setattr(original,'fingerprint',lambda:{'registered239':True})
    model_sha=train.source.source.full.screen.native._state_sha
    def prepared(stage):
        found=fake_found();teacher=fake_found()
        teacher.wrapped.requires_grad_(False);teacher.encoder.requires_grad_(False)
        binding=dict(fingerprint={'registered239':True},dataset=dict(path=str(dataset),size=stat.st_size,mtime_ns=stat.st_mtime_ns),
            frozen_nonprojection_core_sha256=train.projection.frozen_core_sha(found),
            input_statistics_sha256=train.helpers.statistics_sha(found),teacher_model_sha256=model_sha(teacher.wrapped,teacher.encoder))
        return dict(found=found,teacher=teacher,optimizer=stage.optimizer_for(found),seed=0,binding=binding)
    runtime=train.checkpoint.runtime_signature('cpu')
    control=prepared(original);monkeypatch.setattr(original,'ARCHIVE',tmp_path/'c')
    original.write_json(original.archive_for(0)/'admission.json',dict(binding=control['binding']))
    expected=original.fit(control,'cpu',runtime)
    parent=prepared(original);monkeypatch.setattr(original,'ARCHIVE',tmp_path/'o')
    original.write_json(original.archive_for(0)/'admission.json',dict(binding=parent['binding']))
    teacher=original.teacher_targets;calls=[]
    def interrupt(*args):
        calls.append(1)
        if len(calls)==3:raise InterruptedError('freeze actual239 checkpoint after update2')
        return teacher(*args)
    monkeypatch.setattr(original,'teacher_targets',interrupt)
    try:original.fit(parent,'cpu',runtime)
    except InterruptedError:pass
    else:raise AssertionError('parent was not interrupted')
    monkeypatch.setattr(original,'teacher_targets',teacher)
    inherited=(original.archive_for(0)/'encoder_work/boundary.ptx').read_bytes()
    child=prepared(train);old_binding=dict(child['binding'])
    child['original239_binding']=old_binding
    child['binding']=dict(old_binding,fingerprint={'cached240':True},inherited239_stage_updates=2)
    monkeypatch.setattr(train,'ARCHIVE',tmp_path/'n')
    destination=train.archive_for(0)/'inherited239/boundary.ptx';destination.parent.mkdir(parents=True)
    destination.write_bytes(inherited)
    manifest=dict(old_single_GPU_owner_confirmed_stopped=True,source239_fingerprint={'registered239':True},stopped_processes=[],
        seeds={'0':dict(completed_updates=2,checkpoint_sha256=train.sha(inherited),
            original239_admission_sha256=train.sha((original.archive_for(0)/'admission.json').read_bytes()))})
    train.write_json(train.ARCHIVE/'migration.json',manifest)
    bootstrap.confirm_old_owner_stopped()
    proof=bootstrap.admit_cpu(child)
    assert proof['inherited239_updates']==2 and proof['actual_CPU_model_Adam_gradient_restore_passed']
    assert not child['optimizer'].state
    train.write_json(train.archive_for(0)/'admission.json',dict(binding=child['binding']))
    actual=train.fit(child,'cpu',runtime)
    assert actual['actual_Adam_steps']==[4]
    assert actual['inherited239_stage_updates']==2 and actual['updates_executed_in_cached_successor']==2
    for key in ('final_model_sha256','final_encoder_sha256','history_sha256'):
        assert actual[key]==expected[key]
    assert all(torch.equal(a,b) for a,b in zip(control['found'].encoder.parameters(),child['found'].encoder.parameters()))
