"""Actual encoder Adam interruption must retain moments, history and frozen core."""
import copy,json
import numpy as np
import torch
from collaborative_test.SW_0239_native64_best_70000.test_training import fake_found
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher
from . import run
from .loss import groups_from_labels


def test_real_encoder_Adam_resume_matches_uninterrupted(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'STEPS',4);monkeypatch.setattr(run,'BATCH',2);monkeypatch.setattr(run,'CHECKPOINT_EVERY',1)
    monkeypatch.setattr(run.helpers,'effective_frequency',lambda core,gamma:core.gamma_channel_proj(gamma.transpose(1,2)))
    rgb=np.arange(12,dtype=np.float32).reshape(4,3)/10;labels=np.zeros((4,1,1),dtype=np.int64)
    teacher=fake_found()
    with torch.no_grad():
        for p in teacher.encoder.parameters():p.add_(.03)
        gamma=teacher.encode(torch.from_numpy(rgb));frequency=run.helpers.effective_frequency(teacher.wrapped.core,gamma)
    targets=make_teacher(gamma.numpy(),frequency.numpy(),labels)
    def prepared():
        found=fake_found();opt=run.helpers.optimizer_for(found)
        return dict(seed=1,arm='control',found=found,optimizer=opt,rgb=rgb,labels=labels,targets=targets,
            groups=groups_from_labels(labels),binding=dict(frozen_core_sha256=run.state_sha(found.wrapped.state_dict()),
                teacher_arrays_sha256=run.state_sha(targets),input_statistics_sha256=run.helpers.statistics_sha(found)))
    runtime=run.checkpoint.runtime_signature('cpu')
    full=prepared();monkeypatch.setattr(run,'ROOT_ARCHIVE',tmp_path/'full')
    run.write_json(run.archive_for(1,'control')/'admission.json',dict(binding=full['binding']))
    expected=run.fit(full,'cpu',runtime)
    interrupted=prepared();monkeypatch.setattr(run,'ROOT_ARCHIVE',tmp_path/'resume')
    run.write_json(run.archive_for(1,'control')/'admission.json',dict(binding=interrupted['binding']))
    original=run.batch_loss;calls=[]
    def stop_after_two(*args):
        if torch.is_grad_enabled():
            calls.append(1)
            if len(calls)==3:raise InterruptedError('test interrupted after actual Adam2')
        return original(*args)
    monkeypatch.setattr(run,'batch_loss',stop_after_two)
    try:run.fit(interrupted,'cpu',runtime)
    except InterruptedError:pass
    else:raise AssertionError('expected interruption did not happen')
    monkeypatch.setattr(run,'batch_loss',original)
    metadata=run.checkpoint.inspect_committed_metadata(run.archive_for(1,'control')/'encoder_work')
    assert metadata['completed_updates']==2
    resumed=prepared();actual=run.fit(resumed,'cpu',runtime)
    assert actual['actual_Adam_steps']==[4]
    for key in ('final_encoder_sha256','final_model_sha256','history_sha256'):
        assert actual[key]==expected[key]
    assert run.state_sha(resumed['found'].wrapped.state_dict())==resumed['binding']['frozen_core_sha256']
    assert actual['new_core_updates']==0 and actual['new_encoder_updates']==4
