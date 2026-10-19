import copy,io,json
import numpy as np
import torch
import pytest
from . import scope,run
from collaborative_test.SW_0239_native64_best_70000.test_training import fake_found


def selection_fixture():
    keys=scope.training.reference_score.METRICS
    score=dict(fingerprint='fixture',seed_scope=[0,1,2],image_count_per_seed=64,full320_improvement_proven=False,
        scores={k:dict(zip(keys,v)) for k,v in dict(control=(.80,.50,.50),parent237=(.81,.51,.51),slot=(.78,.20,.20),within16=(.86,.48,.48)).items()},within16_promising_for_separate_full320_confirmation=True)
    row=dict(actual_Adam_steps=[128],actual_stage_updates=128,all_core_and_projection_parameters_unchanged=True,actual_final_model_restored_on_CPU=True)
    audit=dict(fingerprint='fixture',score_sha256='score',status='completed_six_actual_CPU_checkpoint_audits_and_raw_mask_rescoring',raw_masks_independently_rescored=768,independent_metrics_per_seed_means_and_deltas_maximum_abs_error=0.,all_prediction_protocols_validated_before_evaluation_GT=True,
        training_audits={a:{str(s):copy.deepcopy(row) for s in scope.SEEDS} for a in scope.training.ARMS})
    return score,audit


def test_conditional_gate_rejects_inconsistent_or_incomplete_evidence():
    score,audit=selection_fixture()
    assert scope.verify_selection(score,audit,'score','fixture',scope.AUDIT_CODE_SHA256)
    audit['training_audits']['control']['2']['actual_Adam_steps']=[127]
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score','fixture',scope.AUDIT_CODE_SHA256)
    score,audit=selection_fixture()
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score','fixture','different audit source')
    score['within16_promising_for_separate_full320_confirmation']=False
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score','fixture',scope.AUDIT_CODE_SHA256)


def test_negative_completed_screen_closes_expansion_and_exact_scope():
    score,audit=selection_fixture()
    score['scores']['within16']['fg_ari']=.79;score['within16_promising_for_separate_full320_confirmation']=False
    assert not scope.verify_selection(score,audit,'score','fixture',scope.AUDIT_CODE_SHA256)
    assert scope.OFFSETS==tuple(range(320)) and scope.MINI==tuple(range(0,320,5))
    assert len(set(scope.OFFSETS)-set(scope.MINI))==256 and scope.MINI[0]==0


def test_actual_serialized_encoder_restore_preserves_core_and_rejects_corruption():
    trained=fake_found();base=fake_found()
    with torch.no_grad():
        for p in trained.encoder.parameters():p.add_(.125)
    history=[dict(update=i+1,indices=[i%4]) for i in range(128)]
    saved=dict(completed_updates=128,history=history,optimizer_state=dict(state={0:dict(step=torch.tensor(128.))}),bindings={'registered':'fixture'},encoder_state=trained.encoder.state_dict())
    b=io.BytesIO();torch.save(saved,b);payload=b.getvalue();b=io.BytesIO();torch.save(dict(payload_bytes=payload,payload_sha256=run.sha(payload)),b);raw=b.getvalue()
    report=dict(checkpoint_sha256=run.sha(raw),checkpoint_binding=saved['bindings'],history_sha256=run.sha(json.dumps(history,sort_keys=True,allow_nan=False).encode()),binding=dict(frozen_core_sha256=run.training.state_sha(base.wrapped.state_dict())),final_encoder_sha256=run.training.state_sha(trained.encoder.state_dict()),final_model_sha256=run.reference.source.source.full.screen.native._state_sha(trained.wrapped,trained.encoder))
    before=run.training.state_sha(base.wrapped.state_dict());scope.restore_encoder_bytes(base,report,raw)
    assert run.training.state_sha(base.encoder.state_dict())==report['final_encoder_sha256']
    assert run.training.state_sha(base.wrapped.state_dict())==before
    with pytest.raises(ValueError):scope.restore_encoder_bytes(fake_found(),report,raw+b'changed')


def test_exclusive_full_scope_queue_has_no_training_arm_argument():
    from . import queue
    assert queue.api is run
    command=queue.worker_command(2,3)
    assert '--arm' not in command and command[-1]=='3'
    with pytest.raises(ValueError):queue.worker_command(2,4)
