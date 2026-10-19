import copy,io,json
import numpy as np
import torch
import pytest
from . import scope,run
from collaborative_test.SW_0239_native64_best_70000.test_training import fake_found


def selection_fixture():
    keys=scope.training.reference_score.METRICS;fp=dict(files={'completion_audit.py':'auditor'})
    means=dict(control512=(.86,.5,.5),within16_512=(.88,.5,.5),control128=(.80,.5,.5),within16_128=(.81,.5,.5),parent237=(.80,.5,.5),slot=(.78,.2,.2))
    score=dict(fingerprint=fp,seed_scope=[0,1,2],image_count_per_seed=32,full320_improvement_proven=False,scores={k:dict(zip(keys,v)) for k,v in means.items()},qualified_arms_for_separate_full320_confirmation=['control','within16'],selected_arm_for_separate_full320_confirmation='within16')
    row=dict(actual_Adam_steps=[512],actual_stage_updates=512,all_core_and_projection_parameters_unchanged=True,actual_final_model_restored_on_CPU=True)
    audit=dict(fingerprint=fp,score_sha256='score',audit_code_sha256='auditor',status='completed_six_actual512_checkpoints_and_raw_mask_audit',raw_masks_independently_rescored=576,independent_metrics_per_seed_means_and_deltas_maximum_abs_error=0.,all_prediction_protocols_and_Slot_before_evaluation_GT=True,training_audits={a:{str(s):copy.deepcopy(row) for s in scope.SEEDS} for a in scope.training.ARMS},selected_arm_for_separate_full320_confirmation='within16')
    return score,audit,fp


def test_audited_arm_selection_and_incomplete_evidence_rejection():
    score,audit,fp=selection_fixture()
    assert scope.verify_selection(score,audit,'score',fp,'auditor')=='within16'
    audit['training_audits']['control']['2']['actual_Adam_steps']=[511]
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score',fp,'auditor')
    score,audit,fp=selection_fixture()
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score',fp,'wrong auditor')
    score['selected_arm_for_separate_full320_confirmation']='control'
    with pytest.raises(ValueError):scope.verify_selection(score,audit,'score',fp,'auditor')


def test_negative_screen_closes_without_changing_registered_scope():
    score,audit,fp=selection_fixture()
    score['scores']['control512']['fg_ari']=.80;score['scores']['within16_512']['fg_ari']=.81
    score['qualified_arms_for_separate_full320_confirmation']=[];score['selected_arm_for_separate_full320_confirmation']=None;audit['selected_arm_for_separate_full320_confirmation']=None
    assert scope.verify_selection(score,audit,'score',fp,'auditor') is None
    assert scope.OFFSETS==tuple(range(320)) and scope.MINI==tuple(range(0,320,10))
    assert len(set(scope.OFFSETS)-set(scope.MINI))==288


def test_actual_serialized_encoder_restore_preserves_core_and_rejects_corruption():
    trained=fake_found();base=fake_found()
    with torch.no_grad():
        for p in trained.encoder.parameters():p.add_(.125)
    history=[dict(update=i+1,indices=[i%4]) for i in range(512)]
    saved=dict(completed_updates=512,history=history,optimizer_state=dict(state={0:dict(step=torch.tensor(512.))}),bindings={'registered':'fixture'},encoder_state=trained.encoder.state_dict())
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
