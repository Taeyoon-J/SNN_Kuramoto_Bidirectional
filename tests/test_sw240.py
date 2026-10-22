"""Portability and actual encoder/projection training checks, without research imports."""
import ast
import hashlib
from pathlib import Path
import sys
import io

import numpy as np
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snn_kuramoto_bidirectional import SW240Model
from snn_kuramoto_bidirectional.sw240.model import default_config
from snn_kuramoto_bidirectional.sw240.model import model_sha256,load_model
from snn_kuramoto_bidirectional.sw240.data import train_ids,modal_labels
from snn_kuramoto_bidirectional.sw240.loss import effective_frequency,make_teacher,teacher_objective
from snn_kuramoto_bidirectional.sw240.partition import foreground_induced_merge
from snn_kuramoto_bidirectional.sw240.rollout import late_rollout
from snn_kuramoto_bidirectional.sw240.integration import PhaseStateIntegration
from snn_kuramoto_bidirectional.sw240.correlation import native_component_rho_q
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

torch.set_num_threads(1)


def test_standalone_imports_and_syntax():
    for path in ROOT.rglob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        if 'tools' in path.relative_to(ROOT).parts:
            continue
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith('collaborative_test')
            if isinstance(node,ast.Import):
                assert not any(v.name.startswith('collaborative_test') for v in node.names)
    assert not any(name.startswith('collaborative_test') for name in sys.modules)
    for name,module in tuple(sys.modules.items()):
        if name.startswith('snn_kuramoto_bidirectional') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT)


def test_registered_training_split_and_modal_ties():
    ids=train_ids()
    assert len(ids)==35000 and len(np.unique(ids))==35000
    assert not np.any((ids>=1000)&(ids<1640))
    assert hashlib.sha256(ids.tobytes()).hexdigest()=='587e3a019299c32f02ba4bc49a710088e3f2429358330f1a97c131e13313df00'
    pixels=np.zeros((1,128,128),np.int64)
    pixels[0,:2,:2]=[[2,1],[2,1]]
    assert modal_labels(pixels)[0,0,0]==1


def test_native_encoder_projection_credit_and_frozen_core():
    model=SW240Model(feature_mean=torch.zeros(1,8,1,1),feature_std=torch.ones(1,8,1,1))
    optimizer=model.optimizer_for_training()
    groups=optimizer.param_groups
    assert [g['lr'] for g in groups]==[3e-5,3e-4,3e-4]
    params=[p for g in groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})
    assert [k for k,v in model.wrapped.named_parameters() if v.requires_grad]==['core.gamma_channel_proj.weight']
    frozen={k:v.detach().clone() for k,v in model.wrapped.state_dict().items() if k!='core.gamma_channel_proj.weight'}
    rgb=torch.from_numpy(np.random.RandomState(240).randint(0,256,(1,128,128,3),dtype=np.uint8))
    gamma=model.encode(rgb)
    assert gamma.shape==(1,8,4096)
    frequency=effective_frequency(model.wrapped.core,gamma)
    labels=np.zeros((1,64,64),np.int64);labels[:,:,:32]=1
    targets=make_teacher(gamma.detach().numpy(),frequency.detach().numpy(),labels)
    loss,_=teacher_objective(gamma,frequency,targets)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert model.wrapped.core.gamma_channel_proj.weight.grad.norm()>0
    optimizer.step()
    assert all(torch.equal(value,model.wrapped.state_dict()[key]) for key,value in frozen.items())
    with pytest.raises(RuntimeError,match='trained portable'):
        model.predict(rgb)


def test_actual_phase_dendrite_membrane_rollout_and_q():
    hp=S2NetHyperparameters(num_regions=16,num_feature_maps=8,osc_dim=4,
        gamma_drive_mode='static',graph_mode='static',sc=torch.eye(16),k=16.,freq_gain=2.,
        kuramoto_backend='factorized',spike_per_component=True,
        dendritic_projection='shared',gate_mode='raw',membrane_vth=.06).validate()
    wrapped=PhaseStateIntegration(S2NetCore(hp,device='cpu'),'phase')
    gamma=torch.randn(1,8,16)
    with torch.no_grad():
        trace=late_rollout(wrapped,gamma,total_steps=12,live_tail_steps=0)
        rho,q=native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=4)
    assert trace['component_spikes'].shape==(1,4,16,12)
    assert rho.shape==(1,4,16,16) and q.shape==(1,16,16)
    assert torch.isfinite(q).all() and bool(((q>=0)&(q<=1)).all())


def test_readout_retains_foreground_and_separates_negative_pairs():
    support=np.array([[1,1],[0,2]],np.int64)
    q=np.full((4,4),.1);np.fill_diagonal(q,1.);q[0,1]=q[1,0]=.9
    result=foreground_induced_merge(q,np.eye(4),8.,support,grid_shape=(2,2))['labels']
    assert np.array_equal(result>0,support>0)
    assert result[0,0]==result[0,1] and result[1,1]!=result[0,1]


def test_actual_shape_portable_bundle_strict_roundtrip_and_digest():
    model=SW240Model(feature_mean=torch.zeros(1,8,1,1),feature_std=torch.ones(1,8,1,1))
    config=dict(model.config,seed=0,model_sha256=model_sha256(model))
    bundle=dict(config=config,wrapped_state=model.wrapped.state_dict(),encoder_state=model.encoder.state_dict(),
                feature_mean=model.feature_mean,feature_std=model.feature_std,
                teacher_encoder_state=model.encoder.state_dict(),
                teacher_projection_state=model.wrapped.core.gamma_channel_proj.state_dict(),
                teacher_model_sha256=config['model_sha256'])
    stream=io.BytesIO()
    torch.save(bundle,stream);stream.seek(0)
    restored=load_model(stream)
    assert restored.checkpoint_loaded and model_sha256(restored)==config['model_sha256']
    rgb=torch.zeros(1,128,128,3,dtype=torch.uint8)
    with torch.no_grad():
        assert torch.equal(model.encode(rgb),restored.encode(rgb))
    del restored,stream
    bundle['config']=dict(config,model_sha256='0'*64)
    stream=io.BytesIO();torch.save(bundle,stream);stream.seek(0)
    with pytest.raises(ValueError,match='SHA'):
        load_model(stream)
