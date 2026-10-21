import importlib,os
import numpy as np
import pytest
from . import scope

def test_gpu_routing_keeps_named_model_and_seed():
    from .queue import command
    cmd=command('245',2,'worker',3)
    assert '--worker' in cmd
    assert cmd[cmd.index('--model')+1]=='245' and cmd[cmd.index('--seed')+1]=='2'
    assert cmd[-2:]==['--gpu','3']
    with pytest.raises(ValueError):command('245',2,'worker',4)

def test_run_import_preserves_assigned_visibility(monkeypatch):
    from . import run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    importlib.reload(run)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2'

def test_reproduction_only_for_originally_missing244(tmp_path,monkeypatch):
    from . import factory
    monkeypatch.setattr(factory,'ARCHIVE',tmp_path)
    assert factory.need_reproduction('244',0) and factory.need_reproduction('244',2)
    assert not factory.need_reproduction('244',1) and not factory.need_reproduction('245',0)
    folder=factory.reproduction(0);folder.mkdir(parents=True);(folder/'training_report.json').write_text('{}')
    assert not factory.need_reproduction('244',0)

def test_history_checks_short_batch_and_rejects_changed_order():
    from .run import history_check
    row=dict(update=1,TRAIN_ids=scope.batch_ids(0,0).tolist(),loss=1.,frequency_denoising=1.,gamma_anchor=0.,within_variance=0.,gradient_norm=1.)
    assert history_check([row],0,False)==16
    row['TRAIN_ids']=row['TRAIN_ids'][::-1]
    with pytest.raises(ValueError):history_check([row],0,False)

def test_spec_preserves_existing_objectives_and_final_endpoint():
    spec=scope.specification()
    assert spec['loss_coefficients']=={'240':1.,'237':1.,'244':16.,'245':16.}
    assert spec['updates']==21880 and spec['last_batch']==8
    assert spec['architecture_and_readout_unchanged'] and spec['final_checkpoint_selection'].startswith('fixed terminal')

def test_actual_checkpoint_API_roundtrip_and_Adam_resume(tmp_path):
    import torch
    from .factory import old,payload
    encoder=torch.nn.Linear(2,2);projection=torch.nn.Linear(2,1,bias=False)
    opt=torch.optim.Adam(list(encoder.parameters())+list(projection.parameters()),lr=.001)
    x=torch.ones((2,2));loss=projection(encoder(x)).square().mean();loss.backward();opt.step()
    expected={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    runtime={'fixture':'CPU-only'};binding={'fixed':True}
    old.checkpoint.save_boundary(tmp_path,completed_updates=1,wrapped=projection,encoder=encoder,optimizer=opt,
        history=[{'update':1}],bindings=binding,runtime=runtime,device='cpu')
    saved=payload(tmp_path/'boundary.ptx')
    assert saved['completed_updates']==1 and saved['runtime']==runtime
    with torch.no_grad():encoder.weight.add_(100)
    old.checkpoint.load_latest(tmp_path,wrapped=projection,encoder=encoder,optimizer=opt,expected_bindings=binding,
        expected_runtime=runtime,device='cpu',max_updates=2)
    assert all(torch.equal(encoder.state_dict()[k],v) for k,v in expected.items())
    assert {int(v['step'].item()) for v in opt.state.values()}=={1}
