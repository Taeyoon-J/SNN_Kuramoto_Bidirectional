"""Verify that reconstruction trains unchanged252 head without any mask targets."""
import importlib.util
from pathlib import Path
import torch

def main():
    torch.set_num_threads(1)
    path=Path(__file__).with_name('model.py')
    spec=importlib.util.spec_from_file_location('sw259_test_model',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(5)
    features=torch.randn(1024,144)
    regions=(torch.arange(1024).reshape(32,32)//64).numpy()
    target=torch.rand(1024,3)
    for arm in m.ARMS:
        model=m.ReconstructionReadout(0)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        initial={k:v.detach().clone() for k,v in model.head.state_dict().items()}
        rgb,detail=model(features,regions)
        assert rgb.shape==(1024,3) and torch.allclose(detail['probability'].sum(0),torch.ones(1024),atol=1e-6)
        loss,_=m.objective(rgb,target,detail,arm);loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None)>0
        assert sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)>0
        opt.step()
        assert any(not torch.equal(v,model.head.state_dict()[k]) for k,v in initial.items())
        pred=model.predict(features,regions)
        assert pred.shape==(32,32) and pred.min()>0
        print(arm,'gradient_and_assignment_passed')

if __name__=='__main__':main()
