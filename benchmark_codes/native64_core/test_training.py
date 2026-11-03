"""Compare deterministic pooling backward to stock CPU, plus chunked decoder gradients."""
import copy,sys,types
from pathlib import Path
import torch
from training import _DeterministicAverage,reconstruction_loss
from portable import HERE,module

def main():
    torch.set_num_threads(1);torch.manual_seed(261)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.randn(1,8,126,126,requires_grad=True)
    gradient=torch.randn(1,8,64,64)
    expected=torch.nn.functional.adaptive_avg_pool2d(x,(64,64));expected.backward(gradient)
    actual_x=x.detach().to(device).requires_grad_(True)
    actual=_DeterministicAverage.apply(actual_x,(64,64));actual.backward(gradient.to(device))
    reference_forward=torch.nn.functional.adaptive_avg_pool2d(actual_x.detach(),(64,64))
    assert torch.equal(actual.detach(),reference_forward)
    assert torch.allclose(actual_x.grad.cpu(),x.grad,atol=2e-6,rtol=2e-5)
    context=module('train_test_context',HERE.parent/'readout64_optimization/portable.py')
    found,head,_=context.load(HERE.parent/'readout64_optimization/checkpoints/sw0260_context_seed0.pt',device);del found
    native=module('train_test_native',HERE/'model.py');head.routing=types.MethodType(native.native_routing,head)
    head.requires_grad_(True);head.content_encoder.requires_grad_(False)
    head.head.prior_scale.requires_grad_(False);head.head.residual_scale.requires_grad_(False)
    other=copy.deepcopy(head)
    # Three groups exercises an incomplete final decoder chunk.
    index=torch.arange(4096,device=device)%3;masks=index[None]==torch.arange(3,device=device)[:,None]
    row=dict(features=torch.randn(4096,144,device=device),gamma=torch.randn(4096,8,device=device),
             masks=masks,prior=masks.float(),rgb=torch.rand(4096,3,device=device))
    loss,_,_=head(row);loss.backward()
    chunked,_=reconstruction_loss(other,row,2);chunked.backward()
    assert torch.allclose(loss,chunked,atol=1e-6,rtol=1e-5)
    for (name,p),(_,q) in zip(head.named_parameters(),other.named_parameters()):
        if p.grad is not None:assert q.grad is not None and torch.allclose(p.grad,q.grad,atol=2e-5,rtol=2e-3),name
    print('pool_forward_exact_backward_equivalent_and_chunked_decoder_gradient_passed',flush=True)

if __name__=='__main__':main()
