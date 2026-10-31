"""CPU numerical controls at a reduced grid; native64 speed needs separate CUDA profiling."""
import argparse,json,sys,types
from pathlib import Path
import numpy as np
import torch
from portable_runtime import HERE,configure


def reduced_constructor():
    # Test-only geometry, never used by run.py or the scientific queue.
    from full_benchmark import model as source
    text=Path(source.__file__).read_text().replace('self.grid=64','self.grid=16')
    text=text.replace('graph_top_k=128 if self.grid==32 else 512','graph_top_k=32')
    namespace=dict(source.__dict__);exec(compile(text,'reduced_grid_numerical_control','exec'),namespace)
    return namespace['FreshModel']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    configure(HERE/'configs/frontier.json');torch.set_num_threads(1)
    from full_benchmark.microbatch import backward_microbatch,select_microbatch
    from full_benchmark.loss import InitialTeacher
    from profile_microbatch import compare
    assert select_microbatch(24000)==2 and select_microbatch(48000)==4
    assert select_microbatch(15000)==1 and select_microbatch(10000) is None
    assert select_microbatch(24000,'4') is None
    assert select_microbatch(24000,'4',admission=True)==2
    Constructor=reduced_constructor();rng=np.random.RandomState(1010)
    rgb=torch.from_numpy(rng.randint(0,256,(4,128,128,3),dtype=np.uint8))
    y,x=np.indices((16,16));truth=np.stack([(x//5+3*(y//6)+i)%7 for i in range(4)]).astype(np.int64)
    ids=np.array([14,7,103,59]);results=[]
    for variant in (253,252,254,240):
        model=Constructor(variant,0);teacher=InitialTeacher(model) if variant!=253 else None
        if model.head is not None:
            with torch.no_grad():model.head.residual_scale.fill_(.25)
        gradients={};losses={}
        for size in (1,2,4):
            model.zero_grad(set_to_none=True);value=0.
            for start in range(0,4,size):
                loss,_=backward_microbatch(model,rgb[start:start+size],truth[start:start+size] if teacher else None,
                    teacher,ids[start:start+size],2,16);value+=loss
            gradients[size]={n:v.grad.detach().clone() for n,v in model.named_parameters() if v.grad is not None}
            losses[size]=value
        comparisons={str(s):compare(gradients[1],gradients[s]) for s in (2,4)}
        for size,values in comparisons.items():
            assert abs(losses[int(size)]-losses[1])<1e-4,(variant,losses)
            assert values['relative_l2']<.001 and values['cosine']>.999999,(variant,size,values)
        results.append(dict(model=variant,losses=losses,gradient_comparisons=comparisons))
    # Exercise image-indexed noise even though the current scientific configuration
    # uses deterministic gamma initialization. Stop after initialization, before the rollout.
    import full_benchmark.model as module
    original=module.late_rollout;captured=[]
    def capture(wrapped,gamma,*,theta_initializer=None,**kwargs):
        core=wrapped.core
        # Isolate RNG identity from convolution kernels' allowed batch roundoff.
        drive=torch.zeros(len(gamma),core.in_dim,core.osc_dim,device=gamma.device)
        value=core._init_theta(drive,len(gamma)) if theta_initializer is None else theta_initializer(core,drive,len(gamma))
        captured.append(value.detach().clone());raise StopIteration
    # The reduced constructor has its own function globals; replace its rollout hook.
    Constructor.forward.__globals__['late_rollout']=capture
    model=Constructor(253,0);model.wrapped.core.theta_init='gamma_noise'
    for index in range(4):
        try:model(rgb[index:index+1],inference_seed=255000+int(ids[index])+200000)
        except StopIteration:pass
    reference=torch.cat(captured);captured.clear()
    try:model(rgb,inference_seeds=[255000+int(i)+200000 for i in ids])
    except StopIteration:pass
    torch.testing.assert_close(reference,captured[0],rtol=0,atol=0)
    result=dict(status='all_four_reduced_grid_loss_gradient_and_per_image_noise_controls_passed',
        grid=16,steps=1024,tail=64,logical_batch=16,physical_sizes=[1,2,4],results=results,
        per_sample_gamma_noise_exact=True,native64_GPU_speed_claimed=False)
    result['nonzero_final_head_residual_tested']=True
    result['automatic_memory_policy_checked']=True
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
