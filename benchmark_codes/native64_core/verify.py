"""Real native64 construction, state mapping and optional full rollout admission."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from portable import load, encode, predict, module, HERE


def verify(seed, device, image=None):
    start = time.monotonic()
    checkpoint = HERE.parent/'readout64_optimization/checkpoints'/f'sw0260_context_seed{seed}.pt'
    found, head, record = load(checkpoint, device)
    core = found.wrapped.core
    assert core.in_dim == 4096 and core.spike_spatial_grid_size == 64
    assert core.graph_generator.top_k == 512
    implementation = module('native64_verify_mapping', HERE/'model.py')
    parent = HERE.parent/'unsupervised_readout_benchmark/checkpoints'/f'sw0259_seed{seed}.pt'
    source = torch.load(parent, map_location='cpu', weights_only=True)['core16_state']
    p = implementation.parent_index()
    for name in implementation.NODE_RANKS:
        assert torch.equal(core.state_dict()[name].cpu(), source[name][p]), name
    pair = core.kuramoto.direction_learner
    assert torch.equal(pair.detach().cpu(), source['kuramoto.direction_learner'][p][:, p])
    assert torch.equal(core.sc.diagonal(), torch.ones(4096, device=device))
    assert core.sc.count_nonzero() == 4096
    distance = core.graph_generator.grid_distance
    assert float(distance[0, 1]) == .25 and float(distance[0, 64]) == .25
    assert core.kuramoto.omega.shape[0] == 4096
    assert core.membrane_layer.tau_m.numel() == 4096
    rgb = np.random.default_rng(261000+seed).integers(0, 256, (128,128,3), dtype=np.uint8)
    with torch.no_grad():
        gamma = encode(found, rgb)
    assert gamma.shape == (1,8,4096) and torch.isfinite(gamma).all()
    # Check that native features really enter routing, not expand32(row.features).
    features = torch.zeros(4096,144,device=device)
    features[1,0] = 1
    masks = torch.stack([torch.arange(4096,device=device)%2==0,
                         torch.arange(4096,device=device)%2==1])
    row = dict(features=features, masks=masks, prior=masks.float(), gamma=gamma[0].T)
    routed = head.routing(row)
    assert torch.equal(routed[1], features) and torch.isfinite(routed[0]).all()
    assert head.predict(row).shape == (64,64)
    # Training objective is live in the transferred readout; frozen core not rerun.
    head.requires_grad_(True)
    row['rgb'] = torch.zeros(4096,3,device=device)
    loss,_,_ = head(row)
    loss.backward()
    assert any(x.grad is not None and bool(torch.isfinite(x.grad).all())
               and bool(x.grad.abs().sum()>0) for x in head.fine.parameters())
    head.zero_grad(set_to_none=True)
    head.requires_grad_(False)
    result = dict(seed=seed,status='construction_mapping_encoding_routing_backward_passed',
                  core_grid=64,oscillator_locations=4096,components=4,
                  graph_top_k=512,native64_training_updates=0,
                  parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    if image is not None:
        from PIL import Image
        rgb = np.asarray(Image.open(image).convert('RGB'))
        labels = predict(found,head,rgb)
        assert labels.shape == (64,64) and np.isfinite(labels).all()
        result.update(full_1024_step_native64_inference_passed=True,
                      groups=int(len(np.unique(labels))),
                      prediction_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    result['elapsed_seconds'] = time.monotonic()-start
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0,choices=[0,1,2])
    p.add_argument('--device',default='cpu')
    p.add_argument('--image')
    p.add_argument('--report',required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result=verify(args.seed,args.device,args.image)
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
