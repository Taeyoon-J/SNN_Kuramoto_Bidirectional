"""Explicit child lineage: continue parent Adam/RNG without changing its files."""
import json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0211_spike_anchor_center_assignment import training as library


def extend(features,labels,protocol,*,parent_work,work,output,device='cuda:0',parent_steps=128,steps=512):
    parent_work,work,output=map(Path,(parent_work,work,output));device=torch.device(device)
    if steps<=parent_steps:raise ValueError('strictly larger child budget required')
    old=library.checkpoint.inspect_committed_metadata(parent_work)['bindings']
    mean,scale=library.normalization(features);order=library.batch_order(len(features),steps)
    common=dict(science=library.cache.fingerprint(),cache_protocol_sha256=library.cache.sha(json.dumps(protocol,sort_keys=True).encode()),
        features_sha256=library.cache.sha(features.tobytes()),TRAIN_labels_sha256=library.cache.sha(labels.tobytes()),
        mean_sha256=library.cache.sha(mean.tobytes()),scale_sha256=library.cache.sha(scale.tobytes()),core_optimizer_steps=0)
    if any(old.get(key)!=value for key,value in common.items()) or old['steps']!=parent_steps or old['batch_order']!=order[:parent_steps]:raise ValueError('parent data/order/scientific source changed')
    parent_hash=library.cache.sha((parent_work/'boundary.ptx').read_bytes())
    origin=dict(parent_boundary_sha256=parent_hash,parent_steps=parent_steps,total_steps=steps,new_steps=steps-parent_steps,
                parent_bindings=old,source='explicit continuation; original parent files remain unchanged')
    origin_path=output/'origin.json'
    if origin_path.exists():
        if json.loads(origin_path.read_bytes())!=origin:raise ValueError('child parent lineage changed')
    else:library.cache.write_json(origin_path,origin)
    if not (work/'boundary.ptx').exists():
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
            torch.manual_seed(211002);head=library.CenterHead(channels=features.shape[1]).to(device)
            head.mean.copy_(torch.from_numpy(mean).to(device));head.scale.copy_(torch.from_numpy(scale).to(device))
            initial=library.state_sha(head.state_dict())
            if initial!=old['initial_model_sha256']:raise ValueError('head initialization/library mismatch')
            optimizer=torch.optim.Adam([dict(name='center_readout',params=list(head.parameters()),lr=3e-4)])
            empty=torch.nn.Identity();runtime=library.compatible_runtime(library.checkpoint.runtime_signature(device))
            restored=library.checkpoint.load_latest(parent_work,wrapped=head,encoder=empty,optimizer=optimizer,
                expected_bindings=old,expected_runtime=runtime,device=device,max_updates=parent_steps)
            if restored['completed_updates']!=parent_steps:raise ValueError('parent endpoint incomplete')
            binding=dict(common,steps=steps,snapshots=[steps],batch_order=order,initial_model_sha256=initial)
            library.checkpoint.save_boundary(work,completed_updates=parent_steps,wrapped=head,encoder=empty,optimizer=optimizer,
                history=restored['history'],bindings=binding,runtime=runtime,device=device)
    report=library.fit(features,labels,protocol,device=device,steps=steps,snapshots=(steps,),work=work,output=output)
    parent_payload=library.cache.source.full.unpack_checkpoint((parent_work/'boundary.ptx').read_bytes())
    if report['history'][:parent_steps]!=parent_payload['history'] or report['actual_Adam_steps']!=[steps]:raise ValueError('child lost actual parent Adam/history')
    if library.cache.sha((parent_work/'boundary.ptx').read_bytes())!=parent_hash:raise ValueError('parent checkpoint mutated')
    return report
