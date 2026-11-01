"""No recurrent simulation during encoder optimization; fixed pretrained core."""
import copy,fcntl,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from frozen_core_benchmark.data import Dataset
from frozen_core_benchmark.model import FrozenCoreModel,digest,checked_file
from frozen_core_benchmark.engine import scientific_sources as parent_sources


def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'run_separate_encoder.py']
    return {**parent_sources(),**{str(p):c.sha(p) for p in paths}}


def state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def bound(path,value):
    value=json.loads(json.dumps(value))
    if path.exists():assert json.loads(path.read_bytes())==value,('changed binding',str(path))
    else:c.write(path,value)
    return c.sha(path)


def settings():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available(),'actual CUDA required'
    limit=int(os.environ.get('SW0257_ALLOCATOR_MIB','13000'))
    torch.cuda.set_per_process_memory_fraction(limit*1024**2/torch.cuda.get_device_properties(0).total_memory,0)


def encoder_parts(model):
    if model.head is not None:model.head.requires_grad_(False)
    if model.variant==252:
        teacher=copy.deepcopy(model).cpu().eval().requires_grad_(False)
        groups=[dict(params=list(model.encoder.base_encoder.parameters()),lr=3e-5),
                dict(params=list(model.encoder.context.parameters()),lr=3e-4)]
        decoder=None
    else:
        from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureDecoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(257000+model.seed);decoder=CNNFeatureDecoder(8,3,3).to('cuda:0')
        teacher=None;groups=[dict(params=list(model.encoder.parameters())+list(decoder.parameters()),lr=1e-3)]
    return teacher,decoder,torch.optim.Adam(groups)


def encoder_loss(model,teacher,decoder,data,ids):
    rgb=data.rgb(ids)
    if model.variant==252:
        from snn_kuramoto_bidirectional.sw240.loss import effective_frequency,make_teacher,teacher_objective
        with torch.no_grad():
            gamma=teacher.encode(torch.from_numpy(rgb))
            freq=effective_frequency(teacher.wrapped.core,gamma)
            targets=make_teacher(gamma.numpy(),freq.numpy(),data.labels(ids))
        gamma=model.encode(torch.from_numpy(rgb).to('cuda:0'))
        loss,_=teacher_objective(gamma,effective_frequency(model.wrapped.core,gamma),targets)
    else:
        rgb=torch.from_numpy(rgb).to('cuda:0').permute(0,3,1,2).float()/255.
        loss=F.mse_loss(decoder(model.encoder(rgb)),rgb)
        assert not data.labels_opened,'253 encoder accessed mask targets'
    return loss


def train_encoder(model,data,out,*,admission=False):
    out.mkdir(parents=True,exist_ok=True)
    teacher,decoder,optimizer=encoder_parts(model)
    initial=state(model.encoder);decoder_initial=state(decoder) if decoder is not None else None
    params=list(model.encoder.parameters())+(list(decoder.parameters()) if decoder is not None else [])
    coverage={name:0. for name,_ in model.encoder.named_parameters()}
    updates=seen=first=offset=0;checkpoint=out/'resume.pt'
    # Reconstruction of teacher precedes resume, so it always uses the immutable source.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
        assert saved['binding_sha256']==c.sha(out.parent/'binding.json')
        model.encoder.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer'])
        if decoder is not None:decoder.load_state_dict(saved['decoder'])
        updates=saved['updates'];seen=saved['images_seen'];first=saved['epoch'];offset=saved['offset'];coverage=saved['coverage']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        assert first*35000+offset==seen
    def snapshot(epoch,end):
        model.assert_frozen()
        r.save(checkpoint,dict(encoder=state(model.encoder),decoder=state(decoder) if decoder is not None else None,
            optimizer=optimizer.state_dict(),updates=updates,images_seen=seen,epoch=epoch,offset=end,
            coverage=coverage,binding_sha256=c.sha(out.parent/'binding.json'),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        c.write(out/'committed.json',dict(updates=updates,images_seen=seen,epoch=epoch,offset=end,
            checkpoint_sha256=c.sha(checkpoint)))
    core=model.wrapped.core;original=core.kuramoto.forward;calls=0
    def forbidden(*args,**kwargs):
        nonlocal calls
        calls+=1;raise AssertionError('recurrent core invoked during independent encoder learning')
    core.kuramoto.forward=forbidden
    started=time.time()
    try:
        for epoch in range(first,1 if admission else 10):
            order=data.train[np.random.RandomState(249000+model.seed+10000*epoch).permutation(35000)]
            for start in range(offset if epoch==first else 0,35000,16):
                ids=order[start:start+16];optimizer.zero_grad(set_to_none=True)
                loss=encoder_loss(model,teacher,decoder,data,ids);loss.backward()
                for name,p in model.encoder.named_parameters():
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
                    coverage[name]+=float(p.grad.detach().abs().sum())
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                updates+=1;seen+=len(ids);end=start+len(ids)
                next_epoch=epoch+int(end==35000);next_offset=0 if end==35000 else end
                with (out/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=updates,epoch=epoch,ids=ids.tolist(),loss=float(loss.detach()),
                        gradient_norm=float(norm),images_seen=seen,seconds=time.time()-started),allow_nan=False)+'\n')
                if updates%128==0 or updates==1 or end==35000 or admission:snapshot(next_epoch,next_offset)
                c.write(out/'progress.json',dict(updates=updates,images_seen=seen,epoch=epoch+1,loss=float(loss.detach())))
                if admission and updates==2:break
            if admission:break
    finally:core.kuramoto.forward=original
    assert (seen,updates)==((32,2) if admission else (350000,21880))
    assert all(v>0 for v in coverage.values()),coverage
    assert any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.encoder.state_dict().items())
    if decoder is not None:assert any(not torch.equal(decoder_initial[k],v.detach().cpu()) for k,v in decoder.state_dict().items())
    model.assert_frozen();assert calls==0
    saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
    before=digest(model.encoder);model.encoder.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer'])
    assert digest(model.encoder)==before
    assert {int(v['step'].item()) for v in optimizer.state.values()}=={updates}
    result=dict(status='independent_encoder_complete',images_seen=seen,updates=updates,
        diagnostic_only=admission,core_simulation_calls=0,core_bitwise_fixed=True,
        frozen_core_sha256=model.frozen_initial_sha256,encoder_sha256=before,gradient_coverage=coverage,
        mask_used=model.variant==252,checkpoint_sha256=c.sha(checkpoint),optimizer_restore_verified=True,
        seconds=time.time()-started)
    c.write(out/'complete.json',result)
    return result


def train_head(model,data,out,head_record,*,admission=False):
    from readout_benchmark.head import activity_features
    from full_benchmark.loss import regions_from_output
    from neutral_readout.head import objective
    out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    model.encoder.requires_grad_(False);model.head.requires_grad_(True)
    saved=torch.load(checked_file(head_record),map_location='cpu',weights_only=True)
    model.head.load_state_dict(saved['head'],strict=True)
    initial=digest(model.head);optimizer=torch.optim.Adam(model.head.parameters(),lr=.003)
    ids=data.train[np.random.RandomState(257000+model.seed).permutation(35000)[:3500]]
    bound(out/'cache_binding.json',dict(encoder_sha256=digest(model.encoder),core_sha256=model.frozen_initial_sha256,
        ids=ids.tolist(),training_targets_only=True,region_proposals_use_truth=False,
        source_head=head_record,initial_head_sha256=initial,optimizer_reset=True))
    selected=ids[:2] if admission else ids
    for index,image_id in enumerate(selected):
        path=cache/f'{int(image_id):06d}.pt'
        if not path.exists():
            with torch.no_grad():
                output=model(torch.from_numpy(data.rgb(np.array([image_id]))).to('cuda:0'),
                    inference_seed=255000+int(image_id),live_tail_steps=0)
                features=activity_features(output['trace'])[0].detach().cpu()
                regions=regions_from_output(model,output)
            assert features.shape==(4096,144) and bool(torch.isfinite(features).all())
            r.save(path,dict(features=features,regions=torch.as_tensor(regions).cpu(),image_id=int(image_id)))
            del output,features
        c.write(out/'progress.json',dict(stage='detached_cache',completed=index+1,total=len(selected)))
    bound(out/'cache_manifest.json',{p.name:c.sha(p) for p in sorted(cache.glob('*.pt'))})
    # Load the 8GB feature cache once. Re-reading it for every epoch would waste I/O.
    rows=[torch.load(cache/f'{int(i):06d}.pt',map_location='cpu',weights_only=True) for i in selected]
    labels=data.labels(selected);first=offset=updates=0;coverage={n:0. for n,_ in model.head.named_parameters()}
    checkpoint=out/'resume.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
        assert saved['binding_sha256']==c.sha(out.parent/'binding.json')
        assert saved['encoder_sha256']==digest(model.encoder)
        model.head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
        first=saved['epoch'];offset=saved['offset'];updates=saved['updates'];coverage=saved['coverage']
        assert first*len(selected)+offset==updates
    def snapshot(epoch,end):
        r.save(checkpoint,dict(head=state(model.head),optimizer=optimizer.state_dict(),epoch=epoch,offset=end,
            updates=updates,coverage=coverage,encoder_sha256=digest(model.encoder),binding_sha256=c.sha(out.parent/'binding.json')))
    for epoch in range(first,1 if admission else 10):
        order=np.random.RandomState(251000+model.seed+10000*epoch).permutation(len(selected))
        for position in range(offset if epoch==first else 0,len(order)):
            j=int(order[position]);row=rows[j];optimizer.zero_grad(set_to_none=True)
            features=row['features'].to('cuda:0');regions=row['regions'].numpy()
            loss=objective(model.head(features,regions),labels[j]);loss.backward()
            for name,p in model.head.named_parameters():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
                coverage[name]+=float(p.grad.abs().sum())
            assert all(p.grad is None for p in model.encoder.parameters())
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1
            end=position+1;ne=epoch+int(end==len(order));no=0 if end==len(order) else end
            with (out/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=updates,epoch=epoch,image_id=int(selected[j]),loss=float(loss.detach())))+'\n')
            if updates%128==0 or updates==1 or no==0 or admission:snapshot(ne,no)
            c.write(out/'progress.json',dict(stage='head_train',epoch=epoch+1,updates=updates,total_updates=2 if admission else 35000))
    assert updates==(2 if admission else 35000)
    assert all(v>0 for v in coverage.values()),coverage
    assert digest(model.head)!=initial
    model.assert_frozen()
    saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
    before=digest(model.head);model.head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
    assert digest(model.head)==before and {int(s['step'].item()) for s in optimizer.state.values()}=={updates}
    result=dict(status='detached_head_complete',updates=updates,unique_train=len(selected),epochs=1 if admission else 10,
        diagnostic_only=admission,checkpoint_sha256=c.sha(checkpoint),head_sha256=before,gradient_coverage=coverage,
        encoder_core_gradients_absent=True,source_head=head_record,optimizer_reset_for_additional_training=True,
        optimizer_restore_verified=True,cache_manifest_sha256=c.sha(out/'cache_manifest.json'))
    c.write(out/'complete.json',result);return result


def run(dataset_name,variant,seed,root,source_inventory,head_inventory,*,admission=False):
    settings();c.immutable();root=Path(root);data=Dataset(dataset_name,root)
    out=root/('admission' if admission else 'jobs')/f'model{variant}'/dataset_name/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True);lock=(out/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    head_record=json.loads(Path(head_inventory).read_bytes())['datasets'][dataset_name][str(seed)] if variant==252 else None
    binding=dict(experiment='SW0257',variant=variant,seed=seed,dataset=data.binding,code=sources(),
        unique_encoder_TRAIN=35000,encoder_epochs=10,physical_encoder_batch=16,last_batch=8,
        encoder_loss='direct_gamma_frequency_teacher' if variant==252 else 'CNN_RGB_reconstruction_MSE',
        core_backward=False,core_weight_updates=False,projection_updates=False,normalization_refit=False,
        head_train=3500 if variant==252 else 0,head_epochs=10 if variant==252 else 0,
        source_inventory_sha256=c.sha(source_inventory),head_source=head_record,
        protected_benchmark=c.immutable(),diagnostic_only=admission)
    bound(out/'binding.json',binding)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_bytes())
    model=FrozenCoreModel(variant,seed,sources=source_inventory).to('cuda:0')
    if (out/'encoder/complete.json').exists():
        encoder=json.loads((out/'encoder/complete.json').read_bytes())
        assert c.sha(out/'encoder/resume.pt')==encoder['checkpoint_sha256']
        saved=torch.load(out/'encoder/resume.pt',map_location='cpu',weights_only=True)
        model.encoder.load_state_dict(saved['encoder']);assert digest(model.encoder)==encoder['encoder_sha256']
    else:encoder=train_encoder(model,data,out/'encoder',admission=admission)
    model.encoder.requires_grad_(False)
    for p in model.encoder.parameters():p.grad=None
    if variant==252:
        if (out/'head/complete.json').exists():
            head=json.loads((out/'head/complete.json').read_bytes());assert c.sha(out/'head/resume.pt')==head['checkpoint_sha256']
            model.head.load_state_dict(torch.load(out/'head/resume.pt',map_location='cpu',weights_only=True)['head'])
        else:head=train_head(model,data,out/'head',head_record,admission=admission)
    from full_benchmark.evaluation import evaluate
    result=evaluate(model.eval(),data,out/'evaluation',count=1 if admission else None)
    model.assert_frozen()
    done=dict(status='admission_passed' if admission else 'training_and_evaluation_complete',
        diagnostic_only=admission,binding_sha256=c.sha(out/'binding.json'),
        encoder_complete_sha256=c.sha(out/'encoder/complete.json'),evaluation_sha256=c.sha(out/'evaluation/result.json'),
        head_complete_sha256=c.sha(out/'head/complete.json') if variant==252 else None,
        count=result['count'],metrics=result['pixel'],variant=variant,dataset=dataset_name,seed=seed)
    c.write(out/'complete.json',done);return done
