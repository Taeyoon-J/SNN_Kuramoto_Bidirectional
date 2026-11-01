import fcntl,json,os,time
from pathlib import Path
import numpy as np
import torch
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from frozen_core_benchmark.data import Dataset
from frozen_core_benchmark.model import FrozenCoreModel,digest,checked_file
from separate_encoder_benchmark.engine import settings,state,bound,encoder_parts,encoder_loss,train_head,sources as old_sources


def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'run_selective3500.py']
    return {**old_sources(),**{str(p):c.sha(p) for p in paths}}


def training_ids(data,seed):return data.train[np.random.RandomState(257000+seed).permutation(35000)[:3500]]


def fresh_encoder(model):
    from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
    from snn_kuramoto_bidirectional.sw240.context import ContextualFeatureEncoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(258000+model.seed);base=CNNFeatureEncoder(8,3,3)
        model.encoder=(base if model.variant==253 else ContextualFeatureEncoder(base)).to('cuda:0')
    model.encoder.requires_grad_(True)


def fit_encoder(model,data,out,*,admission=False):
    out.mkdir(parents=True,exist_ok=True);teacher,decoder,optimizer=encoder_parts(model)
    initial=digest(model.encoder);params=list(model.encoder.parameters())+(list(decoder.parameters()) if decoder is not None else [])
    coverage={n:0. for n,_ in model.encoder.named_parameters()};seen=updates=first=offset=0
    checkpoint=out/'resume.pt';binding_sha=c.sha(out.parent/'binding.json')
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True);assert saved['binding_sha256']==binding_sha
        model.encoder.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer'])
        if decoder is not None:decoder.load_state_dict(saved['decoder'])
        first=saved['epoch'];offset=saved['offset'];seen=saved['images_seen'];updates=saved['updates'];coverage=saved['coverage']
        assert seen==3500*first+offset
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    def save(epoch,end):
        model.assert_frozen();r.save(checkpoint,dict(encoder=state(model.encoder),decoder=state(decoder) if decoder is not None else None,
            optimizer=optimizer.state_dict(),epoch=epoch,offset=end,images_seen=seen,updates=updates,coverage=coverage,
            binding_sha256=binding_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        c.write(out/'committed.json',dict(images_seen=seen,updates=updates,epoch=epoch,offset=end,sha256=c.sha(checkpoint)))
    selected=training_ids(data,model.seed);core=model.wrapped.core;original=core.kuramoto.forward
    def forbidden(*a,**kw):raise AssertionError('independent encoder fit invoked recurrent core')
    core.kuramoto.forward=forbidden;started=time.time()
    try:
        for epoch in range(first,1 if admission else 10):
            order=selected[np.random.RandomState(249000+model.seed+10000*epoch).permutation(3500)]
            for start in range(offset if epoch==first else 0,3500,16):
                ids=order[start:start+16];optimizer.zero_grad(set_to_none=True)
                loss=encoder_loss(model,teacher,decoder,data,ids);loss.backward()
                for name,p in model.encoder.named_parameters():
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
                    coverage[name]+=float(p.grad.abs().sum())
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                seen+=len(ids);updates+=1;end=start+len(ids);ne=epoch+int(end==3500);no=0 if end==3500 else end
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(update=updates,epoch=epoch,ids=ids.tolist(),images_seen=seen,loss=float(loss.detach()),gradient_norm=float(norm)))+'\n')
                if updates==1 or updates%64==0 or no==0 or admission:save(ne,no)
                c.write(out/'progress.json',dict(stage='scratch_encoder3500x10',epoch=epoch+1,updates=updates,images_seen=seen))
                if admission and updates==2:break
            if admission:break
    finally:core.kuramoto.forward=original
    assert (seen,updates)==((32,2) if admission else (35000,2190))
    assert all(v>0 for v in coverage.values()) and digest(model.encoder)!=initial
    assert not data.labels_opened if model.variant==253 else data.labels_opened
    model.assert_frozen();saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
    before=digest(model.encoder);model.encoder.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer'])
    assert digest(model.encoder)==before and {int(v['step'].item()) for v in optimizer.state.values()}=={updates}
    report=dict(status='scratch_encoder_complete',reused=False,diagnostic_only=admission,unique_TRAIN=3500,
        epochs=1 if admission else 10,images_seen=seen,updates=updates,gradient_coverage=coverage,
        encoder_sha256=before,checkpoint_sha256=c.sha(checkpoint),core_fixed=True,core_backward=False,
        core_sha256=model.frozen_initial_sha256,optimizer_restore_verified=True,mask_training=model.variant==252,seconds=time.time()-started)
    c.write(out/'complete.json',report);return report


def import_cache(model,data,out,source,admission):
    if source is None:return 0
    folder=Path(source);path=folder/'cache_binding.json'
    if not path.exists():return 0
    binding=json.loads(path.read_bytes())
    if binding['encoder_sha256']!=digest(model.encoder) or binding['core_sha256']!=model.frozen_initial_sha256:return 0
    old=json.loads((folder.parent/'binding.json').read_bytes());new=json.loads((out.parent/'binding.json').read_bytes())
    assert old['source_inventory_sha256']==new['source_inventory_sha256'] and old['dataset']==new['dataset']
    def canonical(code):return {p.split('/benchmark_codes/',1)[1]:v for p,v in code.items()}
    previous=canonical(old['code']);current=canonical(sources())
    assert all(current.get(k)==v for k,v in previous.items()),'cache operator source changed'
    ids=training_ids(data,model.seed);assert binding['ids']==ids.tolist()
    target=out/'cache';target.mkdir(parents=True,exist_ok=True);count=0
    for image_id in ids[:2] if admission else ids:
        old=folder/'cache'/f'{int(image_id):06d}.pt';new=target/old.name
        if old.exists():
            if new.exists():assert c.sha(old)==c.sha(new)
            else:os.link(old,new)
            count+=1
    c.write(out/'imported_cache.json',dict(source=str(folder),images=count,encoder_core_and_ids_verified=True,
        GT_used_for_features=False,head_weights_do_not_participate_in_feature_generation=True))
    return count


def run(dataset,variant,seed,root,source_inventory,selection,*,admission=False):
    settings();c.immutable();root=Path(root);data=Dataset(dataset,root)
    choice=json.loads(Path(selection).read_bytes())['cases'][f'{variant}/{dataset}/{seed}']
    out=root/('admission' if admission else 'jobs')/f'model{variant}'/dataset/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True);lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=dict(experiment='SW0258',variant=variant,dataset=data.binding,seed=seed,selection=choice,code=sources(),
        fresh_encoder_unique=3500,fresh_encoder_epochs=10,fresh_encoder_batch=16,last_batch=12,
        head_unique=3500 if variant==252 else 0,head_epochs=10 if variant==252 else 0,
        reuse_floor_unique=3500,reuse_floor_exposures=35000,core_updates=False,core_backward=False,
        diagnostic_only=admission,benchmark=c.immutable(),source_inventory_sha256=c.sha(source_inventory))
    bound(out/'binding.json',binding)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_bytes())
    model=FrozenCoreModel(variant,seed,sources=source_inventory).to('cuda:0')
    enc=choice['encoder']
    if enc is not None:
        assert enc['unique_train']>=3500 and enc['exposures']>=35000
        saved=torch.load(checked_file(enc),map_location='cpu',weights_only=True)
        model.encoder.load_state_dict(saved['encoder'],strict=True)
        if 'feature_mean' in saved:model.feature_mean.copy_(saved['feature_mean']);model.feature_std.copy_(saved['feature_std'])
        report=dict(status='qualified_encoder_reused',reused=True,source=enc,additional_images_seen=0,
            encoder_sha256=digest(model.encoder),core_fixed=True,diagnostic_only=admission)
        c.write(out/'encoder/complete.json',report)
    else:
        fresh_encoder(model)
        if (out/'encoder/complete.json').exists():
            report=json.loads((out/'encoder/complete.json').read_bytes());assert report['checkpoint_sha256']==c.sha(out/'encoder/resume.pt')
            model.encoder.load_state_dict(torch.load(out/'encoder/resume.pt',map_location='cpu',weights_only=True)['encoder'])
        else:report=fit_encoder(model,data,out/'encoder',admission=admission)
    model.encoder.requires_grad_(False)
    for p in model.encoder.parameters():p.grad=None
    head=None
    if model.head is not None:
        if choice['head_reused']:
            record=choice['head'];assert record['unique_train']>=3500 and record['exposures']>=35000
            model.head.load_state_dict(torch.load(checked_file(record),map_location='cpu',weights_only=True)['head'])
            head=dict(status='qualified_head_reused',reused=True,source=record,additional_images_seen=0,head_sha256=digest(model.head))
            c.write(out/'head/complete.json',head)
        elif (out/'head/complete.json').exists():
            head=json.loads((out/'head/complete.json').read_bytes());assert head['checkpoint_sha256']==c.sha(out/'head/resume.pt')
            model.head.load_state_dict(torch.load(out/'head/resume.pt',map_location='cpu',weights_only=True)['head'])
        else:
            import_cache(model,data,out/'head',choice.get('cache_source'),admission)
            head=train_head(model,data,out/'head',choice['head'],admission=admission)
    from full_benchmark.evaluation import evaluate
    result=evaluate(model.eval(),data,out/'evaluation',count=1 if admission else None);model.assert_frozen()
    done=dict(status='admission_passed' if admission else 'training_and_evaluation_complete',diagnostic_only=admission,
        variant=variant,dataset=dataset,seed=seed,binding_sha256=c.sha(out/'binding.json'),
        encoder_report_sha256=c.sha(out/'encoder/complete.json'),head_report_sha256=c.sha(out/'head/complete.json') if head else None,
        evaluation_sha256=c.sha(out/'evaluation/result.json'),metrics=result['pixel'],evaluation_count=result['count'],
        core_initial_and_final_sha256=model.frozen_initial_sha256,encoder_reused=enc is not None,head_reused=choice['head_reused'])
    c.write(out/'complete.json',done);return done
