"""One registered zero-shot or 6,000x10 fine-tune + TEST1,000 job."""
import argparse,copy,hashlib,io,json,os,time
from pathlib import Path
import numpy as np
import torch
from snn_kuramoto_bidirectional import load_model
from snn_kuramoto_bidirectional.sw240.model import model_sha256
from . import common as c

def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:torch.save(value,stream)
    os.replace(temp,path)

def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def train(archive,folder,binding,device):
    source=c.EXPORTS/binding['source']['file'];terminal=folder/'trained.pt'
    if (folder/'training_report.json').exists():
        report=json.loads((folder/'training_report.json').read_bytes())
        assert report['binding']==binding and c.sha(terminal)==report['checkpoint_sha256']
        return terminal,report
    model=load_model(source,device=device);teacher=load_model(source,teacher=True,device=device);teacher.requires_grad_(False)
    optimizer=model.optimizer_for_training();params=[p for group in optimizer.param_groups for p in group['params']]
    frozen=c.frozen_sha(model);images=c.rgb(archive,binding['dataset'],'train')
    history=[];boundary=folder/'resume.pt'
    if boundary.exists():
        state=torch.load(boundary,map_location=device,weights_only=True)
        assert state['binding']==binding and state['frozen_sha256']==frozen
        model.encoder.load_state_dict(state['encoder'],strict=True)
        model.wrapped.core.gamma_channel_proj.load_state_dict(state['projection'],strict=True)
        optimizer.load_state_dict(state['optimizer']);history=state['history']
        assert state['completed_updates']==len(history)<=c.UPDATES
    first=len(history);started=time.perf_counter()
    for step in range(first,c.UPDATES):
        epoch,offset=divmod(step,c.TRAIN//c.BATCH)
        ids=c.epoch_order(binding['seed'],epoch)[offset*c.BATCH:(offset+1)*c.BATCH]
        x=np.asarray(images[ids]).copy();labels=c.training_labels(binding['dataset'],ids)
        optimizer.zero_grad(set_to_none=True);loss,parts=model.supervised_loss(x,labels,teacher)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite training loss/gradient')
        optimizer.step();history.append(dict(update=step+1,epoch=epoch,ids=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(norm)))
        if len(history)%50==0 or len(history)==c.UPDATES:
            save(boundary,dict(binding=binding,frozen_sha256=frozen,encoder=model.encoder.state_dict(),
                projection=model.wrapped.core.gamma_channel_proj.state_dict(),optimizer=optimizer.state_dict(),
                completed_updates=len(history),history=history))
            c.write(folder/'progress.json',dict(stage='training',updates=len(history),total_updates=c.UPDATES))
    for epoch in range(c.EPOCHS):
        rows=history[epoch*375:(epoch+1)*375]
        actual=[i for row in rows for i in row['ids']]
        assert actual==c.epoch_order(binding['seed'],epoch).tolist() and len(set(actual))==6000
    assert c.frozen_sha(model)==frozen
    assert {int(v['step'].item()) for v in optimizer.state.values()}=={c.UPDATES}
    bundle=torch.load(source,map_location='cpu',weights_only=True)
    bundle['config']=dict(bundle['config'],endpoint='SW0250_'+binding['dataset']+'_TRAIN6000x10',model_sha256=model_sha256(model))
    bundle['wrapped_state']={k:v.detach().cpu().clone() for k,v in model.wrapped.state_dict().items()}
    bundle['encoder_state']={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    bundle['benchmark_training']=dict(binding_sha256=digest_json(binding),updates=c.UPDATES,unique_train_images=6000,epochs=10)
    save(terminal,bundle)
    report=dict(status='completed_TRAIN6000x10',binding=binding,updates=c.UPDATES,exposures=60000,
        unique_train_images=6000,epochs=10,final_model_sha256=bundle['config']['model_sha256'],
        checkpoint_sha256=c.sha(terminal),frozen_core_sha256=frozen,active_seconds_this_attempt=time.perf_counter()-started,
        validation_or_test_labels_used_for_training=False,checkpoint_selection='fixed_terminal_epoch10')
    c.write(folder/'training_report.json',report)
    del model,teacher,optimizer,bundle;torch.cuda.empty_cache()
    return terminal,report

def predict(archive,folder,binding,path,device):
    # Mandatory original preflight is run on every invocation, even a resumed evaluation.
    preflight=c.preflight(binding['dataset'],folder/'preflight')
    model=load_model(path,device=device);model.requires_grad_(False)
    initial=model_sha256(model);checkpoint_sha=c.sha(path)
    evaluation=dict(job_binding_sha256=digest_json(binding),checkpoint_sha256=checkpoint_sha,
        model_sha256=initial,grid=c.DATASETS[binding['dataset']],split='test',ids=c.TEST_IDS,
        inference_seed_base=193000,preprocessing='protected make128 nearest')
    fingerprint=digest_json(evaluation);images=c.rgb(archive,binding['dataset'],'test')
    rows=[]
    for i,image_id in enumerate(c.TEST_IDS):
        data=folder/'rows'/f'{i:04d}.npz';meta=data.with_suffix('.json');data.parent.mkdir(exist_ok=True)
        x=np.asarray(images[i:i+1]).copy();rgb_sha=hashlib.sha256(x[0].tobytes()).hexdigest()
        expected=dict(evaluation_sha256=fingerprint,index=i,image_id=image_id,inference_seed=193000+i,RGB_sha256=rgb_sha)
        if meta.exists():
            record=json.loads(meta.read_bytes());assert all(record[k]==v for k,v in expected.items())
            assert c.sha(data)==record['data_sha256']
            with np.load(data,allow_pickle=False) as saved:labels=saved['patch_labels'].copy()
        else:
            native=model.predict(x,inference_seeds=[193000+i])[0]
            labels=c.prediction_grid(native,binding['dataset'])
            temp=data.with_name(data.name+'.tmp')
            with temp.open('wb') as stream:np.savez_compressed(stream,patch_labels=labels,native64_labels=native)
            os.replace(temp,data);c.write(meta,dict(expected,data_sha256=c.sha(data)))
        assert labels.shape==(evaluation['grid'],evaluation['grid']) and labels.dtype==np.int64 and np.all(labels>=0)
        rows.append(labels)
        if i%10==0 or i==999:c.write(folder/'progress.json',dict(stage='test_prediction',rows=i+1,total_rows=1000))
    assert model_sha256(model)==initial
    labels=np.stack(rows)
    protocol=dict(evaluation=evaluation,prediction_array_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        count=1000,preflight=preflight,test_GT_used_to_form_predictions=False,benchmark_sha256=c.immutable())
    if (folder/'prediction_protocol.json').exists():
        old=json.loads((folder/'prediction_protocol.json').read_bytes())
        assert old['evaluation']==evaluation and old['prediction_array_sha256']==protocol['prediction_array_sha256']
    else:c.write(folder/'prediction_protocol.json',protocol)
    del model;torch.cuda.empty_cache()
    return labels,protocol

def mbo(prediction,target):
    result=[]
    for pred,truth in zip(prediction,target):
        gg=torch.unique(truth);gg=gg[gg!=0];pp=torch.unique(pred);pp=pp[pp!=0]
        if not len(gg):result.append(float('nan'));continue
        if not len(pp):result.append(0.);continue
        a=(truth.flatten()[None]==gg[:,None]).double();b=(pred.flatten()[None]==pp[:,None]).double()
        overlap=a@b.T;iou=overlap/(a.sum(1)[:,None]+b.sum(1)[None]-overlap)
        result.append(float(iou.max(1).values.mean()))
    return np.asarray(result)

def score(folder,binding,labels,protocol):
    # The complete prediction protocol must exist before TEST pixel truth is inspected.
    assert (folder/'prediction_protocol.json').exists();c.check_inputs(binding)
    ds=binding['dataset'];grid=c.DATASETS[ds]
    manifest=json.loads((c.DATA/f'split_manifest_{ds}.json').read_bytes())
    targets=torch.load(c.DATA/f'targets_{ds}_grid{grid}.pt',map_location='cpu',weights_only=True,mmap=True)
    assert targets['names'][7000:8000]==manifest['splits']['test']
    pixels=torch.load(c.DATA/f'pixel_labels_{ds}.pt',map_location='cpu',weights_only=True,mmap=True)[7000:8000]
    cells=c.benchmark_module('make_targets_grid');metric=c.benchmark_module('evaluation')
    oracle,_,_,_,_=cells.mask_cells(pixels,grid)
    assert torch.equal(oracle,targets['patch_labels'][7000:8000])
    row,_=cells.cell_index(pixels.shape[1],grid,'cpu');col,_=cells.cell_index(pixels.shape[2],grid,'cpu')
    pred=torch.from_numpy(labels);expanded=pred[:,row][:,:,col].contiguous()
    ceiling=oracle[:,row][:,:,col].contiguous()
    scored=metric.evaluate_patch_masks(expanded,pixels);patch=metric.evaluate_patch_masks(pred,oracle)
    ideal=metric.evaluate_patch_masks(ceiling,pixels)
    best=mbo(expanded,pixels)
    values={k:v.numpy() for k,v in scored['per_image'].items()};values['mbo']=best
    values['pred_fg']=(expanded!=0).double().mean((1,2)).numpy()
    values['true_fg']=(pixels!=0).double().mean((1,2)).numpy()
    means={k:float(np.nanmean(v)) for k,v in values.items()}
    raw=folder/'per_image_scores.npz'
    with raw.open('wb') as stream:np.savez_compressed(stream,**values,image_ids=np.asarray(c.TEST_IDS))
    result=dict(status='complete_verified_test1000',dataset=ds,seed=binding['seed'],mode=binding['mode'],
        binding_sha256=digest_json(binding),prediction_protocol_sha256=c.sha(folder/'prediction_protocol.json'),
        count=1000,grid=grid,native_model_grid=64,metrics=means,
        valid_count={k:int(v) for k,v in scored['valid_count'].items()},
        patch_metrics={k:float(v) for k,v in patch['mean'].items()},
        pixel_grid_ceiling={k:float(v) for k,v in ideal['mean'].items()},
        per_image_scores_sha256=c.sha(raw),benchmark_files=c.immutable(),
        known_cluster_count_supplied=False,readout='actual spikes and unchanged signed connected partition')
    c.write(folder/'result.json',result)
    return result

def run(archive,ds,seed,mode):
    torch.set_num_threads(1);c.immutable()
    archive=Path(archive);folder=archive/'jobs'/ds/mode/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    binding=c.job_binding(archive,ds,seed,mode)
    if (folder/'binding.json').exists():assert json.loads((folder/'binding.json').read_bytes())==binding
    else:c.write(folder/'binding.json',binding)
    source=c.EXPORTS/binding['source']['file'];assert c.sha(source)==binding['source']['sha256']
    c.check_inputs(binding)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    path,_=train(archive,folder,binding,device) if mode=='finetune' else (source,None)
    labels,protocol=predict(archive,folder,binding,path,device)
    result=score(folder,binding,labels,protocol)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--dataset',choices=tuple(c.DATASETS),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--mode',choices=('zero_shot','finetune'),required=True)
    args=p.parse_args();run(args.archive,args.dataset,args.seed,args.mode)
