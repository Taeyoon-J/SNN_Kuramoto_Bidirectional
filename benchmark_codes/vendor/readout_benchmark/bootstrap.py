"""Persistent resource-aware queue with the original-data FG-ARI promotion gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from mod_benchmark import common as c
from . import runtime as r


def gpu_free():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=15)
    if result.returncode:raise RuntimeError('GPU query unavailable; preserve running workers')
    return {int(a):int(b) for a,b in (line.split(',') for line in result.stdout.splitlines())}


def run_wave(archive,name,jobs):
    output=archive/'queue'/name;output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    c.write(archive/'status.json',dict(stage=name,status='running_or_queued',job_count=len(jobs),updated_unix=time.time()))
    states=[dict(arguments=job,stage='queued') for job in jobs];running={};handles={}
    if (output/'state.json').exists():
        old=json.loads((output/'state.json').read_bytes())
        assert [v['arguments'] for v in old]==jobs
        for item in old:
            if item['stage']=='running':
                try:os.kill(item['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('previous registered worker still alive; do not duplicate')
        states=old
        for item in states:
            if item['stage']=='running':item['stage']='queued'
            if item['stage']=='failed':raise RuntimeError('inspect failed worker before retrying')
    while any(v['stage']!='complete' for v in states):
        for index,process in list(running.items()):
            code=process.poll()
            if code is None:continue
            states[index]['stage']='complete' if code==0 else 'failed';states[index]['exit_code']=code
            for handle in handles.pop(index):handle.close()
            del running[index]
            c.write(output/'state.json',states)
            if code:
                c.write(archive/'attention.json',dict(stage=name,index=index,job=states[index],log=str(output/f'{index:02d}.stderr')))
                raise RuntimeError('worker failed; other already-running workers remain alive')
        busy={states[i]['physical_gpu'] for i in running}
        try:free=gpu_free()
        except (subprocess.TimeoutExpired,RuntimeError):
            time.sleep(30);continue
        for gpu,available in sorted(free.items(),key=lambda v:-v[1]):
            if gpu in busy or available<8192:continue
            pending=next((i for i,v in enumerate(states) if v['stage']=='queued'),None)
            if pending is None:break
            arguments=[]
            for key,value in jobs[pending].items():
                arguments.extend(['--'+key.replace('_','-'),str(value)])
            command=[sys.executable,'-u','-B','-m','readout_benchmark.runtime','--archive',str(archive),*arguments]
            out=(output/f'{pending:02d}.stdout').open('ab');err=(output/f'{pending:02d}.stderr').open('ab')
            process=subprocess.Popen(command,cwd=c.CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            states[pending].update(stage='running',pid=process.pid,physical_gpu=gpu,started_unix=time.time())
            running[pending]=process;handles[pending]=(out,err);busy.add(gpu)
            c.write(output/'state.json',states)
        if running or any(v['stage']=='queued' for v in states):time.sleep(20)
    c.write(output/'complete.json',dict(status='all_jobs_complete',jobs=states))


def cache_jobs(dataset,mode,splits):
    return [dict(task='cache',dataset=dataset,mode=mode,seed=s,split=split,count=count)
            for s in range(3) for split,count in splits]


def head_jobs(dataset,mode,count,validation_count,arms,archive=None,initial=False):
    jobs=[]
    for s in range(3):
        for arm in arms:
            job=dict(task='train',dataset=dataset,mode=mode,seed=s,count=count,validation_count=validation_count,arm=arm)
            if initial:
                job['initial']=str(archive/'heads'/'clevr'/'zero_shot'/f'seed{s}'/f'{arm}_TRAIN256'/'best.pt')
            jobs.append(job)
    return jobs


def head_path(archive,dataset,mode,seed,arm,count=256):
    return archive/'heads'/dataset/mode/f'seed{seed}'/f'{arm}_TRAIN{count}'/'best.pt'


def score_validation(archive,count,validation_count):
    path=archive/f'validation_TRAIN{count}.json'
    if path.exists():return json.loads(path.read_bytes())
    ids=r.ids_for('clevr','validation',validation_count);truth=r.truth('clevr','validation',ids)
    scores={arm:[] for arm in ('baseline','keep_groups','all_support','learned_background','residual_mask')}
    for seed in range(3):
        folder=r.row_folder(archive,'clevr','zero_shot',seed)
        for arm in scores:
            head=r.load_head(head_path(archive,'clevr','zero_shot',seed,arm,count),'cpu') if arm.startswith(('learned','residual')) else None
            score,_=r.evaluate_rows(folder,ids,truth,head,'cpu',arm=arm)
            scores[arm].append(score)
    means={arm:{k:float(np.mean([v[k] for v in rows])) for k in r.KEYS} for arm,rows in scores.items()}
    result=dict(status='validation_only_comparison',count=validation_count,TRAIN_count=count,per_seed=scores,means=means,
                selection_metric='fg_ari',fixed320_used_for_selection=False)
    c.write(path,result);return result


def original_confirmation(archive,arm):
    output=archive/'original_confirmation';output.mkdir(exist_ok=True)
    if (output/'result.json').exists():return json.loads((output/'result.json').read_bytes())
    ids=r.ids_for('clevr','fixed320',320);all_predictions=[];baselines=[];checks=[]
    # All masks are frozen before this function opens fixed320 truth.
    for seed in range(3):
        folder=r.row_folder(archive,'clevr','zero_shot',seed);head=r.load_head(head_path(archive,'clevr','zero_shot',seed,arm),'cpu')
        pred=[];base=[]
        old=r.ROOT/'collaborative_testing/generated/SW0249_four_model_35000/model240'/f'seed{seed}'
        for index,image_id in enumerate(ids):
            row=r.read_row(folder,image_id,'cpu');pred.append(head.predict(row['features'],row['baseline'],row['all_support']))
            base.append(row['baseline'])
            with np.load(old/'rows'/f'{index:04d}'/'prediction.npz',allow_pickle=False) as saved:
                assert np.array_equal(row['baseline'],saved['labels']), 'original 240 production mask differs'
        all_predictions.append(np.stack(pred));baselines.append(np.stack(base))
        file=output/f'predictions_seed{seed}.npz'
        with file.open('wb') as stream:np.savez_compressed(stream,predictions=all_predictions[-1],baseline=baselines[-1],ids=ids)
        checks.append(dict(seed=seed,file_sha256=c.sha(file),head_sha256=c.sha(head_path(archive,'clevr','zero_shot',seed,arm)),
                           original_240_masks_exactly_equal=True,GT_used_for_prediction=False))
    c.write(output/'prediction_protocol.json',dict(arm=arm,checks=checks,ids=ids.tolist(),GT_read=False))
    truth=r.truth('clevr','fixed320',ids);metric=c.benchmark_module('evaluation');vectors=[];per_seed=[]
    for seed in range(3):
        pair=[]
        for p in (baselines[seed],all_predictions[seed]):
            scored=metric.evaluate_patch_masks(torch.from_numpy(p),torch.from_numpy(truth))
            pair.append(np.stack([scored['per_image'][k].numpy() for k in r.KEYS],1))
        vectors.append(pair[1]-pair[0]);per_seed.append(dict(seed=seed,baseline=dict(zip(r.KEYS,pair[0].mean(0).tolist())),
                                                   modified=dict(zip(r.KEYS,pair[1].mean(0).tolist()))))
    delta=np.stack(vectors).mean(0);draw=np.random.RandomState(251).randint(0,320,(10000,320))
    ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
    change=float(delta[:,0].mean());passed=change>=-.005
    result=dict(status='original_data_gate_passed' if passed else 'original_data_gate_failed',arm=arm,per_seed=per_seed,
        mean_baseline={k:float(np.mean([v['baseline'][k] for v in per_seed])) for k in r.KEYS},
        mean_modified={k:float(np.mean([v['modified'][k] for v in per_seed])) for k in r.KEYS},
        FGARI_change=change,allowed_absolute_decrease=.005,promotion_passed=passed,
        paired_image_CI95={k:ci[:,j].tolist() for j,k in enumerate(r.KEYS)},
        scope='reused fixed320 validation, not untouched test',prediction_protocol_sha256=c.sha(output/'prediction_protocol.json'))
    c.write(output/'result.json',result);return result


def score_new(archive,arm,count):
    report={}
    for ds in c.DATASETS:
        for mode in ('zero_shot','finetune'):
            rows=[]
            for seed in range(3):
                folder=r.row_folder(archive,ds,mode,seed)
                hp=head_path(archive,'clevr','zero_shot',seed,arm) if mode=='zero_shot' else head_path(archive,ds,mode,seed,arm)
                head=r.load_head(hp,'cpu');ids=r.ids_for(ds,'test',count);native=[]
                for image_id in ids:
                    row=r.read_row(folder,image_id,'cpu');native.append(head.predict(row['features'],row['baseline'],row['all_support']))
                predictions=np.stack([c.prediction_grid(p,ds) for p in native])
                out=archive/'results'/ds/mode/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
                path=out/f'predictions_{count}.npz'
                with path.open('wb') as stream:np.savez_compressed(stream,predictions=predictions,native64=np.stack(native),ids=ids)
                c.write(out/f'protocol_{count}.json',dict(head_sha256=c.sha(hp),prediction_sha256=c.sha(path),GT_used_for_prediction=False,
                                                       source_binding=json.loads((folder/'binding.json').read_bytes())))
                c.preflight(ds,out/f'preflight_{count}')
                # Only after the prediction protocol, open pixel truth and original grid expansion.
                truth=torch.load(c.DATA/f'pixel_labels_{ds}.pt',map_location='cpu',weights_only=True,mmap=True)[torch.as_tensor(ids)]
                cells=c.benchmark_module('make_targets_grid');metric=c.benchmark_module('evaluation')
                iy,_=cells.cell_index(truth.shape[1],c.DATASETS[ds],'cpu');ix,_=cells.cell_index(truth.shape[2],c.DATASETS[ds],'cpu')
                pred=torch.from_numpy(predictions)[:,iy][:,:,ix]
                scored=metric.evaluate_patch_masks(pred,truth)
                vectors={k:v.numpy() for k,v in scored['per_image'].items()}
                with (out/f'scores_{count}.npz').open('wb') as stream:np.savez_compressed(stream,**vectors,ids=ids)
                result=dict(seed=seed,count=count,metrics={k:float(np.nanmean(v)) for k,v in vectors.items()},benchmark=c.immutable())
                c.write(out/f'result_{count}.json',result);rows.append(result)
            report[f'{ds}/{mode}']=dict(per_seed=rows,mean={k:float(np.mean([v['metrics'][k] for v in rows])) for k in r.KEYS},
                                       sample_std={k:float(np.std([v['metrics'][k] for v in rows],ddof=1)) for k in r.KEYS})
    c.write(archive/f'benchmark_results_{count}.json',dict(status='all_twelve_scored',count=count,arm=arm,results=report,
        extra_readout_training='finetune adds TRAIN256x20 readout adaptation after verified SW0250 encoder TRAIN6000x10',
        benchmark_unchanged=c.immutable()))


def run(archive):
    torch.set_num_threads(1);archive=Path(archive);archive.mkdir(parents=True,exist_ok=True)
    lease=(archive/'coordinator.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.immutable()
    try:
        for count,val in ((64,16),(256,64)):
            run_wave(archive,f'clevr_cache_{count}',cache_jobs('clevr','zero_shot',[('train',count),('validation',val)]))
            run_wave(archive,f'clevr_heads_{count}',head_jobs('clevr','zero_shot',count,val,('learned_background','residual_mask')))
            comparison=score_validation(archive,count,val)
        arm=max(('learned_background','residual_mask'),key=lambda x:comparison['means'][x]['fg_ari'])
        c.write(archive/'validation_selection.json',dict(arm=arm,comparison_sha256=c.sha(archive/'validation_TRAIN256.json'),selection_before_fixed320=True))
        if comparison['means'][arm]['fg_ari']<comparison['means']['baseline']['fg_ari']-.005:
            c.write(archive/'status.json',dict(status='needs_research_revision',reason='validation regression; no new-dataset tests launched'))
            return
        run_wave(archive,'original_fixed320',cache_jobs('clevr','zero_shot',[('fixed320',320)]))
        original=original_confirmation(archive,arm)
        if not original['promotion_passed']:
            c.write(archive/'status.json',dict(status='needs_research_revision',reason='original fixed320 FGARI gate failed; preserve 240'))
            return
        # Actual dataset fine-tuning checkpoint is reused with verified 3750-step evidence.
        run_wave(archive,'new_finetune_features',[job for ds in c.DATASETS for job in cache_jobs(ds,'finetune',[('train',256),('validation',64)])])
        run_wave(archive,'new_finetune_heads',[job for ds in c.DATASETS for job in head_jobs(ds,'finetune',256,64,(arm,),archive,True)])
        for count in (360,1000):
            run_wave(archive,f'new_test_{count}',[job for ds in c.DATASETS for mode in ('zero_shot','finetune')
                                                for job in cache_jobs(ds,mode,[('test',count)])])
            score_new(archive,arm,count)
        c.write(archive/'status.json',dict(status='all_requested_experiments_scored_pending_independent_audit',arm=arm,updated_unix=time.time()))
    except Exception:
        c.write(archive/'coordinator_error.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);run(parser.parse_args().archive)
