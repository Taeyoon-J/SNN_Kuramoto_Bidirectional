import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from portable_runtime import HERE
from mod_benchmark import common as c
from selective3500_benchmark.engine import sources,training_ids

def jobs():return [dict(variant=v,dataset=d,seed=s) for v in (252,253) for d in ('clevr','tetrominoes','clevrtex_original') for s in (0,1,2)]
def folder(root,job,admission):return root/('admission' if admission else 'jobs')/f"model{job['variant']}"/job['dataset']/f"seed{job['seed']}"
def alive(record):
    p=Path('/proc')/str(record['pid'])
    try:
        parts=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        return parts[0]!='Z' and int(parts[19])==record['start_ticks'] and p.stat().st_uid==os.getuid() and [x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]==record['command']
    except OSError:return False
def verify(out,admission):
    done=json.loads((out/'complete.json').read_bytes());binding=json.loads((out/'binding.json').read_bytes())
    assert binding['code']==sources() and binding['experiment']=='SW0258' and done['diagnostic_only']==admission
    assert not binding['core_updates'] and not binding['core_backward'] and binding['benchmark']==c.immutable()
    assert done['binding_sha256']==c.sha(out/'binding.json') and done['evaluation_sha256']==c.sha(out/'evaluation/result.json')
    enc=json.loads((out/'encoder/complete.json').read_bytes());assert done['encoder_report_sha256']==c.sha(out/'encoder/complete.json')
    if enc['reused']:
        assert enc['source']['unique_train']>=3500 and enc['source']['exposures']>=35000
    else:
        assert enc['images_seen']==(32 if admission else 35000) and enc['updates']==(2 if admission else 2190)
        assert enc['core_fixed'] and not enc['core_backward'] and all(x>0 for x in enc['gradient_coverage'].values())
        assert enc['checkpoint_sha256']==c.sha(out/'encoder/resume.pt') and enc['optimizer_restore_verified']
    if binding['variant']==252:
        head=json.loads((out/'head/complete.json').read_bytes());assert done['head_report_sha256']==c.sha(out/'head/complete.json')
        if done['head_reused']:assert head['source']['unique_train']>=3500 and head['source']['exposures']>=35000
        else:assert head['updates']==(2 if admission else 35000) and head['checkpoint_sha256']==c.sha(out/'head/resume.pt')
    assert done['evaluation_count']==(1 if admission else 320 if done['dataset']=='clevr' else 1000)
    return done
def audit(out,root):
    from frozen_core_benchmark.data import Dataset
    from full_benchmark.evaluation import metrics
    binding=json.loads((out/'binding.json').read_bytes());data=Dataset(binding['dataset']['dataset'],root);seed=binding['seed']
    ids=training_ids(data,seed)
    if binding['selection']['encoder'] is None:
        rows={x['update']:x for x in map(json.loads,(out/'encoder/training.jsonl').read_text().splitlines())}
        assert sorted(rows)==list(range(1,2191));update=seen=0
        for epoch in range(10):
            order=ids[np.random.RandomState(249000+seed+10000*epoch).permutation(3500)]
            for start in range(0,3500,16):
                update+=1;selected=order[start:start+16].tolist();seen+=len(selected)
                assert rows[update]['ids']==selected and rows[update]['epoch']==epoch and rows[update]['images_seen']==seen
        assert seen==35000
    if binding['variant']==252 and not binding['selection']['head_reused']:
        rows={x['update']:x for x in map(json.loads,(out/'head/training.jsonl').read_text().splitlines())}
        assert sorted(rows)==list(range(1,35001))
        for epoch in range(10):
            order=np.random.RandomState(251000+seed+10000*epoch).permutation(3500)
            assert [rows[epoch*3500+j+1]['image_id'] for j in range(3500)]==ids[order].tolist()
    frozen=json.loads((out/'evaluation/frozen_predictions.json').read_bytes());assert frozen['ids']==data.test.tolist()
    assert all(c.sha(out/'evaluation'/name)==sha for name,sha in frozen['hashes'].items())
    predictions=np.stack([np.load(out/'evaluation'/f'{int(i):06d}.npy',allow_pickle=False) for i in data.test])
    score,_=metrics(predictions,data,data.test);original=json.loads((out/'evaluation/result.json').read_bytes())
    assert all(abs(v-original['pixel'][k])<1e-12 for k,v in score['pixel'].items())
    value=dict(variant=binding['variant'],dataset=data.dataset,seed=seed,metrics=score['pixel'],
        complete_sha256=c.sha(out/'complete.json'),training_history_and_prediction_hashes_verified=True,
        protected_metrics_independently_recomputed=True,encoder_reused=binding['selection']['encoder'] is not None,
        head_reused=binding['selection']['head_reused'])
    c.write(out/'audit.json',value);return value
def queue(root,config,source_inventory,selection):
    root=Path(root);lock=(root/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=json.loads((root/'selective_preflight.json').read_bytes());assert pre['code']==sources() and pre['selection_sha256']==c.sha(selection)
    finished={};records={};errors={}
    while True:
        for job in jobs():
            for admission in (True,False):
                out=folder(root,job,admission);key=str(out)
                if (out/'complete.json').exists():
                    if key not in finished:finished[key]=verify(out,admission)
                    records.pop(key,None)
                elif (out/'launch.json').exists() and key not in errors:
                    record=json.loads((out/'launch.json').read_bytes())
                    if alive(record):records[key]=record
                    else:records.pop(key,None);errors[key]=dict(job=job,admission=admission,reason='terminal_without_completion; preserve_and_diagnose')
        free={int(a):int(b) for a,b in (x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().splitlines())}
        # Measured SW0257 peaks <2GiB, same fixed-core/native64 inference here.
        # Reserve8GiB per worker, keep original13GB allocator and max2/GPU.
        for job in jobs():
            admission=str(folder(root,job,True)) not in finished;out=folder(root,job,admission);key=str(out)
            if key in records or key in errors or key in finished:continue
            counts={g:sum(x['gpu']==g for x in records.values()) for g in free}
            choices=[g for g in free if counts[g]<2 and free[g]>=9500]
            if not choices:break
            gpu=min(choices,key=lambda g:(counts[g],-free[g]));out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-B',str(HERE/'run_selective3500.py'),'--output',str(root),'--config',str(config),
                '--sources',str(source_inventory),'--selection',str(selection),'train','--model',str(job['variant']),
                '--dataset',job['dataset'],'--seed',str(job['seed'])]
            if admission:command.append('--admission')
            with (out/'worker.log').open('ab') as stream:
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SW0257_ALLOCATOR_MIB='13000'))
            ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
            record=dict(pid=proc.pid,start_ticks=ticks,command=command,gpu=gpu,created=time.time(),reservation_mib=8000)
            c.write(out/'launch.json',record);records[key]=record;free[gpu]-=8000
        results=[]
        for job in jobs():
            out=folder(root,job,False)
            if str(out) in finished:
                value=json.loads((out/'audit.json').read_bytes()) if (out/'audit.json').exists() else audit(out,root)
                results.append(value)
        groups=[]
        for variant in (252,253):
            for ds in ('clevr','tetrominoes','clevrtex_original'):
                rows=[r for r in results if r['variant']==variant and r['dataset']==ds]
                if len(rows)==3:groups.append(dict(variant=variant,dataset=ds,per_seed=rows,mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}))
        value=dict(status='complete' if len(results)==18 else 'active',required=18,completed_training=sum('/jobs/' in x for x in finished),
            completed_admissions=sum('/admission/' in x for x in finished),audited=len(results),groups=groups,running=list(records.values()),errors=errors,updated=time.time())
        c.write(root/'status.json',value)
        if len(results)==18:return value
        time.sleep(20)
