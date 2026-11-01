"""Persistent four-GPU admission/training queue; no foreign process interruption."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from mod_benchmark import common as c
from portable_runtime import HERE
from separate_encoder_benchmark.engine import sources


def jobs():
    return [dict(model=v,dataset=d,seed=s) for v in (252,253)
        for d in ('clevr','tetrominoes','clevrtex_original') for s in (0,1,2)]


def path(root,job,admission):
    return root/('admission' if admission else 'jobs')/f"model{job['model']}"/job['dataset']/f"seed{job['seed']}"


def live(record):
    p=Path('/proc')/str(record['pid'])
    try:
        stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        return (stat[0]!='Z' and int(stat[19])==record['start_ticks'] and p.stat().st_uid==os.getuid()
            and [x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]==record['command'])
    except (OSError,ValueError):return False


def verify(out,admission):
    done=json.loads((out/'complete.json').read_bytes());binding=json.loads((out/'binding.json').read_bytes())
    assert done['binding_sha256']==c.sha(out/'binding.json') and binding['code']==sources()
    assert binding['protected_benchmark']==c.immutable() and not binding['core_backward']
    assert done['diagnostic_only']==admission and binding['diagnostic_only']==admission
    enc=json.loads((out/'encoder/complete.json').read_bytes())
    assert done['encoder_complete_sha256']==c.sha(out/'encoder/complete.json')
    assert enc['images_seen']==(32 if admission else 350000) and enc['updates']==(2 if admission else 21880)
    assert enc['core_simulation_calls']==0 and enc['core_bitwise_fixed'] and enc['optimizer_restore_verified']
    assert enc['mask_used']==(binding['variant']==252)
    assert all(v>0 for v in enc['gradient_coverage'].values())
    assert enc['checkpoint_sha256']==c.sha(out/'encoder/resume.pt')
    if binding['variant']==252:
        head=json.loads((out/'head/complete.json').read_bytes())
        assert head['updates']==(2 if admission else 35000) and head['unique_train']==(2 if admission else 3500)
        assert head['encoder_core_gradients_absent'] and head['optimizer_restore_verified']
        assert head['checkpoint_sha256']==c.sha(out/'head/resume.pt')
        assert done['head_complete_sha256']==c.sha(out/'head/complete.json')
    assert done['evaluation_sha256']==c.sha(out/'evaluation/result.json')
    expected=1 if admission else 320 if binding['dataset']['dataset']=='clevr' else 1000
    assert done['count']==expected
    return done


def audit(out):
    binding=json.loads((out/'binding.json').read_bytes());seed=binding['seed']
    from frozen_core_benchmark.data import Dataset
    data=Dataset(binding['dataset']['dataset'],out.parents[3])
    checkpoint=json.loads((out/'encoder/complete.json').read_bytes())
    lines=[json.loads(x) for x in (out/'encoder/training.jsonl').read_text().splitlines()]
    # A resumed snapshot can leave uncommitted trailing log entries. Select only
    # the final committed occurrence of each update, and verify every actual ID.
    rows={x['update']:x for x in lines if x['update']<=checkpoint['updates']}
    assert sorted(rows)==list(range(1,21881))
    update=0
    for epoch in range(10):
        ids=data.train[np.random.RandomState(249000+seed+10000*epoch).permutation(35000)]
        for start in range(0,35000,16):
            update+=1;assert rows[update]['epoch']==epoch and rows[update]['ids']==ids[start:start+16].tolist()
    if binding['variant']==252:
        selected=data.train[np.random.RandomState(257000+seed).permutation(35000)[:3500]]
        rows={x['update']:x for x in map(json.loads,(out/'head/training.jsonl').read_text().splitlines())}
        assert sorted(rows)==list(range(1,35001))
        for epoch in range(10):
            order=np.random.RandomState(251000+seed+10000*epoch).permutation(3500)
            assert [rows[epoch*3500+j+1]['image_id'] for j in range(3500)]==selected[order].tolist()
    from full_benchmark.evaluation import metrics,KEYS
    frozen=json.loads((out/'evaluation/frozen_predictions.json').read_bytes())
    assert frozen['ids']==data.test.tolist() and not frozen['GT_used_for_prediction']
    for name,sha in frozen['hashes'].items():assert c.sha(out/'evaluation'/name)==sha
    predictions=np.stack([np.load(out/'evaluation'/f'{int(i):06d}.npy',allow_pickle=False) for i in data.test])
    score,_=metrics(predictions,data,data.test)
    original=json.loads((out/'evaluation/result.json').read_bytes())
    assert all(abs(score['pixel'][k]-original['pixel'][k])<1e-12 for k in KEYS)
    result=dict(status='independently_audited',metrics=score['pixel'],seed=seed,variant=binding['variant'],
        dataset=data.dataset,complete_sha256=c.sha(out/'complete.json'),all_TRAIN_IDs_and_updates_verified=True,
        frozen_prediction_hashes_verified=True,protected_metrics_recomputed=True)
    c.write(out/'audit.json',result);return result


def queue(root,config,source_inventory,head_inventory,gpus):
    root=Path(root);lock=(root/'separate_coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=json.loads((root/'separate_preflight.json').read_bytes())
    assert pre['code']==sources() and pre['heads_sha256']==c.sha(head_inventory)
    records={};errors={};finished={};last_audit=0
    while True:
        for job in jobs():
            for admission in (True,False):
                out=path(root,job,admission);key=str(out)
                if (out/'complete.json').exists():
                    if key not in finished:finished[key]=verify(out,admission)
                    records.pop(key,None)
                elif (out/'launch.json').exists() and key not in errors:
                    rec=json.loads((out/'launch.json').read_bytes())
                    if live(rec):records[key]=rec
                    else:
                        errors[key]=dict(job=job,admission=admission,record=rec,
                            reason='worker terminal without completion; preserve output and diagnose before resume')
                        records.pop(key,None)
        free={int(a):int(b) for a,b in (x.split(',') for x in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().splitlines())}
        occupied={r['gpu'] for r in records.values()}
        for job in jobs():
            if not free:break
            admission=str(path(root,job,True)) not in finished
            out=path(root,job,admission);key=str(out)
            if key in records or key in errors or key in finished:continue
            # Full jobs run only after their exact seed/dataset admission passes.
            if not admission and str(path(root,job,True)) not in finished:continue
            candidates=[g for g in gpus if g not in occupied and free.get(g,0)>=14000]
            if not candidates:break
            gpu=max(candidates,key=lambda g:free[g]);out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-B',str(HERE/'run_separate_encoder.py'),'--config',str(config),
                '--sources',str(source_inventory),'--heads',str(head_inventory),'--output',str(root),
                'train','--model',str(job['model']),'--dataset',job['dataset'],'--seed',str(job['seed'])]
            if admission:command.append('--admission')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SW0257_ALLOCATOR_MIB='13000')
            stream=(out/'worker.log').open('ab');proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                env=env,start_new_session=True);stream.close()
            stat=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
            record=dict(pid=proc.pid,start_ticks=int(stat[19]),command=command,gpu=gpu,created=time.time())
            c.write(out/'launch.json',record);records[key]=record;occupied.add(gpu)
        # Audits execute only after a complete condition, with metrics independently recomputed.
        if time.time()-last_audit>60:
            for job in jobs():
                out=path(root,job,False)
                if str(out) in finished and not (out/'audit.json').exists():audit(out)
            last_audit=time.time()
        groups=[];audits=[]
        for job in jobs():
            p=path(root,job,False)/'audit.json'
            if p.exists():audits.append(json.loads(p.read_bytes()))
        for v in (252,253):
            for ds in ('clevr','tetrominoes','clevrtex_original'):
                rows=[x for x in audits if x['variant']==v and x['dataset']==ds]
                if len(rows)==3:
                    groups.append(dict(variant=v,dataset=ds,seeds=rows,
                        mean={k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']}))
        status=dict(status='complete' if len(audits)==18 else 'active',audited=len(audits),required=18,
            running=list(records.values()),errors=errors,completed_admissions=sum('/admission/' in x for x in finished),
            completed_training=sum('/jobs/' in x for x in finished),groups=groups,updated=time.time())
        c.write(root/'status.json',status)
        if len(audits)==18:return status
        time.sleep(20)
