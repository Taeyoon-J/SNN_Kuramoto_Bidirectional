"""Persistent portable queue; same ordering, budgets and scientific engine as live run."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from portable_runtime import HERE
from mod_benchmark import common as c
from full_benchmark.engine import scientific_sources

def conditions():
    request=json.loads((HERE/'snapshot/controller/REQUEST.json').read_text())
    return sorted(request['conditions'],key=lambda x:x['priority'])
def location(root,job,admission):
    return root/('engine_admission' if admission else 'jobs')/f"model{job['model']}"/job['dataset']/f"seed{job['seed']}"
def alive(launch):
    try:
        proc=Path('/proc')/str(launch['pid']);args=proc.joinpath('cmdline').read_bytes().split(b'\0')
        return proc.stat().st_uid==os.getuid() and str(HERE/'run.py').encode() in args and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False
def completed(root,job,admission):
    out=location(root,job,admission)
    if not (out/'complete.json').exists():return False
    record=json.loads((out/'complete.json').read_bytes());binding=json.loads((out/'binding.json').read_bytes())
    assert record['binding_sha256']==c.sha(out/'binding.json')
    assert binding['code']==scientific_sources() and binding['benchmark']==c.immutable()
    assert record['updates']==(2 if admission else 2190) and record['images_seen']==(4 if admission else 35000)
    assert record['training_report_sha256']==c.sha(out/'training_report.json')
    assert record['evaluation_sha256']==c.sha(out/'evaluation/result.json')
    report=json.loads((out/'training_report.json').read_bytes())
    assert all(x>0 for x in report['active_tensor_gradient_coverage'].values())
    assert report['mask_opened_during_training']==(job['model']!=253)
    return True
def rows_for(root):
    rows=[];live=[]
    for condition in conditions():
        for seed in (0,1,2):
            row=dict(model=condition['model'],dataset=condition['dataset'],seed=seed,phase=condition['phase'],priority=condition['priority'])
            for admission,key in ((True,'admission'),(False,'training')):
                out=location(root,row,admission)
                if completed(root,row,admission):row[key]='complete'
                elif (out/'launch.json').exists():
                    launch=json.loads((out/'launch.json').read_bytes())
                    if alive(launch):row[key]='running';live.append(launch)
                    else:row[key]='terminal_incomplete_requires_diagnosis'
                else:row[key]='queued'
            rows.append(row)
    return rows,live
def audit_all(root):
    from snapshot.controller.audit_matrix import write_reports
    rows,_=rows_for(root);audited=write_reports(root,rows)
    return dict(audited_conditions=audited,required_conditions=20,complete=audited==20)
def run_queue(config,root,gpus):
    root=Path(root);lock=(root/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert gpus and len(gpus)==len(set(gpus)) and all(g>=0 for g in gpus)
    assert (root/'preflight.json').exists(),'run preflight --dataset all first'
    assert set(json.loads((root/'preflight.json').read_text()))=={'clevr','multi_dsprites','tetrominoes','objects_room','clevrtex_original'}
    while True:
        rows,live=rows_for(root)
        phase=2 if all(x['training']=='complete' for x in rows if x['phase']==1) else 1
        query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        free={int(a):int(b) for a,b in (line.split(',') for line in query.stdout.splitlines()) if int(a) in gpus}
        usage=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        used={int(a):int(b) for a,b in (line.split(',') for line in usage.stdout.splitlines()) if a.strip().isdigit() and b.strip().isdigit()}
        def budget(model):return 3000 if model==253 else 11000
        for worker in live:free[worker['gpu']]=max(0,free[worker['gpu']]-max(0,budget(worker['model'])-used.get(worker['pid'],0)))
        while len(live)<8:
            chosen=None
            eligible=[r for r in rows if r['phase']==phase]
            reserve=any(r['admission']!='complete' for r in eligible) and not any(r['admission'] for r in live)
            for gpu in sorted(free,key=free.get,reverse=True):
                if sum(w['gpu']==gpu for w in live)>=2:continue
                modes=(True,False) if reserve else (False,True)
                for admission in modes:
                    chosen=next((r for r in eligible if r['admission' if admission else 'training']=='queued' and (admission or r['admission']=='complete') and free[gpu]>budget(r['model'])),None)
                    if chosen is not None:break
                if chosen is not None:break
            if chosen is None:break
            out=location(root,chosen,admission);out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-B',str(HERE/'run.py'),'--config',str(Path(config).resolve()),'--output',str(root),'train','--dataset',chosen['dataset'],'--model',str(chosen['model']),'--seed',str(chosen['seed'])]
            if admission:command.append('--admission')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
                proc=subprocess.Popen(command,env=env,cwd=HERE,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            launch=dict(model=chosen['model'],dataset=chosen['dataset'],seed=chosen['seed'],admission=admission,pid=proc.pid,gpu=gpu,command=command,started=time.time())
            c.write(out/'launch.json',launch);live.append(launch)
            chosen['admission' if admission else 'training']='running';free[gpu]-=budget(chosen['model'])
        c.write(root/'queue.json',dict(status='active',phase=phase,rows=rows,live=live,updated=time.time()))
        result=audit_all(root)
        if result['complete']:c.write(root/'complete.json',result);return result
        time.sleep(20)
