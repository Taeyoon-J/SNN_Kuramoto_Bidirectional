"""Persistent18-job queue; no process eviction, no automatic training restart."""
import fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from portable_runtime import HERE
from mod_benchmark import common as c
from frozen_core_benchmark.engine import scientific_sources
from frozen_core_benchmark.execution import verify_code,allowed_entrypoints


def jobs():
    request=json.loads((Path(__file__).parent/'REQUEST.json').read_bytes())
    result=[dict(model=row['model'],dataset=row['dataset'],seed=seed,priority=i*3+seed)
            for i,row in enumerate(request['conditions']) for seed in row['seeds']]
    assert len(result)==18 and len({(r['model'],r['dataset'],r['seed']) for r in result})==18
    return result


def location(root,job,admission):
    return Path(root)/('engine_admission' if admission else 'jobs')/f"model{job['model']}"/job['dataset']/f"seed{job['seed']}"


def alive(record):
    p=Path('/proc')/str(record['pid'])
    try:
        args=p.joinpath('cmdline').read_bytes().split(b'\0')
        if record.get('legacy'):recognized=b'full_benchmark.engine' in args
        else:
            root=Path(record['output']).parents[3]
            scripts=allowed_entrypoints(root,scientific_sources())
            recognized=any(script.encode() in args for script in scripts)
            if 'command' in record:recognized=recognized and [x for x in args if x]==[x.encode() for x in record['command']]
        return p.stat().st_uid==os.getuid() and recognized and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False


def completed(root,job,admission):
    out=location(root,job,admission)
    if not (out/'complete.json').exists():return False
    done=json.loads((out/'complete.json').read_bytes());binding=json.loads((out/'binding.json').read_bytes())
    assert binding['experiment']=='SW0256' and binding['unique_train']==35000
    assert binding['core_weight_updates'] is False and binding['encoder_finetuned']
    assert binding['microbatch']==1
    verify_code(root,binding['code'],scientific_sources())
    assert binding['benchmark']==c.immutable() and done['binding_sha256']==c.sha(out/'binding.json')
    assert done['updates']==(2 if admission else 21880) and done['images_seen']==(4 if admission else 350000)
    assert done['evaluation_count']==(1 if admission else 320 if job['dataset']=='clevr' else 1000)
    assert done['diagnostic_only']==admission
    assert done['training_report_sha256']==c.sha(out/'training_report.json')
    assert done['evaluation_sha256']==c.sha(out/'evaluation/result.json')
    report=json.loads((out/'training_report.json').read_bytes())
    assert report['frozen_core_verified'] and report['optimizer_resume_verified']
    assert all(value>0 for value in report['active_tensor_gradient_coverage'].values())
    assert report['mask_opened_during_training']==(job['model']==252)
    return True


def rows_for(root):
    rows=[];live=[]
    for job in jobs():
        row=dict(job)
        for admission,label in ((True,'admission'),(False,'training')):
            out=location(root,job,admission)
            if completed(root,job,admission):row[label]='complete'
            elif (out/'launch.json').exists():
                record=json.loads((out/'launch.json').read_bytes())
                if alive(record):row[label]='running';live.append(record)
                else:row[label]='terminal_incomplete_requires_diagnosis'
            else:row[label]='queued'
        rows.append(row)
    return rows,live


def budget(root,job,admission=False):
    # Admission is physical1 with the same native64 live graph/tail as training.
    out=location(root,job,True);p=out/'training_report.json'
    if not admission and (out/'complete.json').exists():
        report=json.loads(p.read_bytes())
        return max(6500,256*math.ceil((report['peak_GPU_reserved_bytes']/1024**2+1024)/256))
    return 7500


def legacy_budget(worker,used_mib):
    """Prefer the legacy worker's verified native64 admission peak plus1.5GiB.

    The earlier11GiB constant is a scheduling reservation, not measured usage.
    Preserve it unless actual two-update/save/restore evidence is available.
    """
    try:
        out=Path(worker['output'])
        admission=out.parents[3]/'engine_admission'/out.parents[1].name/out.parent.name/out.name
        done=json.loads((admission/'complete.json').read_bytes())
        report=json.loads((admission/'training_report.json').read_bytes())
        binding=json.loads((admission/'binding.json').read_bytes())
        evaluation=json.loads((admission/'evaluation/result.json').read_bytes())
        assert done['diagnostic_only'] and done['updates']==2 and done['images_seen']==4
        assert done['training_report_sha256']==c.sha(admission/'training_report.json')
        assert done['binding_sha256']==c.sha(admission/'binding.json')
        assert done['evaluation_sha256']==c.sha(admission/'evaluation/result.json')
        assert binding['variant']==worker['variant'] and binding['seed']==worker['seed']
        assert binding['microbatch']==1 and evaluation['native_model_grid']==64
        assert binding['total_rollout_steps']==1024 and binding['gradient_tail_steps']==64
        assert report['optimizer_resume_verified']
        return max(used_mib,math.ceil(report['peak_GPU_bytes']/1024**2)+1536)
    except (OSError,KeyError,AssertionError,ValueError):return 11000


def audit_all(root):
    from frozen_core_benchmark.audit import audit_seed
    root=Path(root);rows,_=rows_for(root);finished=[]
    for row in rows:
        if row['training']!='complete':continue
        out=location(root,row,False);path=root/'reports/per_seed'/f"model{row['model']}_{row['dataset']}_seed{row['seed']}.json"
        stamp=c.sha(out/'complete.json')
        if path.exists():
            result=json.loads(path.read_bytes());assert result['completion_sha256']==stamp
        else:
            verified=audit_seed(root,row)
            result=dict(model=row['model'],dataset=row['dataset'],completion_sha256=stamp,**verified)
            c.write(path,result)
        finished.append(result)
    groups=[]
    for variant in (252,253):
        for dataset in ('clevr','tetrominoes','clevrtex_original'):
            selected=sorted([r for r in finished if r['model']==variant and r['dataset']==dataset],key=lambda r:r['seed'])
            if len(selected)!=3:continue
            assert [r['seed'] for r in selected]==[0,1,2]
            group=dict(model=variant,dataset=dataset,seeds=[0,1,2],unique_TRAIN=35000,epochs=10,
                per_seed=selected,mean={k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']},
                std_population={k:float(np.std([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']},
                frozen_core=True,encoder_finetuned=True,core_source='same-seed SW0097 native16 strictly mapped64',
                evaluation_contract=selected[0]['evaluation_contract'],protected_benchmark=c.immutable())
            c.write(root/'reports'/f'model{variant}_{dataset}.json',group);groups.append(group)
    index=dict(status='complete' if len(finished)==18 and len(groups)==6 else 'active',
        audited_seed_jobs=len(finished),required_seed_jobs=18,three_seed_groups_complete=len(groups),
        groups=groups,per_seed_results=finished)
    c.write(root/'reports/index.json',index);return index


def run_queue(config,sources,root,gpus,legacy_root):
    root=Path(root);lock=(root/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(gpus)==len(set(gpus)) and gpus
    ready=json.loads((root/'preflight.json').read_bytes())
    assert ready['status']=='SW0256_sources_full35000_inputs_and_protected_preflights_passed'
    assert ready['sources_sha256']==c.sha(sources)
    checked=json.loads((root/'CPU_validation.json').read_bytes())
    assert checked['source_inventory_sha256']==c.sha(sources) and len(checked['rows'])==6
    verify_code(root,checked['code'],scientific_sources())
    while True:
        rows,live=rows_for(root)
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
        free={int(a):int(b) for a,b in (line.split(',') for line in query.splitlines()) if int(a) in gpus}
        free={gpu:max(0,value-512) for gpu,value in free.items()}  # global allocation/race margin
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits']).decode()
        used={int(a):int(b) for a,b in (line.split(',') for line in query.splitlines()) if a.strip().isdigit() and b.strip().isdigit()}
        legacy=[];old=Path(legacy_root)/'queue.json'
        if old.exists():
            for worker in json.loads(old.read_bytes())['live']:
                record=dict(worker,legacy=True)
                if alive(record):legacy.append(record)
        for worker in live:
            free[worker['gpu']]=max(0,free[worker['gpu']]-max(0,budget(root,worker,worker['admission'])-used.get(worker['pid'],0)))
        for worker in legacy:
            if worker['gpu'] in free:
                usage=used.get(worker['pid'],0)
                free[worker['gpu']]=max(0,free[worker['gpu']]-max(0,legacy_budget(worker,usage)-usage))
        while len(live)<16:
            candidate=None;diagnostic=False
            # Give one real admission priority while other admitted jobs train.
            reserve=not any(w['admission'] for w in live)
            for gpu in sorted(free,key=free.get,reverse=True):
                if sum(w['gpu']==gpu for w in live+legacy)>=4:continue
                modes=(True,False) if reserve else (False,True)
                for diagnostic in modes:
                    if diagnostic and any(w['admission'] for w in live):continue
                    candidate=next((r for r in rows if r['admission' if diagnostic else 'training']=='queued'
                        and (diagnostic or r['admission']=='complete') and free[gpu]>budget(root,r,diagnostic)),None)
                    if candidate is not None:break
                if candidate is not None:break
            if candidate is None:break
            out=location(root,candidate,diagnostic);out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-B',str(HERE/'run_frozen_core.py'),'--config',str(Path(config).resolve()),
                '--sources',str(Path(sources).resolve()),'--output',str(root),'train','--model',str(candidate['model']),
                '--dataset',candidate['dataset'],'--seed',str(candidate['seed'])]
            if diagnostic:command.append('--admission')
            with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                env['SW0256_ALLOCATOR_LIMIT_MIB']=str(budget(root,candidate,diagnostic)-768)
                proc=subprocess.Popen(command,cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            record=dict(model=candidate['model'],variant=candidate['model'],dataset=candidate['dataset'],seed=candidate['seed'],
                pid=proc.pid,gpu=gpu,admission=diagnostic,output=str(out),command=command,started=time.time())
            c.write(out/'launch.json',record);live.append(record)
            candidate['admission' if diagnostic else 'training']='running';free[gpu]-=budget(root,candidate,diagnostic)
        c.write(root/'queue.json',dict(status='active',rows=rows,live=live,legacy_workers_preserved=legacy,
            scientific_jobs_complete=sum(r['training']=='complete' for r in rows),diagnostic_jobs_complete=sum(r['admission']=='complete' for r in rows),
            total_scientific_seed_jobs=18,updated=time.time(),controller_version='SW0256_fixed_core35000_3seeds'))
        result=audit_all(root)
        if result['status']=='complete':c.write(root/'complete.json',result);return result
        time.sleep(30)
