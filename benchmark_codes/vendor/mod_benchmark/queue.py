"""Register all 12 jobs and share any GPU with sufficient remaining memory."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from . import common as c

def summarize(archive,jobs):
    rows=[json.loads((Path(job['folder'])/'result.json').read_bytes()) for job in jobs]
    groups={}
    for ds in c.DATASETS:
        groups[ds]={}
        for mode in ('zero_shot','finetune'):
            chosen=[r for r in rows if r['dataset']==ds and r['mode']==mode]
            assert sorted(r['seed'] for r in chosen)==[0,1,2]
            keys=chosen[0]['metrics']
            groups[ds][mode]={k:dict(mean=float(np.mean([r['metrics'][k] for r in chosen])),
                seed_std=float(np.std([r['metrics'][k] for r in chosen],ddof=1))) for k in keys}
    c.write(archive/'summary.json',dict(status='complete_twelve_dataset_mode_seed_jobs',
        protocol='TEST1000 pixel metrics; 3 independently trained source seeds; finetune TRAIN6000x10',
        groups=groups,per_seed=rows,benchmark_files=c.immutable(),followup_search=False))

def main(archive):
    archive=Path(archive);archive.mkdir(parents=True,exist_ok=True)
    lock=(archive/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (archive/'preparation_complete.json').exists();c.immutable()
    registration=archive/'registration.json'
    if registration.exists():
        registered=json.loads(registration.read_bytes());jobs=registered['jobs']
        for job in jobs:
            if job['state']=='running':
                pid=job.get('pid')
                if pid and Path('/proc',str(pid)).exists():raise RuntimeError('prior registered worker is still alive; do not duplicate')
                job['state']='queued'
    else:
        jobs=[]
        for seed in (0,1,2):
            for mode in ('zero_shot','finetune'):
                for ds in c.DATASETS:
                    folder=archive/'jobs'/ds/mode/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                    jobs.append(dict(dataset=ds,seed=seed,mode=mode,folder=str(folder),state='queued',attempt=0))
        c.write(registration,dict(status='twelve_jobs_registered',jobs=jobs))
    active={};leases=Path('/tmp/kevinswk_sw0113_gpu_leases');leases.mkdir(exist_ok=True)
    while True:
        for gpu,(job,process,lease,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            log.close();fcntl.flock(lease,fcntl.LOCK_UN);lease.close();del active[gpu]
            folder=Path(job['folder'])
            if code==0 and (folder/'result.json').exists():job['state']='complete'
            elif 'out of memory' in Path(job['log']).read_text(errors='replace')[-12000:].lower() and job['attempt']<4:
                job['state']='queued';job['retry_after']=time.time()+60
            else:job.update(state='failed_preserved',exit_code=code)
        if all(job['state']=='complete' for job in jobs):
            summarize(archive,jobs)
            c.write(archive/'state.json',dict(status='complete_no_followup',jobs=jobs));break
        # No yielding on arrival of another user's process; no signalling foreign PIDs.
        snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if snapshot.returncode:raise RuntimeError(snapshot.stderr)
        capacities=[tuple(map(int,line.split(','))) for line in snapshot.stdout.strip().splitlines()]
        for gpu,free in sorted(capacities,key=lambda row:(-row[1],row[0])):
            if gpu in active or free<8192:continue
            waiting=next((j for j in jobs if j['state']=='queued' and j.get('retry_after',0)<=time.time()),None)
            if waiting is None:break
            lease=(leases/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();continue
            waiting['attempt']+=1;path=Path(waiting['folder'])/f'worker_attempt{waiting["attempt"]:03d}.log'
            log=path.open('xb')
            command=[sys.executable,'-u','-B','-m','mod_benchmark.worker','--archive',str(archive),
                '--dataset',waiting['dataset'],'--mode',waiting['mode'],'--seed',str(waiting['seed'])]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            process=subprocess.Popen(command,cwd=c.CODE,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lease.fileno(),))
            waiting.update(state='running',pid=process.pid,gpu=gpu,log=str(path))
            active[gpu]=(waiting,process,lease,log)
        c.write(registration,dict(status='twelve_jobs_registered',jobs=jobs))
        c.write(archive/'state.json',dict(status='jobs_running_or_waiting',pid=os.getpid(),
            completed=sum(j['state']=='complete' for j in jobs),running=len(active),jobs=jobs))
        if not active and all(j['state'] in ('complete','failed_preserved') for j in jobs):break
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);main(p.parse_args().archive)
