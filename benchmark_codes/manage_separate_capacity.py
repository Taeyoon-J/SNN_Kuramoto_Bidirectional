#!/usr/bin/env python3
"""Optional SW0257 feeder: at most two unchanged scientific workers per GPU.

This manager lives outside the immutable scientific package. It launches the
exact source-bound entrypoint, batch16, original13GB allocator and all original
budgets. Its8GB scheduling reservation is guarded by actual CUDA profiling.
The existing coordinator is briefly quiesced only while it is sleeping between
scans, so both managers cannot race to launch the same condition. Training
workers are never signalled. No terminal worker is restarted automatically.
"""
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=path.joinpath('stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=fields[0],start_ticks=int(fields[19]),uid=path.stat().st_uid,
            command=[x.decode() for x in path.joinpath('cmdline').read_bytes().split(b'\0') if x])
    except (OSError,ValueError):return None


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def alive(record):
    current=process(record['pid'])
    return bool(current and current['state']!='Z' and current['uid']==os.getuid()
        and current['start_ticks']==record['start_ticks'] and current['command']==record['command'])


def main(root,*,once=False,dry_run=False):
    root=Path(root);lock=(root/'capacity_feeder.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=json.loads((root/'coordinator_launch.json').read_bytes())
    package=Path(controller['package']);entry=package/'run_separate_encoder.py'
    preflight=json.loads((root/'separate_preflight.json').read_bytes())
    queue_source=package/'separate_encoder_benchmark/queue.py'
    assert hashlib.sha256(queue_source.read_bytes()).hexdigest()==preflight['code'][str(queue_source)]
    proof=json.loads((root/'capacity_control_same_gpu/complete.json').read_bytes())
    assert len(proof['rows'])==2 and {r['variant'] for r in proof['rows']}=={252,253}
    # Keep the ORIGINAL allocator setting, regardless of the8GB control result.
    primary=[r['results'][0] for r in proof['rows']]
    assert all(r['cap']==13000 and r['core_fixed'] for r in primary)
    peak=max(r['peak_reserved_MiB'] for r in primary)
    assert peak*2+1024<8000,('profile does not justify8GB scheduling reservation',peak)
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected=[(v,d,s) for v in (252,253) for d in ('clevr','tetrominoes','clevrtex_original') for s in (0,1,2)]
    while True:
        current=process(controller['pid'])
        if not current or current['state']=='Z':return
        assert current['uid']==os.getuid() and current['start_ticks']==controller['start_ticks']
        assert str(entry) in current['command'] and str(root) in current['command']
        status=json.loads((root/'status.json').read_bytes())
        if status['status']=='complete':return
        # A fresh status is written immediately before the coordinator sleeps20s.
        # This host hides wchan (always0). The hash-locked coordinator has no
        # blocking code between publishing status and its20-second sleep.
        # Fresh published status plus stateS therefore identifies that interval.
        age=time.time()-status['updated']
        if not 0<=age<10 or current['state']!='S':
            time.sleep(2);continue
        paused=False;started=[]
        try:
            os.kill(controller['pid'],signal.SIGSTOP);paused=True
            deadline=time.time()+2
            while process(controller['pid'])['state']!='T':
                if time.time()>deadline:raise RuntimeError('coordinator did not quiesce')
                time.sleep(.01)
            assert json.loads((root/'status.json').read_bytes())['updated']==status['updated']
            active=[]
            for path in root.glob('jobs/model*/*/seed*/launch.json'):
                record=json.loads(path.read_bytes())
                if alive(record):active.append(record)
            gpu_rows=[x.split(',') for x in subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']).decode().splitlines()]
            app_rows=[x.split(',') for x in subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).decode().splitlines()]
            devices={int(r[0]):dict(uuid=r[1].strip(),total=int(r[2]),used=int(r[3]),free=int(r[4])) for r in gpu_rows}
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
            # Reserve10GiB host RAM for each worker's eventual3500-feature cache.
            future=0
            for record in active:
                rss=Path(f"/proc/{record['pid']}/status").read_text()
                used=int(next(x.split()[1] for x in rss.splitlines() if x.startswith('VmRSS:')))/1024
                future+=max(0,10240-used)
            for variant,dataset,seed in expected:
                out=root/'jobs'/f'model{variant}'/dataset/f'seed{seed}'
                admission=root/'admission'/f'model{variant}'/dataset/f'seed{seed}/complete.json'
                if (out/'launch.json').exists() or (out/'complete.json').exists() or not admission.exists():continue
                choices=[]
                for gpu,device in devices.items():
                    own=[r for r in active if r['gpu']==gpu]
                    if len(own)>=2:continue
                    own_ids={r['pid'] for r in own}
                    own_used=sum(int(row[2]) for row in app_rows if row[0].strip()==device['uuid'] and int(row[1]) in own_ids)
                    foreign=max(0,device['used']-own_used)
                    reserved=sum(r.get('reservation_mib',13000) for r in own)
                    if reserved+8000+foreign+1024<=device['total'] and device['free']>=9024:choices.append(gpu)
                if not choices or available-future<18432:continue
                gpu=min(choices,key=lambda g:sum(r['gpu']==g for r in active))
                command=[sys.executable,'-B',str(entry),'--config',str(package/'configs/frontier.json'),
                    '--sources',str(package/'frozen_core_benchmark/sources.frontier.json'),
                    '--heads',str(package/'separate_encoder_benchmark/heads.frontier.json'),'--output',str(root),
                    'train','--model',str(variant),'--dataset',dataset,'--seed',str(seed)]
                if dry_run:
                    pid=-1-len(started);ticks=0
                else:
                    out.mkdir(parents=True,exist_ok=True)
                    with (out/'worker.log').open('ab') as stream:
                        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SW0257_ALLOCATOR_MIB='13000',
                                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                    current=process(proc.pid);assert current and current['command']==command
                    pid=proc.pid;ticks=current['start_ticks']
                record=dict(pid=pid,start_ticks=ticks,command=command,gpu=gpu,created=time.time(),
                    reservation_mib=8000,allocator_limit_MiB=13000,capacity_feeder_sha256=code_sha,
                    capacity_evidence_sha256=hashlib.sha256((root/'capacity_control_same_gpu/complete.json').read_bytes()).hexdigest())
                if not dry_run:write(out/'launch.json',record)
                active.append(record);started.append(record)
                devices[gpu]['free']-=8000;future+=10240
            target=root/('capacity_feeder_dry_run.json' if dry_run else 'capacity_feeder_status.json')
            write(target,dict(observed=time.time(),manager_sha256=code_sha,dry_run=dry_run,
                maximum_workers_per_GPU=2,actual_original13GB_peak_MiB=peak,workers_started=started,
                unchanged_scientific_entrypoint=str(entry),original_allocator_preserved=True,active=active))
        finally:
            current=process(controller['pid'])
            if paused and current and current['start_ticks']==controller['start_ticks'] and current['uid']==os.getuid():
                os.kill(controller['pid'],signal.SIGCONT)
        if once:return
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--once',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();main(args.root,once=args.once,dry_run=args.dry_run)
