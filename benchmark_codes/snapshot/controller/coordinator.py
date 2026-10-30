"""Persistent server queue: genuine admissions first; phase2 cannot overtake phase1."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/work/USERS/kevinswk/experiments/SW0255_full3500')
CODE=ROOT/'code/engine_3500_v1'
OLD=Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/SW0251_background_mask')
sys.path[:0]=[str(CODE),str(OLD/'operations/SW0252_code'),str(OLD/'code')]
from mod_benchmark import common as c
from full_benchmark.engine import scientific_sources
from readout_benchmark.bootstrap import gpu_free
from audit_matrix import write_reports


def budget(variant):return 3000 if variant==253 else 11000


def usable_memory(live,free,used_by_pid):
    result=dict(free)
    for worker in live:
        extra=max(0,budget(worker['variant'])-used_by_pid.get(worker['pid'],0))
        result[worker['gpu']]=max(0,result[worker['gpu']]-extra)
    return result


def folder(job,admission):
    return ROOT/('engine_admission' if admission else 'jobs')/f"model{job['model']}"/job['dataset']/f"seed{job['seed']}"


def alive(record):
    process=Path('/proc')/str(record['pid'])
    try:
        args=process.joinpath('cmdline').read_bytes().split(b'\x00')
        state=process.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
        return process.stat().st_uid==os.getuid() and b'full_benchmark.engine' in args and state!='Z' and (
            args[args.index(b'--variant')+1]==str(record['variant']).encode() and
            args[args.index(b'--dataset')+1]==record['dataset'].encode() and
            args[args.index(b'--seed')+1]==str(record['seed']).encode())
    except (OSError,ValueError,IndexError):return False


def completed(job,admission,sources):
    out=folder(job,admission);path=out/'complete.json'
    if not path.exists():return False
    record=json.loads(path.read_bytes());binding=json.loads((out/'binding.json').read_bytes())
    assert record['binding_sha256']==c.sha(out/'binding.json')
    assert binding['code']==sources and binding['benchmark']==c.immutable()
    assert record['training_report_sha256']==c.sha(out/'training_report.json')
    assert record['evaluation_sha256']==c.sha(out/'evaluation/result.json')
    report=json.loads((out/'training_report.json').read_bytes())
    assert report['optimizer_resume_verified']
    assert all(v>0 for v in report['active_tensor_gradient_coverage'].values())
    assert all(v>0 for v in report['parameter_changes'].values())
    assert report['mask_opened_during_training']==(job['model']!=253)
    assert record['updates']==(2 if admission else 2190)
    assert record['images_seen']==(4 if admission else 35000)
    assert record['diagnostic_only']==admission
    assert record['evaluation_count']==(1 if admission else 320 if job['dataset']=='clevr' else 1000)
    return True


def launch(job,gpu,admission):
    out=folder(job,admission);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'launch.json').exists(),'never auto-relaunch a terminal worker'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(CODE),str(OLD/'operations/SW0252_code'),str(OLD/'code')]),
        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-u','-B','-m','full_benchmark.engine','--variant',str(job['model']),
        '--seed',str(job['seed']),'--dataset',job['dataset'],'--root',str(ROOT)]
    if admission:command.append('--admission')
    with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
        proc=subprocess.Popen(command,cwd=OLD/'code',env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    record=dict(variant=job['model'],dataset=job['dataset'],seed=job['seed'],pid=proc.pid,gpu=gpu,
        command=command,source=str(CODE),output=str(out),scientific_training_started=not admission,started=time.time())
    c.write(out/'launch.json',record);return record


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specification=json.loads((Path(__file__).parent/'REQUEST.json').read_bytes())
    conditions=sorted(specification['conditions'],key=lambda x:x['priority'])
    jobs=[dict(model=x['model'],dataset=x['dataset'],seed=seed,phase=x['phase'],priority=x['priority']) for x in conditions for seed in (0,1,2)]
    assert len(jobs)==60 and sum(x['phase']==1 for x in jobs)==36
    sources=scientific_sources()
    # Confirm each MOD cache's actual file SHA, not just a readiness filename.
    for dataset in ('multi_dsprites','tetrominoes','objects_room','clevrtex_original'):
        directory=ROOT/'inputs'/dataset;manifest=json.loads((directory/'manifest.json').read_bytes())
        assert manifest['status']=='complete_verified_TRAIN35000'
        for name,identity in manifest['files'].items():
            file=directory/name
            assert file.stat().st_size==identity['size'] and c.sha(file)==identity['sha256']
    for dataset in ('objects_room','clevrtex_original'):
        preflight=json.loads((ROOT/'extra_preflight'/dataset/'preflight.json').read_bytes())
        assert preflight['status']=='actual_phase2_input_split_modal_and_pixel_scoring_preflight_passed'
        assert preflight['input_manifest_sha256']==c.sha(ROOT/'inputs'/dataset/'manifest.json')
    c.write(ROOT/'controller_contract.json',dict(conditions=conditions,total_seed_jobs=60,
        first_phase_seed_jobs=36,phase2_requires_all_phase1_training_and_evaluation=True,
        minimum_free_MiB_native64=11000,minimum_free_MiB_native32=3000,
        maximum_owned_workers_per_GPU=2,maximum_total_owned_workers=8,external_gpu_users_not_preempted=True,
        admission_slot_reserved_until_phase_admissions_complete=True,source=sources))
    while True:
        rows=[];live=[]
        for job in jobs:
            row=dict(job)
            for admission,label in ((True,'admission'),(False,'training')):
                out=folder(job,admission)
                if completed(job,admission,sources):row[label]='complete';continue
                path=out/'launch.json'
                if path.exists():
                    record=json.loads(path.read_bytes())
                    if alive(record):row[label]='running';live.append(dict(record,admission=admission))
                    else:row[label]='terminal_incomplete_requires_diagnosis'
                else:row[label]='queued'
            rows.append(row)
        phase=2 if all(x['training']=='complete' for x in rows if x['phase']==1) else 1
        eligible=[x for x in rows if x['phase']==phase]
        unadmitted=any(x['admission']!='complete' for x in eligible)
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        used_by_pid={}
        for line in query.stdout.splitlines():
            values=[x.strip() for x in line.split(',')]
            if len(values)==2 and all(x.isdigit() for x in values):used_by_pid[int(values[0])]=used_by_pid.get(int(values[0]),0)+int(values[1])
        free=usable_memory(live,gpu_free(),used_by_pid)
        counts={g:sum(x['gpu']==g for x in live) for g in free}
        while len(live)<8:
            candidate=None;diagnostic=False
            reserve_admission=unadmitted and not any(x['admission'] for x in live)
            for gpu in sorted(free,key=free.get,reverse=True):
                if counts[gpu]>=2:continue
                if reserve_admission:
                    candidate=next((x for x in eligible if x['admission']=='queued' and free[gpu]>budget(x['model'])),None)
                    diagnostic=True
                if candidate is None:
                    candidate=next((x for x in eligible if x['admission']=='complete' and x['training']=='queued' and free[gpu]>budget(x['model'])),None)
                    diagnostic=False
                if candidate is None:
                    candidate=next((x for x in eligible if x['admission']=='queued' and free[gpu]>budget(x['model'])),None)
                    diagnostic=True
                if candidate is not None:break
            if candidate is None:break
            record=launch(candidate,gpu,diagnostic);live.append(dict(record,admission=diagnostic))
            candidate['admission' if diagnostic else 'training']='running'
            free[gpu]-=budget(candidate['model']);counts[gpu]+=1
        c.write(ROOT/'queue.json',dict(status='active',phase=phase,rows=rows,live=live,
            scientific_jobs_complete=sum(x['training']=='complete' for x in rows),
            diagnostic_jobs_complete=sum(x['admission']=='complete' for x in rows),updated=time.time(),
            phase2_data_contracts_pending=False,source_version='engine_3500_v1',
            controller_version='controller_v2'))
        audited=write_reports(ROOT,rows)
        if all(x['training']=='complete' for x in rows):
            assert audited==20
            c.write(ROOT/'complete.json',dict(status='all60_seed_jobs_trained_evaluated_and20_conditions_independently_audited',reports_index_sha256=c.sha(ROOT/'reports/index.json')))
            return
        time.sleep(20)


if __name__=='__main__':main()
