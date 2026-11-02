"""Durable SW0259 three-seed admission/train/score queue. Never stops other jobs."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).with_name('run.py');tasks=[(seed,stage) for stage in ('admission','worker') for seed in range(3)]
    running={};finished=set();launches=[]
    while tasks or running:
        gpurows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(r.split(',')[0]):int(r.split(',')[1]) for r in gpurows.strip().splitlines()}
        used={r['gpu'] for r in running.values()}
        for task in list(tasks):
            seed,stage=task
            if stage=='worker' and (seed,'admission') not in finished:continue
            choices=[g for g,m in free.items() if m>=6500 and g not in used]
            if not choices:break
            gpu=max(choices,key=lambda g:free[g]);used.add(gpu)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            logfile=(out/f'seed{seed}_{stage}.log').open('a')
            cmd=[sys.executable,'-B',str(script),stage,'--output',str(out),'--seed',str(seed)]
            proc=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            row=dict(seed=seed,stage=stage,gpu=gpu,pid=proc.pid,started=time.time())
            running[task]=dict(**row,process=proc,stream=logfile);launches.append(row);tasks.remove(task)
            write(out/'launches.json',launches)
        for task,row in list(running.items()):
            code=row['process'].poll()
            if code is None:continue
            row['stream'].close();del running[task]
            if code:
                write(out/'error.json',dict(seed=row['seed'],stage=row['stage'],exit_code=code,remaining=list(tasks),other_workers_left_running=True));return
            finished.add(task)
        write(out/'status.json',dict(updated=time.time(),running=[{k:v for k,v in r.items() if k not in ('process','stream')} for r in running.values()],pending=list(tasks),finished=list(finished)))
        time.sleep(15)
    # Scoring remains queued independently of this chat and SSH connection.
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        choices=[(int(r.split(',')[1]),int(r.split(',')[0])) for r in rows.strip().splitlines()]
        mem,gpu=max(choices)
        if mem>=4000:break
        time.sleep(30)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (out/'score.log').open('a') as f:
        result=subprocess.run([sys.executable,'-B',str(script),'score','--output',str(out)],stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:write(out/'error.json',dict(stage='score',exit_code=result.returncode))
    else:write(out/'complete.json',dict(status='all_three_seeds_scored',completed=time.time()))

if __name__=='__main__':main()
