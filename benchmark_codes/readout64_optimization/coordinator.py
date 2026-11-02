"""Durable safe GPU queue: prepare3, admit3, then15 short native64-readout trials."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ARMS=['local_rgb','local_rgb_tv','local_rgb_spike','local_rgb_trust','local_rgb_r2']
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=[(stage,seed,'') for stage in ['prepare','admission'] for seed in range(3)]+[('train',seed,arm) for arm in ARMS for seed in range(3)]
 running={};done=set();launches=[];script=Path(__file__).with_name('run.py')
 while tasks or running:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()};used={r['gpu'] for r in running.values()}
  for task in list(tasks):
   stage,seed,arm=task
   if stage=='admission' and ('prepare',seed,'') not in done:continue
   if stage=='train' and not all(('admission',s,'') in done for s in range(3)):continue
   choices=[g for g,m in free.items() if m>=4500 and g not in used]
   if not choices:break
   gpu=max(choices,key=free.get);used.add(gpu)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   name=f'{stage}_s{seed}'+('_'+arm if arm else '');log=(out/(name+'.log')).open('a')
   command=[sys.executable,'-B',str(script),stage,'--output',str(out),'--seed',str(seed)]
   if arm:command+=['--arm',arm]
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   record=dict(stage=stage,seed=seed,arm=arm,gpu=gpu,pid=proc.pid,started=time.time(),name=name)
   running[task]=dict(**record,process=proc,stream=log);launches.append(record);tasks.remove(task);write(out/'launches.json',launches)
  for task,row in list(running.items()):
   code=row['process'].poll()
   if code is None:continue
   row['stream'].close();del running[task]
   if code:
    write(out/'error.json',dict(stage=row['stage'],seed=row['seed'],arm=row['arm'],exit_code=code,log=row['name']+'.log',other_workers_not_stopped=True));return
   done.add(task)
  write(out/'status.json',dict(updated=time.time(),running=[{k:v for k,v in r.items() if k not in ['process','stream']} for r in running.values()],pending=[list(t) for t in tasks],done=[list(t) for t in done]))
  time.sleep(15)
 summary={arm:[] for arm in ARMS}
 for arm in ARMS:
  for seed in range(3):summary[arm].append(json.loads((out/'screen'/f'seed{seed}'/arm/'complete.json').read_bytes()))
 means={arm:sum(r['best_validation_FGARI'] for r in rows)/3 for arm,rows in summary.items()}
 write(out/'screen_results.json',dict(per_arm=summary,validation_means=means,selected_arm=max(means,key=means.get),selection='held-out labelled validation only; no fixed320 tuning',core_grid=32,head_grid=64))
 write(out/'screen_complete.json',dict(completed=time.time(),trained_jobs=15))
if __name__=='__main__':main()
