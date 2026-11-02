"""Three R2 loss controls x three seeds; original input/core/encoder unchanged."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ARMS=['delta_prior1','delta_prior05','delta_spike_trust']
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=[(arm,seed) for arm in ARMS for seed in range(3)];running={};done=[]
 while tasks or running:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()};used={r['gpu'] for r in running.values()}
  for task in list(tasks):
   choices=[g for g,m in free.items() if m>=4500 and g not in used]
   if not choices:break
   gpu=max(choices,key=free.get);used.add(gpu);arm,seed=task;name=f'delta_s{seed}_{arm}';log=(root/(name+'.log')).open('a')
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('delta_screen.py')),'--root',str(root),'--code',a.code,'--seed',str(seed),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   running[task]=dict(arm=arm,seed=seed,gpu=gpu,pid=proc.pid,name=name,started=time.time(),process=proc,stream=log);tasks.remove(task)
  for task,row in list(running.items()):
   result=row['process'].poll()
   if result is None:continue
   row['stream'].close();del running[task]
   if result:write(root/'error.json',dict(seed=row['seed'],arm=row['arm'],exit_code=result,log=row['name']+'.log'));return
   done.append(task)
  write(root/'status.json',dict(updated=time.time(),running=[{k:v for k,v in r.items() if k not in ['process','stream']} for r in running.values()],pending=tasks,done=done));time.sleep(15)
 means={arm:sum(json.loads((root/'screen'/f'seed{s}'/arm/'complete.json').read_bytes())['best_validation_FGARI'] for s in range(3))/3 for arm in ARMS}
 write(root/'screen_results.json',dict(validation_means=means,selected_arm=max(means,key=means.get),selection='validation1000-1031 only; no fixed320 tuning',routing_variant='baseline_preserving_delta',core_grid=32,head_grid=64))
 write(root/'screen_complete.json',dict(completed=time.time(),trained_jobs=9))
if __name__=='__main__':main()
