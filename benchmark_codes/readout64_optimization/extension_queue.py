"""Continuation queue selected strictly by validation, leaving the fixed320 untouched."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);a=p.parse_args();root=Path(a.root)
 lock=(root/'extension.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 screen=json.loads((root/'screen_results.json').read_bytes())
 arms=sorted(screen['validation_means'],key=screen['validation_means'].get,reverse=True)[:2]
 write(root/'extension_selection.json',dict(arms=arms,basis='common three-seed held-out validationFGARI, before fixed320 scoring'))
 tasks=[(arm,seed) for arm in arms for seed in range(3)];running={};done=[]
 while tasks or running:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()};used={r['gpu'] for r in running.values()}
  for task in list(tasks):
   choices=[g for g,m in free.items() if m>=4500 and g not in used]
   if not choices:break
   gpu=max(choices,key=free.get);used.add(gpu);arm,seed=task
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   name=f'extend_s{seed}_{arm}';log=(root/(name+'.log')).open('a')
   proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('extend.py')),'--root',str(root),'--code',a.code,'--seed',str(seed),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   running[task]=dict(arm=arm,seed=seed,gpu=gpu,pid=proc.pid,started=time.time(),name=name,process=proc,stream=log);tasks.remove(task)
  for task,row in list(running.items()):
   exit_code=row['process'].poll()
   if exit_code is None:continue
   row['stream'].close();del running[task]
   if exit_code:write(root/'extension_error.json',dict(seed=row['seed'],arm=row['arm'],exit_code=exit_code,log=row['name']+'.log'));return
   done.append(task)
  write(root/'extension_status.json',dict(updated=time.time(),running=[{k:v for k,v in r.items() if k not in ['process','stream']} for r in running.values()],pending=tasks,done=done))
  time.sleep(15)
 means={arm:sum(json.loads((root/'extended'/f'seed{s}'/arm/'complete.json').read_bytes())['best_validation_FGARI'] for s in range(3))/3 for arm in arms}
 write(root/'final_selection.json',dict(arm=max(means,key=means.get),validation_means=means,selection='labelled validation only; no fixed320 tuning',core_grid=32,head_grid=64))
 write(root/'extension_complete.json',dict(completed=time.time(),trained_jobs=6))
if __name__=='__main__':main()
