"""Module-launched CPU audit; actual checkpoints and all scored masks are required."""
import json,os,subprocess,sys,time
from pathlib import Path


def activate_repo(root):
    root=Path(root).resolve();os.chdir(root)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return root


def main():
    import ctypes
    if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    here=Path(__file__).resolve().parent;root=activate_repo(here.parents[1]);a=here/'results_archive';out=a/'completion_audit_v1';out.mkdir(exist_ok=True)
    def write(name,value):
        p=out/(name+'.tmp');p.write_bytes((json.dumps(value,indent=2)+chr(10)).encode());p.replace(out/name)
    def state(status,**fields):write('state.json',dict(status=status,pid=os.getpid(),updated_unix=time.time(),**fields))
    if (out/'state.json').exists():raise FileExistsError('preserve existing auditor; do not duplicate')
    try:
        done=set();deadline=time.monotonic()+7*86400
        while time.monotonic()<deadline:
            for seed in (0,1,2):
                for arm in ('control','within16'):
                    folder=a/arm/f'seed{seed}';key=(seed,arm);qp=folder/'queue/state.json'
                    if qp.exists() and json.loads(qp.read_bytes())['status']=='failed':raise RuntimeError('preserved registered queue failure')
                    if key in done or not (folder/'training_report.json').exists():continue
                    state('auditing_actual_CPU_checkpoint',seed=seed,arm=arm,completed_audits=len(done))
                    with (out/f'seed{seed}_{arm}.stdout').open('xb') as stdout,(out/f'seed{seed}_{arm}.stderr').open('xb') as stderr:
                        subprocess.run([sys.executable,'-B','-m','collaborative_test.SW_0247_native64_precision_convergence.run','--arm',arm,'--seed',str(seed),'--audit-training'],cwd=root,env=os.environ,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,check=True,timeout=1200)
                    done.add(key)
            if len(done)==6 and (a/'score_v1/summary.json').exists():break
            state('waiting_for_registered_jobs',completed_audits=len(done));time.sleep(45)
        else:raise TimeoutError('observation deadline; GPU jobs untouched')
        import numpy as np
        from collaborative_test.SW_0247_native64_precision_convergence import run
        run.activate('control');score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw)
        if score['fingerprint']!=run.family_fingerprint():raise ValueError('registered completed source changed')
        bundles,old,slots,dataset=run.evaluation_bundles()
        gt=run.reference.source.source.full.read_gt(dict(dataset=dataset))[list(run.OFFSETS)]
        vectors,maximum=run.metric_vectors(bundles,old,slots,gt);private=json.loads((a/'score_v1/evaluation_private.json').read_bytes());keys=run.reference_score.METRICS;means={k:v.mean((0,1)) for k,v in vectors.items()}
        for name,values in vectors.items():
            maximum=max(maximum,float(np.abs(values-np.asarray(private['vectors'][name])).max()),float(np.abs(means[name]-np.array([score['scores'][name][k] for k in keys])).max()))
            for seed in run.SEEDS:maximum=max(maximum,float(np.abs(values[seed].mean(0)-np.array([score['per_seed'][name][str(seed)][k] for k in keys])).max()))
        qualified=[]
        for arm,new,old_name in (('control','control512','control128'),('within16','within16_512','within16_128')):
            delta=means[new]-means[old_name];maximum=max(maximum,float(np.abs(delta-np.array([score['budget512_minus128'][arm][k] for k in keys])).max()))
            if delta[0]>=.01 and means[new][0]-means['parent237'][0]>=.01 and np.all(means[new]>means['slot']):qualified.append(arm)
        winner=sorted(qualified,key=lambda arm:(-means['control512' if arm=='control' else 'within16_512'][0],arm))[0] if qualified else None
        if maximum>=1e-12 or qualified!=score['qualified_arms_for_separate_full320_confirmation'] or winner!=score['selected_arm_for_separate_full320_confirmation']:raise ValueError('independent means/deltas/selection differ')
        audits={arm:{str(seed):json.loads((run.archive_for(seed,arm)/'training_audit.json').read_bytes()) for seed in run.SEEDS} for arm in run.ARMS}
        for rows in audits.values():
            for row in rows.values():
                if row['actual_Adam_steps']!=[512] or row['actual_stage_updates']!=512 or not row['all_core_and_projection_parameters_unchanged']:raise ValueError('all six actual512-update checkpoint audits required')
        result=dict(status='completed_six_actual512_checkpoints_and_raw_mask_audit',fingerprint=run.family_fingerprint(),score_sha256=run.sha(score_raw),audit_code_sha256=run.sha(Path(__file__).read_bytes()),training_audits=audits,
            raw_masks_independently_rescored=576,independent_metrics_per_seed_means_and_deltas_maximum_abs_error=maximum,all_prediction_protocols_and_Slot_before_evaluation_GT=True,
            qualified_arms_for_separate_full320_confirmation=qualified,selected_arm_for_separate_full320_confirmation=winner,optimizer_updates=0,additional_native_forwards=0,all_GPU_jobs_untouched=True)
        write('summary.json',result);state('complete',raw_masks_independently_rescored=576)
    except BaseException as exc:state('failed_preserved_for_review',error=repr(exc),all_GPU_jobs_untouched=True);raise


if __name__=='__main__':main()
