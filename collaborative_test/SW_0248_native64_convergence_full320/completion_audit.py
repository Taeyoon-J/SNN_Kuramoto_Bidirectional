"""CPU-only post-completion model restoration and independent raw-mask audit."""
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
    here=Path(__file__).resolve().parent;root=here.parents[1];sys.path.insert(0,str(root));a=here/'results_archive';out=a/'completion_audit_v1';out.mkdir(exist_ok=True);os.chdir(root)
    def state(status,**fields):
        p=out/'state.tmp';p.write_bytes(json.dumps(dict(status=status,pid=os.getpid(),updated_unix=time.time(),**fields)).encode());p.replace(out/'state.json')
    if (out/'state.json').exists():raise FileExistsError('existing full320 auditor must be preserved')
    try:
        deadline=time.monotonic()+7*86400
        while not (a/'score_v1/summary.json').exists():
            for seed in (0,1,2):
                path=a/f'seed{seed}/queue/state.json'
                if path.exists() and json.loads(path.read_bytes())['status']=='failed':raise RuntimeError('preserved full320 queue failed')
            if time.monotonic()>deadline:raise TimeoutError('observation deadline, experiment jobs untouched')
            state('waiting_for_full320_score');time.sleep(45)
        for seed in (0,1,2):
            state('actual_trained_model_CPU_restoration',seed=seed)
            with (out/f'seed{seed}.stdout').open('xb') as stdout,(out/f'seed{seed}.stderr').open('xb') as stderr:
                subprocess.run([sys.executable,'-B','-m','collaborative_test.SW_0248_native64_convergence_full320.run','--seed',str(seed),'--source-check'],cwd=root,env=os.environ,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,check=True,timeout=1200)
        import numpy as np
        from collaborative_test.SW_0248_native64_convergence_full320 import run,evaluate
        raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(raw)
        if score['fingerprint']!=run.fingerprint():raise ValueError('completed full320 source changed')
        bundles=evaluate.validated_all();slots=run.training.reference_score.load_slot_masks();accepted=evaluate.references()
        datasets=[bundles[s]['protocol']['binding']['dataset'] for s in run.SEEDS]
        if any(d!=datasets[0] for d in datasets):raise ValueError('seed dataset identity changed')
        gt=run.reference.source.source.full.read_gt(dict(dataset=datasets[0]));vectors={k:[] for k in ('candidate','parent237','slot')};maximum=0.
        for seed in run.SEEDS:
            for name,labels in (('candidate',bundles[seed]['labels']),('parent237',bundles[seed]['baseline']),('slot',slots[seed]['labels64'])):
                value,error=run.reference.source.source.full.metrics.measured_vectors(labels,gt);vectors[name].append(value);maximum=max(maximum,float(error))
        private=json.loads((a/'score_v1/evaluation_private.json').read_bytes());vectors={k:np.stack(v) for k,v in vectors.items()};means={k:v.mean((0,1)) for k,v in vectors.items()}
        maximum=max(maximum,evaluate.verify_reference_means(vectors,accepted))
        for name,values in vectors.items():
            maximum=max(maximum,float(np.abs(values-np.asarray(private['vectors'][name])).max()),float(np.abs(means[name]-np.array([score['scores'][name][k] for k in evaluate.METRICS])).max()))
            for seed in run.SEEDS:maximum=max(maximum,float(np.abs(values[seed].mean(0)-np.array([score['per_seed'][name][str(seed)][k] for k in evaluate.METRICS])).max()))
        delta=means['candidate']-means['parent237'];maximum=max(maximum,float(np.abs(delta-np.array([score['candidate_minus_parent237'][k] for k in evaluate.METRICS])).max()))
        strict=bool(np.all(means['candidate']>means['slot']))
        if maximum>=1e-12 or strict!=score['three_seed_strictly_exceeds_Slot_means'] or bool(strict and means['candidate'][0]>=.90)!=score['native64_FG090_three_seed_goal_verified']:raise ValueError('independent metrics or promotion flags differ')
        result=dict(status='completed_full320_actual_model_restoration_and_raw_mask_audit',fingerprint=run.fingerprint(),score_sha256=run.sha(raw),audit_code_sha256=run.sha(Path(__file__).read_bytes()),
            actual_final_trained_models_restored_on_CPU=3,actual_source_Adam_checkpoints_audited_in247=6,raw_masks_independently_rescored=2880,
            independent_per_seed_means_and_deltas_maximum_abs_error=maximum,exact_mask_and_frequency_replays_validated=3,
            three_seed_strictly_exceeds_Slot_means=strict,native64_FG090_three_seed_goal_verified=bool(strict and means['candidate'][0]>=.90),
            optimizer_updates=0,additional_native_forwards=0,all_prediction_protocols_and_Slot_before_evaluation_GT=True,all_GPU_jobs_untouched=True)
        run.write_json(out/'summary.json',result);state('complete')
    except BaseException as exc:state('failed_preserved_for_review',error=repr(exc),all_GPU_jobs_untouched=True);raise


if __name__=='__main__':main()
