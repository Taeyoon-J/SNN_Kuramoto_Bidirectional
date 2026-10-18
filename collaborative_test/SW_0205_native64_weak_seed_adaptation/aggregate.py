"""Audit real TRAIN8 boundaries and pair the same adaptation recipe across seeds."""
import inspect,io,json
from pathlib import Path
import numpy as np
import torch
from . import run as api,evaluate
from collaborative_test.SW_0179_qce_seed_replication import train as state_api


def paired(delta):
    if delta.shape!=(16,3) or not np.isfinite(delta).all():raise ValueError('fixed16 finite paired metrics required')
    draws=np.random.RandomState(205).randint(0,16,(10000,16))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    return {m:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,m in enumerate(api.metrics.independent.METRICS)}


def boundary(path):
    raw=path.read_bytes();outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if api.sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('checkpoint payload corrupt')
    value=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
    model=state_api._state_sha_from_payload(dict(wrapped=value['wrapped_state'],encoder=value['encoder_state']))
    steps=sorted({int(s['step'].item()) for s in value['optimizer_state']['state'].values()})
    return value,model,steps,api.sha(raw)


def run():
    # Seed0 reuses actual completed192 TRAIN8; the two newly trained seeds remain independent.
    for name in ('batch_loss','run_probe'):
        if inspect.getsource(getattr(api,name))!=inspect.getsource(getattr(api.train,name)):
            raise ValueError('adaptation algorithm differs from the reused seed0 recipe')
    if (api.BATCH,api.PROPOSALS)!=(api.train.BATCH,api.train.PROPOSALS):raise ValueError('training budget differs')
    frozen,ref=api.metrics.validated_pair();labels={0:dict(baseline=frozen['baseline'][list(evaluate.OFFSETS)],candidate=frozen['candidate'][list(evaluate.OFFSETS)])}
    value,model,steps,sha=boundary(api.train.ARCHIVE/'probe_work_seed0/boundary.ptx')
    if value['bindings']['fingerprint']!=api.train.fingerprint() or model!=ref['candidate_model_sha256'] or value['completed_updates']!=2:
        raise ValueError('reused source seed0 boundary changed')
    histories={0:value['history']};bindings={0:value['bindings']}
    audits={'0':dict(model_sha256=model,boundary_sha256=sha,actual_Adam_steps=steps,reused_completed_TRAIN8=True)}
    reports={};protocols={}
    for seed in (1,2):
        api.configure(seed);candidate,protocol=evaluate.validated_candidate();binding=protocol['binding']
        source,source_binding=api.baseline.validated_pair(seed)
        if binding['reference_binding']!=ref or source_binding['reference_binding']!=ref:raise ValueError('seed cohorts differ')
        prepared=api.prepare_inputs_cpu()
        if prepared['bindings']!=binding:raise ValueError('CPU initial-model admission changed')
        snapshot=api.tx.capture_transaction({'wrapped':prepared['found'].wrapped,'encoder':prepared['found'].encoder},prepared['optimizer'],offload_to_cpu=True)
        initial=api.tx.transaction_digest(snapshot);del prepared,snapshot
        value,model,steps,sha=boundary(api.ARCHIVE/'training_work/boundary.ptx')
        report=json.loads((api.ARCHIVE/'training/report.json').read_bytes())
        api.validate_report(report,binding)
        if (value['bindings']!=binding or value['history']!=report['history'] or value['completed_updates']!=len(report['history'])
            or model!=report['final_model_state_sha256'] or model!=protocol['model_sha256'] or steps!=report['actual_optimizer_steps']
            or initial!=report['history'][0]['start_transaction_sha256']):raise ValueError('actual model/Adam/history or initial tensor transaction changed')
        cnn=api.warm._base_encoder_sha_from_state(value['encoder_state'])
        if report['source_CNN_changed']!=bool(cnn!=binding['factory_proof']['source_CNN_sha256']):raise ValueError('actual CNN learning differs from report')
        for key in ('ordered_train_ids','TRAIN64_modal_sha256','source_dataset_arrays_sha256'):
            if binding[key]!=bindings[0][key]:raise ValueError('adaptation data differs from seed0')
        labels[seed]=dict(baseline=source[list(evaluate.OFFSETS)],candidate=candidate)
        histories[seed]=report['history'];reports[seed]=report
        audits[str(seed)]=dict(model_sha256=model,boundary_sha256=sha,actual_Adam_steps=steps,
            initial_tensor_transaction_verified=True,RNG_not_in_transaction_digest=True,CNN_changed=report['source_CNN_changed'])
        protocols[str(seed)]=api.sha((api.ARCHIVE/'candidate/protocol.json').read_bytes())
    # All six mask arrays and actual model boundaries precede GT scoring.
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots={s:api.metrics.slot._load_slot(SLOT_ROOT,s) for s in (0,1,2)}
    if any(not np.array_equal(v['ids'][:32],evaluate.screen.IDS) for v in slots.values()):raise ValueError('Slot IDs differ')
    gt=api.metrics.read_ground_truth(ref)[list(evaluate.OFFSETS)];vectors={};maximum=0.
    names=api.metrics.independent.METRICS
    for seed,variants in labels.items():
        for mode,pred in variants.items():
            vectors[f'seed{seed}_{mode}'],error=api.metrics.measured_vectors(pred,gt);maximum=max(maximum,error)
        vectors[f'slot_seed{seed}'],error=api.metrics.measured_vectors(slots[seed]['labels64'][list(evaluate.OFFSETS)],gt);maximum=max(maximum,error)
        if seed in reports:
            local=json.loads((api.ROOT_ARCHIVE/f'seed{seed}/score_v1/summary.json').read_bytes())
            scores={m:dict(zip(names,vectors[f'seed{seed}_{m}'].mean(0).tolist())) for m in variants}
            if local['scores']!=scores or local['paired_candidate_minus_baseline']!=paired(vectors[f'seed{seed}_candidate']-vectors[f'seed{seed}_baseline']):raise ValueError('individual scores did not independently replay')
    for mode in ('baseline','candidate'):
        vectors[mode+'_three_seed_mean']=np.mean([vectors[f'seed{s}_{mode}'] for s in (0,1,2)],axis=0)
    vectors['slot_three_seed_mean']=np.mean([vectors[f'slot_seed{s}'] for s in (0,1,2)],axis=0)
    change=paired(vectors['candidate_three_seed_mean']-vectors['baseline_three_seed_mean'])
    healthy=all(len(histories[s])==2 and not any(r['credit_failures'] for r in histories[s]) and reports[s]['accepted_updates']>0 and reports[s]['source_CNN_changed'] for s in (1,2))
    summary=dict(status='three_independent_seed_TRAIN8_adaptation_fixed16_audited',images_per_seed=16,seed_scope=[0,1,2],grid=[64,64],
        scores={k:dict(zip(names,v.mean(0).tolist())) for k,v in vectors.items()},paired_candidate_minus_baseline=change,
        paired_candidate_minus_Slot=paired(vectors['candidate_three_seed_mean']-vectors['slot_three_seed_mean']),
        extension_gate_passed=bool(healthy and change['fg_ari']['mean']>=.01 and change['fg_ari']['ci95'][0]>0 and all(change[k]['mean']>=0 for k in ('foreground_iou','matched_object_iou'))),
        actual_boundaries=audits,newly_trained_seeds=[1,2],seed0_reused_completed_SW0192=True,
        independent_maximum_metric_abs_error=maximum,protocol_sha256=protocols,
        source_sha256=api.sha(Path(__file__).read_bytes()),bootstrap_scope='paired images conditional on three fixed seed lineages',
        raw_imagewise_scores_public=False,mask_supervised=True,unsupervised_leaderboard_eligible=False,full320_three_seed_goal_verified=False)
    for name,val in [('summary.json',summary),('evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))]:
        path=api.ROOT_ARCHIVE/'aggregate_v1'/name
        if path.exists():
            if json.loads(path.read_bytes())!=val:raise ValueError('preserve frozen aggregate')
        else:api.write_json(path,val)
    print(json.dumps(summary));return summary


if __name__=='__main__':run()
