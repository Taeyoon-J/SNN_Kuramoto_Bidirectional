"""Read-only decomposition of frozen effective frequency; no model forwards."""
import ctypes,sys
if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP admission failed')
import io,json,math
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0223_native64_joint_sensory_projection import train,evaluate


def decomposition(frequency,omega,gain,labels):
    x=np.asarray(frequency,dtype=np.float64);o=np.asarray(omega,dtype=np.float64)/(abs(float(gain))*math.pi)
    labels=np.asarray(labels)
    if not np.isfinite(gain) or abs(gain)<=1e-6 or x.ndim!=3 or o.shape!=x.shape[1:] or labels.shape!=(len(x),64,64):raise ValueError('aligned native64 finite inputs required')
    if not np.isfinite(x).all() or not np.isfinite(o).all():raise ValueError('nonfinite frequency')
    d=x-o;rows=[]
    for f,drive,grid in zip(x,d,labels):
        entries=[];flat=grid.reshape(-1)
        for k in np.unique(flat):
            take=flat==k
            a=f[take]-f[take].mean(0);b=o[take]-o[take].mean(0);c=drive[take]-drive[take].mean(0)
            entries.append([np.mean(a*a),np.mean(b*b),np.mean(c*c),2*np.mean(b*c)])
        rows.append(np.mean(entries,axis=0))
    values=np.asarray(rows)
    error=float(np.max(np.abs(values[:,0]-values[:,1:].sum(1))))
    if error>1e-12:raise ValueError('variance identity failed')
    return dict(zip(('effective_within','static_within','sensory_within','twice_cross_covariance'),values.mean(0).tolist())),error


def run():
    a=train.ARCHIVE;score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw)
    audit=json.loads((a/'completion_audit_v1/summary.json').read_bytes())
    if audit['score_sha256']!=train.sha(score_raw) or score['fingerprint']!=train.fingerprint():raise ValueError('accepted source audit changed')
    records=[];dataset=None
    root=Path('/Data0/kevinswk/patch_v2_sw/collaborative_test')
    for seed in train.SEEDS:
        p=train.archive_for(seed,64);report_raw=(p/'training_report.json').read_bytes();report=json.loads(report_raw);protocol=json.loads((p/'protocol.json').read_bytes())
        proof=audit['checkpoints'][str(seed)]
        if report['checkpoint_file_sha256']!=proof['checkpoint_sha256'] or train.sha(report_raw)!=protocol['binding']['training_report_sha256'] or train.sha((p/'protocol.json').read_bytes())!=proof['protocol_sha256']:raise ValueError('accepted prediction identity changed')
        source=root/('SW_0192_native64_incumbent_mini/results_archive/probe_work_seed0/boundary.ptx' if seed==0 else f'SW_0205_native64_weak_seed_adaptation/results_archive/seed{seed}/training_work/boundary.ptx')
        raw=source.read_bytes();expected=('14d0386db64d3890163574ac2421b1e80d3100536ec25c9940434e0689327e82','9ef50f95559ca47f0f8f87e1cf7d78e3f8a44c5fb51445b3acd03bf7a9b24e85','b88668e8cac21ca1b40d65660051443f6ec1881e623a7ee368fb745726f3525c')[seed]
        if train.sha(raw)!=expected:raise ValueError('source checkpoint changed')
        outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if train.sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('source payload changed')
        state=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)['wrapped_state']
        omega=state['core.kuramoto.omega'].numpy();gain=float(state['core.kuramoto.freq_gain'])
        reference_raw=(p/'reference_frequency.npz').read_bytes()
        if train.sha(reference_raw)!=report['reference_frequency_file_sha256']:raise ValueError('reference frequency changed')
        with np.load(io.BytesIO(reference_raw),allow_pickle=False) as values:reference=values['frequency'].copy()
        rows=[evaluate.read_row(p/'rows'/f'{i:04d}',protocol['binding'],i) for i in range(16)]
        if train.sha(np.stack([v[0] for v in rows]).tobytes())!=protocol['labels_sha256']:raise ValueError('accepted masks changed')
        current=report['binding']['parent']['dataset']
        if dataset is None:dataset=current
        elif dataset!=current:raise ValueError('dataset mismatch')
        records.append((seed,omega,gain,reference,np.stack([v[1] for v in rows]),expected))
    # All three accepted prediction protocols and source tensors checked first.
    gt=train.source.source.full.read_gt(dict(dataset=dataset))[list(evaluate.OFFSETS)]
    result=[]
    for seed,omega,gain,before,after,source_sha in records:
        old,error=decomposition(before,omega,gain,gt);new,error2=decomposition(after,omega,gain,gt)
        result.append(dict(seed=seed,source_checkpoint_sha256=source_sha,gain=gain,before=old,after=new,variance_identity_maximum_abs_error=max(error,error2)))
    summary=dict(status='complete_read_only_frequency_component_diagnosis',accepted_score_sha256=train.sha(score_raw),rows=result,
        images=16,seeds=[0,1,2],additional_model_forwards=0,additional_training_updates=0,
        static_over_total_is_not_causal_attribution=True,not_a_gating_ablation=True,not_a_full320_or_data_scaling_result=True,
        diagnostic_source_sha256=train.sha(Path(__file__).read_bytes()))
    train.write_json(Path(__file__).parent/'results_archive/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(run()))
