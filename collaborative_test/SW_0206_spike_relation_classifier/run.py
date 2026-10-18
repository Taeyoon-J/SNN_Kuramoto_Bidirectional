"""Frozen native64 core, TRAIN8 relation classifier, exact paired mask replay."""
import io,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import classifier
from collaborative_test.SW_0205_native64_weak_seed_adaptation import run as source
from collaborative_test.SW_0193_native64_probe_screen import run as screen,score as metrics
from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
train=source.train
HERE=Path(__file__).resolve().parent
ROOT_ARCHIVE=HERE/'results_archive'
ARCHIVE=ROOT_ARCHIVE/'seed2'
ARMS=('prediction',)
OFFSETS=tuple(range(0,32,2))
write_json=source.write_json
sha=source.sha


def configure(seed):
    if type(seed) is not int or seed!=2:raise ValueError('registered seed2 screen only')
    source.configure(seed)


def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('run.py','queue.py','classifier.py','test_classifier.py','test_run.py','DESIGN_KO.md')},
        source=source.fingerprint(),baseline=source.baseline.fingerprint(2),seed=2,TRAIN_images=8,
        pairs_per_TRAIN_image=8192,classifier_hidden=16,classifier_parameters=97,classifier_Adam_steps=200,
        classifier_lr=.001,classifier_seed=206002,core_optimizer_steps=0,offsets=list(OFFSETS))


def prepare_cpu():
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission must precede CUDA')
    configure(2);prepared=source.prepare_inputs_cpu();found=prepared['found']
    old,baseline=source.baseline.validated_pair(2);ref=baseline['reference_binding']
    if source.native._state_sha(found.wrapped,found.encoder)!=baseline['model_sha256']:raise ValueError('wrong independent baseline')
    plans=[classifier.sample_pairs(prepared['labels'][i],seed=206100+i) for i in range(8)]
    from collaborative_test.SW_0130_phase_state_integration import run as data
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False)[:32].copy()
    if sha(rgb.tobytes())!=ref['development_RGB_sha256']:raise ValueError('canonical development RGB changed')
    binding=dict(fingerprint=fingerprint(),model_sha256=baseline['model_sha256'],source_binding=prepared['bindings'],
        reference_binding=ref,image_ids=screen.IDS[list(OFFSETS)].tolist(),
        frozen_old_sha256=[sha(old[i].tobytes()) for i in OFFSETS],
        sample_plan_sha256=[sha(b''.join(v.tobytes() for v in plan)) for plan in plans],
        TRAIN_RGB_row_sha256=[sha(prepared['rgb'][i].tobytes()) for i in range(8)],
        validation_GT_used_for_prediction=False,core_optimizer_steps=0)
    prepared.update(plans=plans,development_rgb=rgb,binding=binding)
    return found,prepared,rgb,binding


def cache_binding(binding,index):
    return dict(binding_sha256=sha(json.dumps(binding,sort_keys=True).encode()),TRAIN_index=index,
        TRAIN_image_id=binding['source_binding']['ordered_train_ids'][index],inference_seed=206100+index,
        plan_sha256=binding['sample_plan_sha256'][index])


def read_cache(folder,expected):
    meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'pairs.npz').read_bytes()
    if meta['binding']!=expected or meta['sha256']!=sha(raw):raise ValueError('frozen TRAIN relation cache changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:x=z['features'].copy();y=z['target'].copy()
    if x.dtype!=np.float32 or x.shape!=(8192,4) or not np.isfinite(x).all() or (np.abs(x)>1).any():raise ValueError('invalid cached spike correlations')
    if y.dtype!=np.float32 or not np.array_equal(y,np.r_[np.ones(4096,np.float32),np.zeros(4096,np.float32)]):raise ValueError('relation targets changed')
    return x,y


def validate_head_payload(payload,binding,cache_shas):
    if payload['binding']!=binding or payload['cache_sha256']!=cache_shas:raise ValueError('classifier data/model binding changed')
    report=payload['fit_report'];state=payload['state_dict']
    steps=sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})
    if steps!=[200] or report['actual_Adam_steps']!=steps or report['parameters']!=97 or report['core_optimizer_steps']!=0:
        raise ValueError('classifier optimizer contract changed')
    if classifier.state_sha(state)!=report['final_parameter_sha256']:raise ValueError('trained classifier parameters changed')
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite classifier parameters')
    if report['seed']!=206002 or report['steps']!=200 or not np.isfinite(report['final_loss']):raise ValueError('classifier training policy changed')
    head=classifier.RelationClassifier();head.load_state_dict(state,strict=True);head.eval();head.requires_grad_(False)
    return head,report


def head_from_cache(binding,*,allow_fit=False):
    rows=[read_cache(ARCHIVE/'TRAIN_pairs'/f'{i:04d}',cache_binding(binding,i)) for i in range(8)]
    shas=[sha((ARCHIVE/'TRAIN_pairs'/f'{i:04d}'/'row.json').read_bytes()) for i in range(8)]
    path=ARCHIVE/'classifier_v1/state.ptx'
    if path.exists():
        outer=torch.load(io.BytesIO(path.read_bytes()),map_location='cpu',weights_only=False)
        if sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('classifier checkpoint corrupt')
        payload=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
    else:
        if not allow_fit:raise FileNotFoundError('validation must not retrain a missing classifier')
        head,optimizer,report=classifier.fit(np.concatenate([x for x,y in rows]),np.concatenate([y for x,y in rows]))
        payload=dict(binding=binding,cache_sha256=shas,state_dict=head.state_dict(),optimizer_state=optimizer,fit_report=report)
        data=io.BytesIO();torch.save(payload,data);raw=data.getvalue()
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{time.time_ns()}.tmp')
        with temporary.open('xb') as stream:
            torch.save(dict(payload_bytes=raw,payload_sha256=sha(raw)),stream);stream.flush();os.fsync(stream.fileno())
        if path.exists():raise FileExistsError('preserve existing classifier')
        temporary.rename(path)
    head,report=validate_head_payload(payload,binding,shas)
    if report['pair_samples']!=65536 or report['positive_pairs']!=32768:raise ValueError('classifier pair exposure changed')
    public=dict(report,binding=binding,cache_sha256=shas,unique_TRAIN_images=8,classifier_checkpoint_sha256=sha(path.read_bytes()))
    target=ARCHIVE/'classifier_v1/report.json'
    if target.exists():
        if json.loads(target.read_bytes())!=public:raise ValueError('classifier report changed')
    else:write_json(target,public)
    return head,public


def read_row(folder,binding,index,head_sha):
    meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'masks.npz').read_bytes()
    if meta['binding']!=binding or meta['index']!=index or meta['classifier_sha256']!=head_sha or meta['sha256']!=sha(raw):raise ValueError('frozen candidate row changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:row={n:z[n].copy() for n in ('old','new')}
    if any(v.dtype!=np.int64 or v.shape!=(64,64) or (v<0).any() for v in row.values()):raise ValueError('invalid native64 masks')
    if sha(row['old'].tobytes())!=binding['frozen_old_sha256'][index] or not np.array_equal(row['old']>0,row['new']>0):raise ValueError('baseline replay or foreground support changed')
    return row


def predict(expected_binding=None):
    found,prepared,rgb,binding=prepare_cpu()
    if binding!=expected_binding:raise ValueError('CPU admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    source.model64.move_to_device(found,device)
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    for index in range(8):
        folder=ARCHIVE/'TRAIN_pairs'/f'{index:04d}';expected=cache_binding(binding,index)
        if folder.exists():read_cache(folder,expected);continue
        torch.manual_seed(206100+index);torch.cuda.manual_seed(206100+index)
        with torch.no_grad():
            image=torch.from_numpy(np.ascontiguousarray(prepared['rgb'][index:index+1])).to(device)
            gamma,trace,adj,z,q,rho=source.native._native_forward(found,image,live_tail_steps=0)
            i,j,target=prepared['plans'][index];a=torch.from_numpy(i).to(device);b=torch.from_numpy(j).to(device)
            features=(.5*(rho[0,:,a,b]+rho[0,:,b,a])).T.contiguous().cpu().numpy()
        stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True)
        with (stage/'pairs.npz').open('xb') as stream:
            np.savez_compressed(stream,features=features,target=target);stream.flush();os.fsync(stream.fileno())
        write_json(stage/'row.json',dict(binding=expected,sha256=sha((stage/'pairs.npz').read_bytes())))
        read_cache(stage,expected);stage.rename(folder);del image,gamma,trace,adj,z,q,rho,a,b,features
    if source.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256']:raise ValueError('TRAIN extraction mutated core')
    head,report=head_from_cache(binding,allow_fit=True);head.to(device);head_sha=report['final_parameter_sha256']
    for index,offset in enumerate(OFFSETS):
        folder=ARCHIVE/'prediction/rows'/f'{index:04d}'
        if folder.exists():read_row(folder,binding,index,head_sha);continue
        torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
        with torch.no_grad():
            image=torch.from_numpy(rgb[offset:offset+1]).to(device)
            gamma,trace,adj,z,q,rho=source.native._native_forward(found,image,live_tail_steps=0)
            old=screen.prediction_from_trace(found,trace,adj,q)
            if sha(old.tobytes())!=binding['frozen_old_sha256'][index]:raise ValueError('original mask did not replay exactly')
            learned=classifier.predict_affinity(head,rho[0]).cpu().numpy().astype(np.float64)
            gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
            new=np.asarray(foreground_induced_merge(learned,adj[0].cpu().numpy().astype(np.float64),gain,old,grid_shape=(64,64))['labels'],dtype=np.int64)
        stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True)
        with (stage/'masks.npz').open('xb') as stream:
            np.savez_compressed(stream,old=old,new=new);stream.flush();os.fsync(stream.fileno())
        write_json(stage/'row.json',dict(binding=binding,index=index,classifier_sha256=head_sha,sha256=sha((stage/'masks.npz').read_bytes())))
        read_row(stage,binding,index,head_sha);stage.rename(folder);del image,gamma,trace,adj,z,q,rho,learned
    if source.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256'] or classifier.state_sha(head.state_dict())!=head_sha:raise ValueError('inference mutated model/head')
    write_json(ARCHIVE/'prediction/protocol.json',dict(binding=binding,classifier_sha256=head_sha,classifier_checkpoint_sha256=report['classifier_checkpoint_sha256'],rows_sha256=[sha((ARCHIVE/'prediction/rows'/f'{i:04d}'/'row.json').read_bytes()) for i in range(16)]))


def validated_pair():
    p=json.loads((ARCHIVE/'prediction/protocol.json').read_bytes());binding=p['binding']
    if binding['fingerprint']!=fingerprint() or binding['validation_GT_used_for_prediction'] is not False:raise ValueError('candidate protocol changed')
    head,report=head_from_cache(binding)
    if classifier.state_sha(head.state_dict())!=p['classifier_sha256'] or report['classifier_checkpoint_sha256']!=p['classifier_checkpoint_sha256']:raise ValueError('frozen classifier changed')
    rows=[]
    for i in range(16):
        folder=ARCHIVE/'prediction/rows'/f'{i:04d}'
        if sha((folder/'row.json').read_bytes())!=p['rows_sha256'][i]:raise ValueError('prediction metadata changed')
        rows.append(read_row(folder,binding,i,p['classifier_sha256']))
    return {n:np.stack([r[n] for r in rows]) for n in ('old','new')},p


def score():
    labels,protocol=validated_pair();binding=protocol['binding'];old,source_binding=source.baseline.validated_pair(2)
    if source_binding['reference_binding']!=binding['reference_binding'] or not np.array_equal(labels['old'],old[list(OFFSETS)]):raise ValueError('baseline changed')
    gt=metrics.read_ground_truth(binding['reference_binding'])[list(OFFSETS)];vectors={};maximum=0.
    for name,pred in labels.items():vectors[name],error=metrics.measured_vectors(pred,gt);maximum=max(maximum,error)
    if not np.array_equal(vectors['old'][:,1],vectors['new'][:,1]):raise ValueError('foreground IoU changed')
    from collaborative_test.SW_0205_native64_weak_seed_adaptation.aggregate import paired
    delta=paired(vectors['new']-vectors['old'])
    summary=dict(status='completed_native64_learned_spike_relation_classifier_screen',seed_scope=[2],images=16,grid=[64,64],
        scores={n:dict(zip(metrics.independent.METRICS,v.mean(0).tolist())) for n,v in vectors.items()},paired_new_minus_old=delta,
        extension_gate_passed=bool(delta['fg_ari']['mean']>=.01 and delta['fg_ari']['ci95'][0]>0 and delta['matched_object_iou']['mean']>=0),
        classifier_sha256=protocol['classifier_sha256'],classifier_checkpoint_sha256=protocol['classifier_checkpoint_sha256'],
        source_model_sha256=binding['model_sha256'],core_optimizer_steps=0,classifier_optimizer_steps=200,
        unique_TRAIN_images_for_classifier=8,foreground_support_identical=True,mask_supervised=True,
        unsupervised_leaderboard_eligible=False,causal_gate_contribution_proved=False,full320_three_seed_goal_verified=False,
        independent_maximum_metric_abs_error=maximum,raw_imagewise_scores_public=False)
    write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={n:v.tolist() for n,v in vectors.items()}))
    write_json(ARCHIVE/'score_v1/summary.json',summary);return summary
