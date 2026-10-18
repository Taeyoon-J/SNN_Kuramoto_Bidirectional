"""TRAIN-only center/offset learning diagnosis plus unlabeled DEV outputs."""
import json
from pathlib import Path
import numpy as np
import torch
from . import run as api
from .center_head import targets,decode,loss


def run():
    torch.set_num_threads(1)
    arrays,protocol=api.cache.load_arrays();folder=api.checkpoint_for('step128')
    record=json.loads((folder/'report.json').read_bytes())
    payload=api.cache.source.full.unpack_checkpoint((folder/'boundary.ptx').read_bytes())
    assert payload['completed_updates']==128
    head=api.CenterHead();head.load_state_dict(payload['wrapped_state'],strict=True);head.eval();head.requires_grad_(False)
    model=api.state_sha(head.state_dict());assert model==record['model_sha256']
    by_arm={};forwards=0
    for arm in ('real','mean_input','moments_only'):
        rows=[]
        for index,features in enumerate(arrays['TRAIN_features']):
            x=torch.from_numpy(features[None])
            if arm!='real':x=api.head_input(x,head,arm)
            with torch.no_grad():out=head(x)
            forwards+=1;t=targets(arrays['TRAIN_labels'][index]);prob=out['heat'][0].sigmoid().numpy()
            delta=(out['offset'][0].numpy()-t['offset'])*63
            _,info=decode(out['heat'][0].numpy(),out['offset'][0].numpy(),arrays['TRAIN_baseline'][index])
            centers=t['centers'];fg=(out['foreground'][0].numpy()>=0);truth=t['foreground']>0
            rows.append(dict(offset_error_object_balanced_px=float((np.sqrt((delta**2).sum(0))*t['offset_weights']).sum()),
                zero_offset_error_object_balanced_px=float((np.sqrt(((t['offset']*63)**2).sum(0))*t['offset_weights']).sum()),
                center_probability_at_GT_centers=float(prob[centers[:,0],centers[:,1]].mean()),
                GT_center_fraction_reaching_fixed_threshold=float((prob[centers[:,0],centers[:,1]]>=.3).mean()),
                max_center_probability=float(prob.max()),fallback=float(info['fallback']),predicted_objects=info['predicted_objects'],
                foreground_fraction=float(fg.mean()),foreground_IoU=float((fg&truth).sum()/max((fg|truth).sum(),1))))
        by_arm[arm]={key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
    dev=[]
    for features in arrays['DEV_features']:
        with torch.no_grad():out=head(torch.from_numpy(features[None]))
        forwards+=1
        dev.append(dict(max_center_probability=float(out['heat'].sigmoid().max()),
                        foreground_fraction=float((out['foreground']>=0).float().mean())))
    assert api.state_sha(head.state_dict())==model
    summary=dict(status='posthoc_TRAIN_center_offset_diagnosis_DEV_outputs_without_GT',TRAIN_by_frozen_input_arm=by_arm,
        DEV_unlabeled_mean={key:float(np.mean([row[key] for row in dev])) for key in dev[0]},
        new_core_forward_passes=0,new_optimizer_steps=0,new_head_image_forward_passes=forwards,
        TRAIN_images=32,DEV_images=16,DEV_GT_read=False,source_model_sha256=model,
        source_sha256=api.sha(Path(__file__).read_bytes()),not_a_new_leaderboard_result=True,
        ablations_are_frozen_input_controls_not_retrained_physical_gating_causality=True)
    api.write_json(api.ARCHIVE/'head_diagnosis_v1/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(run()))
