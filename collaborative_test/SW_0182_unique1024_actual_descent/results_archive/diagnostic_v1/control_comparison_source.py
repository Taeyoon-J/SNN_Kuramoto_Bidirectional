import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from collaborative_test.SW_0182_unique1024_actual_descent import train

private = Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0182_delivery_20261016')
path = private / 'proposal2_replay_seed1_attempt01.json'
raw = path.read_bytes()
report = json.loads(raw)
control = train._control_prefix(1)
archived = control['history'][1]
current = report['replays']['control512_A']
old_runtime = report['SW0181_control_runtime_signature']
new_runtime = report['candidate_boundary_runtime_signature']
runtime_differences = {key: dict(archived=old_runtime.get(key), current=new_runtime.get(key))
    for key in set(old_runtime) | set(new_runtime)
    if old_runtime.get(key) != new_runtime.get(key)
    and key not in {'sw0182_proposal_cursor_semantics','proposal_cursor','accepted_updates'}}
gradient_differences, delta_differences = [], []
for family, rows in archived['named_parameter_credit'].items():
    for row in rows:
        name = family + '/' + row['name']
        current_gradient = current['preclip_gradient_summary'][name]['l2']
        if current_gradient != row['gradient_l2']:
            gradient_differences.append(dict(name=name, archived=row['gradient_l2'], current=current_gradient,
                difference=current_gradient-row['gradient_l2']))
        current_delta = next(x['delta_l2'] for x in current['parameter_deltas'][family] if x['name']==name)
        if current_delta != row['parameter_delta_l2']:
            delta_differences.append(dict(name=name, archived=row['parameter_delta_l2'], current=current_delta,
                difference=current_delta-row['parameter_delta_l2']))
result = dict(status='SW0182_current_replay_vs_archived_control_comparison_complete',
    report_sha256=hashlib.sha256(raw).hexdigest(), control_history_sha256=control['history_sha256'],
    runtime_differences=runtime_differences,
    microbatch_q_z_hashes_exact=current['microbatch_q_z_sha256']==archived['microbatches'],
    gradient_norm_differences=gradient_differences, parameter_delta_norm_differences=delta_differences,
    archived_before_loss=archived['before_loss'], archived_after_loss=archived['after_loss'],
    archived_accepted=archived['accepted'], current_before_loss=current['before_loss_train_update'],
    current_after_loss=current['after_loss_full1024'], current_accepted=current['accepted_by_registered_rule'],
    archived_proposal_sha256=archived['proposal_transaction_sha256'], current_proposal_sha256=current['proposed_live_sha256'],
    validation_data_read=False, optimizer_updates=0)
with (private/'proposal2_control_comparison.json').open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
