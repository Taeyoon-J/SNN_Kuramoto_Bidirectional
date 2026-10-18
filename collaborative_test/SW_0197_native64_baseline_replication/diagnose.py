"""Aggregate attribution from frozen native64 baseline masks, no new forward."""
import json
from pathlib import Path
import numpy as np
from . import run as api
from collaborative_test.SW_0189_frozen_error_attribution.run import errors
from collaborative_test.SW_0194_native64_train64.diagnose_partition import paired_summary,KEYS

REFERENCE='e547d5795a82e8dbbc3113938eea225470060a98b40fbf806bb4242111298f88'


def run():
    path=api.ROOT_ARCHIVE/'aggregate_v1/summary.json'
    if api.train.sha(path.read_bytes())!=REFERENCE:raise ValueError('completed baseline summary changed')
    controls,ref=api.metrics.validated_pair();predictions={'seed0':controls['baseline']}
    protocols={'seed0':api.train.sha((api.screen.ARCHIVE/'baseline/protocol.json').read_bytes())}
    for seed in (1,2):
        labels,binding=api.validated_pair(seed)
        if binding['reference_binding']!=ref:raise ValueError('baseline cohorts differ')
        predictions['seed'+str(seed)]=labels
        protocols['seed'+str(seed)]=api.train.sha((api.archive_for(seed)/'prediction/protocol.json').read_bytes())
    gt=api.metrics.read_ground_truth(ref)
    if gt.shape!=(32,64,64):raise ValueError('fixed32 native64 cohort required')
    rows={key:[errors(p,g) for p,g in zip(labels,gt)] for key,labels in predictions.items()}
    keys=KEYS+('foreground_precision','true_object_count','predicted_positive_label_count','raw_count_exact')
    means={name:{key:float(np.mean([r[key] for r in values if r[key] is not None]))
        if any(r[key] is not None for r in values) else None for key in keys} for name,values in rows.items()}
    from collaborative_test.SW_0189_frozen_error_attribution import run as operator
    from collaborative_test.SW_0194_native64_train64 import diagnose_partition as paired
    report=dict(status='completed_three_seed_baseline_error_attribution',grid=[64,64],images=32,seed_scope=[0,1,2],
        per_seed_means=means,paired_seed_minus_seed0={name:paired_summary(rows['seed0'],rows[name],keys) for name in ('seed1','seed2')},
        summary_sha256=REFERENCE,protocol_sha256=protocols,
        error_operator_sha256=api.train.sha(Path(operator.__file__).read_bytes()),
        paired_operator_sha256=api.train.sha(Path(paired.__file__).read_bytes()),
        analysis_source_sha256=api.train.sha(Path(__file__).read_bytes()),
        raw_positive_label_count_is_not_a_trained_count_classifier=True,pair_rates_are_not_additive_error_shares=True,
        bootstrap_scope='fixed trained models, paired development images; not population variation across training seeds',
        causal_claim=False,new_model_forward_passes=0,optimizer_updates=0,raw_imagewise_values_included=False)
    out=api.ROOT_ARCHIVE/'error_diagnosis_v1/summary.json'
    if out.exists():
        if json.loads(out.read_bytes())!=report:raise ValueError('preserve different frozen diagnosis')
    else:api.train.write_json(out,report)
    return report


if __name__=='__main__':print(json.dumps(run()),flush=True)
