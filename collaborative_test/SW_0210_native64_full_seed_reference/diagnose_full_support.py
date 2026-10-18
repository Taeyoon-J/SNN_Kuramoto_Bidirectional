"""Posthoc attribution of the frozen three-seed reference; no model forward."""
import json
from pathlib import Path
from . import run as api
from collaborative_test.SW_0208_native64_full320_confirmation.diagnose_fixed_support import analyze


def run():
    source=api.ARCHIVE/'score_v1/summary.json';summary=json.loads(source.read_bytes())
    if summary['fingerprint']!=api.fingerprint():raise ValueError('original reference source changed')
    original,b0=api.full.validated_all();masks={0:original['warm192']}
    for seed in api.SEEDS:
        masks[seed],binding=api.validated_seed(seed)
        if binding['dataset']!=b0['dataset'] or binding['RGB_sha256']!=b0['RGB_sha256']:raise ValueError('different full320 inputs')
    # Existing GT-evaluated predictions have all been validated first.
    gt=api.full.read_gt(b0);results={}
    for seed,labels in masks.items():
        result=analyze(labels,gt)
        if abs(result['actual_FG_ARI']-summary['scores']['seed'+str(seed)]['fg_ari'])>1e-12:raise ValueError('different actual frozen masks')
        if abs(result['actual_foreground_IoU']-summary['scores']['seed'+str(seed)]['foreground_iou'])>1e-12:raise ValueError('different foreground support')
        results[str(seed)]=result
    previous=api.full.ARCHIVE/'support_diagnosis_v1/summary.json'
    if results['0']!=json.loads(previous.read_bytes())['warm192_seed0_full320']:raise ValueError('known seed0 diagnosis did not exactly replay')
    helper=Path(__file__).resolve().parents[1]/'SW_0208_native64_full320_confirmation/diagnose_fixed_support.py'
    result=dict(status='posthoc_full320_three_seed_fixed_support_attribution',by_seed=results,
        actual_three_seed_FG_ARI=sum(v['actual_FG_ARI'] for v in results.values())/3,
        source_summary_sha256=api.sha(source.read_bytes()),source_sha256=api.sha(Path(__file__).read_bytes()),
        diagnostic_helper_sha256=api.sha(helper.read_bytes()),seed0_prior_diagnosis_replayed_exactly=True,
        new_model_forward_passes=0,new_training_updates=0,uses_GT=True,not_a_new_leaderboard_result=True,
        reference_scope='Concrete GT-assisted partitions/attachments, not predictors or a certified learnable performance ceiling; effects are nonadditive.')
    if abs(result['actual_three_seed_FG_ARI']-summary['scores']['our_three_seed_mean']['fg_ari'])>1e-12:raise ValueError('seed mean changed')
    api.write_json(api.ARCHIVE/'support_diagnosis_v1/summary.json',result);return result


if __name__=='__main__':print(json.dumps(run()))
