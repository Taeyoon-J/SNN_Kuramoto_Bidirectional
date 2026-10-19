"""Protocol order and benchmark alignment are required before promotion."""
from unittest import mock
import numpy as np
import pytest
from . import evaluate,run


def test_missing_seed_prevents_GT_and_missing_Slot_prevents_GT():
    with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(run.reference.source.source.full,'read_gt') as gt:
        with pytest.raises(FileNotFoundError):run.score()
        gt.assert_not_called()
    with mock.patch.object(evaluate,'validated_all',return_value={}),mock.patch.object(run.training.reference_score,'load_slot_masks',side_effect=FileNotFoundError),mock.patch.object(run.reference.source.source.full,'read_gt') as gt:
        with pytest.raises(FileNotFoundError):run.score()
        gt.assert_not_called()


def test_changed_full320_benchmark_is_rejected():
    groups={name:np.zeros((3,320,3)) for name in ('parent237','slot')}
    accepted=dict(scores={name:dict.fromkeys(evaluate.METRICS,0.) for name in ('joint_projection','slot')})
    assert evaluate.verify_reference_means(groups,accepted)==0
    accepted['scores']['slot']['fg_ari']=.1
    with pytest.raises(ValueError):evaluate.verify_reference_means(groups,accepted)


def test_complete_full320_pipeline_requires_all_evidence_before_GT(tmp_path):
    masks=np.ones((320,64,64),dtype=np.int64)
    bundle={s:dict(labels=masks,baseline=masks,protocol=dict(binding=dict(dataset={},RGB_sha256='RGB'))) for s in run.SEEDS}
    slot={s:dict(labels64=masks) for s in run.SEEDS};events=[]
    accepted=dict(scores={name:dict(zip(evaluate.METRICS,[.7,.4,.4])) for name in ('joint_projection','slot')})
    def valid():events.append('three_full_protocols');return bundle
    def slots():assert events==['three_full_protocols'];events.append('three_slot_protocols');return slot
    def refs():assert events==['three_full_protocols','three_slot_protocols'];events.append('audited237');return accepted
    def gt(_):assert events==['three_full_protocols','three_slot_protocols','audited237'];return masks
    calls=[]
    def measured(labels,truth):
        i=len(calls);calls.append(i)
        values=[.92,.6,.6] if i%3==0 else [.7,.4,.4]
        return np.tile(values,(320,1)),0.
    with mock.patch.object(evaluate,'validated_all',side_effect=valid),mock.patch.object(run.training.reference_score,'load_slot_masks',side_effect=slots),mock.patch.object(evaluate,'references',side_effect=refs),mock.patch.object(run.reference.source.source.full,'read_gt',side_effect=gt),mock.patch.object(run.reference.source.source.full.metrics,'measured_vectors',side_effect=measured),mock.patch.object(run,'ARCHIVE',tmp_path),mock.patch.object(run,'fingerprint',return_value='fixture'):
        result=run.score()
    assert result['native64_FG090_three_seed_goal_verified'] and result['image_count']==320
    assert result['optimizer_updates']==0 and len(calls)==9
