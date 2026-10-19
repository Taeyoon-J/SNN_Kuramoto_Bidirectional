import unittest
import numpy as np
from . import run


class ScopeTests(unittest.TestCase):
    def test_only_registered_seed_and_arms_are_accepted(self):
        self.assertEqual(run.require_seed(1),1)
        for seed in (-1,3,True):
            with self.assertRaises(ValueError):run.require_seed(seed)
        with self.assertRaises(ValueError):run.require_arm('within4')

    def test_queue_carries_explicit_matched_arm(self):
        from . import queue
        self.assertIs(queue.api,run);run.activate('control')
        command=queue.worker_command(1,3)
        self.assertEqual(command[-2:],['--arm','control'])
        self.assertEqual(command[command.index('--physical-gpu')+1],'3')
        with self.assertRaises(ValueError):queue.worker_command(3,3)


if __name__=='__main__':unittest.main()



def test_budget_comparison_requires_all_controls_and_keeps_seed_scores():
    import numpy as np
    from .scope import aggregate_vectors
    assert run.STEPS==512 and run.SEEDS==(0,1,2) and run.OFFSETS==tuple(range(0,320,10))
    vectors={k:np.full((3,32,3),.5) for k in ('control512','within16_512','control128','within16_128','parent237','slot')}
    vectors['control512'][:]=.8;vectors['within16_512'][:]=.85;vectors['within16_512'][2,:,0]=.4
    r=aggregate_vectors(vectors,('fg_ari','foreground_iou','matched_object_iou'))
    assert r['selected_arm_for_separate_full320_confirmation']=='control'
    assert r['per_seed']['within16_512']['2']['fg_ari']<r['per_seed']['slot']['2']['fg_ari']
    vectors['slot']=vectors['slot'][:2]
    with np.testing.assert_raises(ValueError):aggregate_vectors(vectors,('fg_ari','foreground_iou','matched_object_iou'))


def test_missing_prior128_seed_blocks_evaluation_GT():
    from unittest import mock
    calls=[]
    def old(seed,arm):
        calls.append((seed,arm))
        if len(calls)==6:raise FileNotFoundError('one old128 protocol missing')
        return {}
    with mock.patch.object(run,'validated_seed',return_value={}),mock.patch.object(run.prior,'validated_seed',side_effect=old),mock.patch.object(run.reference.source.source.full,'read_gt') as gt:
        with np.testing.assert_raises(FileNotFoundError):run.score()
        gt.assert_not_called()
    assert len(calls)==6


def test_missing_comparable_slot_blocks_evaluation_GT():
    from unittest import mock
    labels=np.zeros((32,64,64),dtype=np.int64);oldlabels=np.zeros((64,64,64),dtype=np.int64)
    b={k:'same' for k in ('TRAIN_ids','TRAIN_RGB_sha256','TRAIN_labels_sha256','teacher_arrays_sha256','initial_model_sha256','frozen_core_sha256','input_statistics_sha256','DEV_RGB_sha256')};b['dataset']={}
    new=dict(binding=b,parent=labels,labels=labels);old=dict(binding=b,parent=oldlabels,labels=oldlabels)
    with mock.patch.object(run,'validated_seed',return_value=new),mock.patch.object(run.prior,'validated_seed',return_value=old),mock.patch.object(run.reference_score,'load_slot_masks',side_effect=FileNotFoundError),mock.patch.object(run.reference.source.source.full,'read_gt') as gt:
        with np.testing.assert_raises(FileNotFoundError):run.score()
        gt.assert_not_called()
