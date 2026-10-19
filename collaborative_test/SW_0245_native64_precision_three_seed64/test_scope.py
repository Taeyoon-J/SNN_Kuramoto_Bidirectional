import unittest
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


def test_three_seed_means_keep_scope_and_negative_seed_visible():
    import numpy as np
    from .scope import aggregate_vectors
    from . import run
    assert run.SEEDS==(0,1,2) and run.OFFSETS==tuple(range(0,320,5))
    vectors={name:np.full((3,64,3),.5) for name in ('control','within16','parent237','slot')}
    vectors['within16'][:]=.8;vectors['within16'][2,:,0]=.4
    result=aggregate_vectors(vectors,('fg_ari','foreground_iou','matched_object_iou'))
    assert result['within16_promising_for_separate_full320_confirmation']
    assert abs(result['scores']['within16']['fg_ari']-2/3)<1e-12
    assert result['per_seed']['within16']['2']['fg_ari']<result['per_seed']['slot']['2']['fg_ari']
    vectors['slot']=vectors['slot'][:2]
    with np.testing.assert_raises(ValueError):aggregate_vectors(vectors,('fg_ari','foreground_iou','matched_object_iou'))
