import unittest
from unittest.mock import patch
import numpy as np
from . import run


class PipelineTests(unittest.TestCase):
    def test_full_N_operator_induction_preserves_actual_energy_without_renormalization(self):
        rng=np.random.RandomState(217);x=rng.uniform(.1,.9,(6,6));q=(x+x.T)/2;a=rng.uniform(0,.2,(6,6));gain=1.2;indices=np.array([0,2,5])
        full=run.signed.signed_pair_energy(q,a,gain)
        induced=run.signed.signed_pair_energy(q[np.ix_(indices,indices)],a[np.ix_(indices,indices)],gain)
        np.testing.assert_array_equal(induced,full[np.ix_(indices,indices)])

    def test_GT_not_opened_if_any_seed_prediction_fails(self):
        with patch.object(run,'validated_predictions',side_effect=ValueError('missing seed')),patch.object(run.source.full,'read_gt') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()

    def test_fixed_source_seed_ids_and_physical_GPU_command(self):
        from .queue import worker_command
        binding=dict(seed=2,RGB_rows_sha256=['x']*16)
        row=run.row_binding(binding,15);self.assertEqual(row['image_id'],1350);self.assertEqual(row['inference_seed'],193030)
        self.assertIn('0',worker_command(0,3));self.assertEqual(worker_command(0,3)[-2:],['--physical-gpu','3'])
        with self.assertRaises(ValueError):worker_command(True,1)
        with self.assertRaises(ValueError):worker_command(1,True)
        with self.assertRaises(ValueError):run.row_binding(binding,16)


if __name__=='__main__':unittest.main()
