import json
import unittest
from unittest.mock import patch
import numpy as np
from . import run
from collaborative_test.SW_0194_native64_train64.test_train import temporary


class BenchmarkBindingTests(unittest.TestCase):
    def fixture(self):
        archive=temporary()
        binding=dict(fingerprint=None,image_ids=[0,4,8,12],RGB_rows_sha256=['rgb']*4,
            source_model_sha256='frozen',GT_used_for_prediction=False,optimizer_updates=0)
        mask=np.zeros((64,64),dtype=np.int64);mask[5:50,8:40]=1
        gradient=dict(binding=binding,result=dict(gate_passed=True,distance_allclose=True))
        run.train.write_json(archive/'gradient.json',gradient)
        hashes=[]
        for i in range(4):
            folder=archive/'comparison/rows'/f'{i:04d}';folder.mkdir(parents=True)
            with (folder/'masks.npz').open('xb') as stream:np.savez_compressed(stream,reference=mask,matmul=mask)
            run.train.write_json(folder/'row.json',dict(binding=run.row_binding(binding,i),
                prediction_sha256=run.train.sha((folder/'masks.npz').read_bytes()),
                comparisons=dict(mask_exact=True,reference_foreground_patches=int((mask>0).sum()),
                    geodesic_allclose=True,adjacency_allclose=True,q_allclose=True),
                timing=dict(reference=dict(seconds=10.,geodesic_seconds=8.),matmul=dict(seconds=2.,geodesic_seconds=1.))))
            hashes.append(run.train.sha((folder/'row.json').read_bytes()))
        run.train.write_json(archive/'comparison/protocol.json',dict(binding=binding,rows_sha256=hashes,
            gradient_sha256=run.train.sha((archive/'gradient.json').read_bytes())))
        return archive

    def test_actual_four_mask_pairs_and_metrics_validate(self):
        archive=self.fixture()
        with patch.object(run,'ARCHIVE',archive),patch.object(run,'fingerprint',return_value=None):
            result=run.score()
            self.assertTrue(result['forward_and_z_gradient_candidate'])
            self.assertFalse(result['production_training_admitted'])
            self.assertFalse(result['optimizer_proposal_equivalence_verified'])
            self.assertFalse(result['FG_ARI_improvement_claim'])
            self.assertEqual(result['whole_prediction_speedup'],5.)

    def test_mask_payload_corruption_blocks_result(self):
        archive=self.fixture()
        with (archive/'comparison/rows/0000/masks.npz').open('ab') as stream:stream.write(b'bad')
        with patch.object(run,'ARCHIVE',archive),patch.object(run,'fingerprint',return_value=None):
            with self.assertRaises(ValueError):run.score()

    def test_edited_gradient_proof_blocks_result(self):
        archive=self.fixture()
        with (archive/'gradient.json').open('ab') as stream:stream.write(b' ')
        with patch.object(run,'ARCHIVE',archive),patch.object(run,'fingerprint',return_value=None):
            with self.assertRaises(ValueError):run.validated_pair()

    def test_prediction_identity_binds_image_rng_RGB_and_model(self):
        binding=dict(image_ids=[123],RGB_rows_sha256=['RGB'],source_model_sha256='model')
        row=run.row_binding(binding,0)
        self.assertEqual(row['inference_seed'],196000);self.assertEqual(row['image_id'],123)
        self.assertEqual(row['source_model_sha256'],'model');self.assertEqual(row['RGB_sha256'],'RGB')

    def test_queue_waits_for_actual195_and_does_not_change195_source(self):
        source=(run.HERE/'queue.py').read_text()
        self.assertIn("b'collaborative_test.SW_0195_native64_generalization_gap.queue'",source)
        self.assertNotIn('SW_0195_native64_train64',source)
        self.assertEqual(run.PREDECESSOR_ARCHIVE.name,'results_archive')
        self.assertEqual(run.PREDECESSOR_ARCHIVE.parent.name,'SW_0195_native64_generalization_gap')

    def test_independent_ready_queue_preserves_assigned_GPU_on_all_four_workers(self):
        from . import queue
        with patch.object(queue,'READY',True):
            for gpu in range(4):
                argv=queue.worker_command(gpu)
                self.assertEqual(argv[argv.index('--physical-gpu')+1],str(gpu))
                self.assertIn('--ready-now',argv)
        with patch.object(queue,'READY',False):
            self.assertNotIn('--ready-now',queue.worker_command(0))


if __name__=='__main__':unittest.main()
