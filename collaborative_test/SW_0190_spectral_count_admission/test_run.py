import unittest
import numpy as np
from . import spectral


class SpectralAdmission(unittest.TestCase):
    def setUp(self):
        self.labels=np.array([0,0,0,0,1,1,2,2])
        self.q=np.where(self.labels[:,None]==self.labels[None,:],1.,.001)

    def test_block_affinity_recovers_whole_objects_up_to_label_permutation(self):
        found=spectral.cluster(self.q,3)
        np.testing.assert_array_equal(found[:,None]==found[None,:],self.labels[:,None]==self.labels[None,:])

    def test_background_uses_existing_model_mask_not_oracle_geometry(self):
        clusters=np.array([[3,3,8,8],[9,9,8,8]])
        prior=np.array([[0,0,1,1],[2,2,1,1]])
        output=spectral.background_from_model(clusters,prior)
        np.testing.assert_array_equal(output==0,clusters==3)

    def test_variants_bind_geometry_and_keep_existing_support_exact(self):
        prior=self.labels.reshape(2,4)
        outputs=spectral.transform(self.q,prior,2)
        for found in outputs.values():
            self.assertEqual(found.shape,prior.shape)
            np.testing.assert_array_equal(found>0,prior>0)
            self.assertEqual(len(np.unique(found[found>0])),2)

    def test_empty_support_and_single_node_do_not_crash(self):
        outputs=spectral.transform(np.eye(4),np.zeros((2,2),int),1)
        self.assertFalse(outputs['spectral_existing_support'].any())
        np.testing.assert_array_equal(spectral.cluster(np.ones((1,1)),5),[0])

    def test_invalid_or_asymmetric_graph_and_count_are_rejected(self):
        for q,k in ((np.ones((2,3)),2),(np.array([[1.,0.],[1.,1.]]),2),
                    (np.array([[np.nan]]),1),(-np.ones((2,2)),2),(np.eye(2),0)):
            with self.assertRaises(ValueError):spectral.cluster(q,k)


if __name__=='__main__':unittest.main()
