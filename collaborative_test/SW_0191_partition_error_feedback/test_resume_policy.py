import unittest
from .resume_policy import compatible_runtime,POLICY


class ResumeCompatibility(unittest.TestCase):
    def test_physical_index_only_is_excluded_and_input_is_not_mutated(self):
        left=dict(cuda_visible_devices='2',device_index=0,device_name='same GPU',device_total_memory=123,
                  torch_version='same',deterministic_algorithms=False)
        right={**left,'cuda_visible_devices':'3'}
        self.assertEqual(compatible_runtime(left),compatible_runtime(right))
        self.assertEqual(left['cuda_visible_devices'],'2')
        self.assertEqual(compatible_runtime(left)['sw0191_resume_policy'],POLICY)
        for key,value in (('device_name','different GPU'),('device_index',1),('torch_version','different'),
                          ('deterministic_algorithms',True),('device_total_memory',124)):
            self.assertNotEqual(compatible_runtime(left),compatible_runtime({**right,key:value}))

    def test_multiple_visible_devices_rejected(self):
        with self.assertRaises(ValueError):compatible_runtime(dict(cuda_visible_devices='0,1'))


if __name__=='__main__':unittest.main()
