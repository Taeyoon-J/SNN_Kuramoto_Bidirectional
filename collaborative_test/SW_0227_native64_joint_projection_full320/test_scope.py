import unittest
import numpy as np
from . import scope


class ScopeTests(unittest.TestCase):
    def test_full320_contains_exact_old64_and_new256(self):
        self.assertEqual(len(scope.MINI),64);self.assertEqual(len(scope.NEW),256)
        self.assertFalse(set(scope.MINI)&set(scope.NEW));self.assertEqual(sorted(scope.MINI+scope.NEW),list(range(320)))
        self.assertIn(0,scope.MINI);self.assertEqual((1320+scope.OFFSETS[0],1320+scope.OFFSETS[-1]),(1320,1639))

    def test_existing_canonical_float_conversion_and_byte_RGB_preservation(self):
        data=np.zeros((2,128,128,3),np.float32);data[...,1]=.5;data[...,2]=1.
        rgb=scope.canonical_rgb(data);self.assertEqual(rgb.dtype,np.uint8);np.testing.assert_array_equal(rgb[0,0,0],[0,128,255])
        before=rgb.copy();np.testing.assert_array_equal(scope.canonical_rgb(rgb),before);self.assertEqual(tuple(rgb.shape),(2,128,128,3))

    def test_geometry_not_silently_resized(self):
        with self.assertRaises(ValueError):scope.canonical_rgb(np.zeros((2,126,126,3),np.float32))


if __name__=='__main__':unittest.main()
