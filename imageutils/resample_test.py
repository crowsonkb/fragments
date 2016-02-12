"""Unit tests for imageutils.py"""

import unittest

import numpy as np

from resample import resample


def flatten(arr):
    """Convert a NumPy array into a form testable by assertEqual()"""
    return (arr.shape, arr.dtype, list(arr.reshape(arr.size)))


class TestResample(unittest.TestCase):
    """Tests resample.resample()"""

    def test_1x1_nearest(self):
        """Ensure the do-nothing resample is working"""
        img = np.ones((1, 1), dtype=np.float64)
        result = resample(img, w=1, h=1, method='nearest')
        self.assertEqual(img.shape, result.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(img[0, 0], result[0, 0])

    def test_2_to_4_nearest(self):
        img = np.float32([[1, 2], [3, 4]])
        result = resample(img, w=4, h=4, method='nearest')
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result[1, 1], 1)
        self.assertEqual(result[2, 2], 4)

    def test_bilinear_downsample(self):
        img = np.uint8([[[1, 2, 3], [5, 6, 9]]])
        result = resample(img, w=1, h=1, method='bilinear')
        self.assertEqual(flatten(result[0, 0, :]),
                         flatten(np.uint8([3, 4, 6])))

if __name__ == '__main__':
    unittest.main()
