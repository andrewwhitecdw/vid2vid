import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel


class TestBaseModelBuildPyr(unittest.TestCase):
    def test_build_pyr_nearest_shapes(self):
        model = BaseModel()
        model.n_scales = 3
        x = torch.arange(24).float().view(1, 1, 1, 4, 6)
        pyr = model.build_pyr(x, nearest=True)
        self.assertEqual(len(pyr), 3)
        self.assertEqual(tuple(pyr[0].shape), (1, 1, 1, 4, 6))
        self.assertEqual(tuple(pyr[1].shape), (1, 1, 1, 2, 3))
        self.assertEqual(tuple(pyr[2].shape), (1, 1, 1, 1, 2))
        # nearest=True path uses 1x1 pooling and just subsamples rows/cols
        self.assertTrue(torch.equal(pyr[0][0, 0, 0], x[0, 0, 0]))
        self.assertTrue(torch.equal(pyr[1][0, 0, 0], x[0, 0, 0, ::2, ::2]))
        self.assertTrue(torch.equal(pyr[2][0, 0, 0, 0], x[0, 0, 0, 0, ::4]))

    def test_build_pyr_non_nearest_shapes(self):
        model = BaseModel()
        model.n_scales = 3
        x = torch.arange(24).float().view(1, 1, 1, 4, 6)
        pyr = model.build_pyr(x, nearest=False)
        self.assertEqual(len(pyr), 3)
        self.assertEqual(tuple(pyr[0].shape), (1, 1, 1, 4, 6))
        self.assertEqual(tuple(pyr[1].shape), (1, 1, 1, 2, 3))
        self.assertEqual(tuple(pyr[2].shape), (1, 1, 1, 1, 2))
        # Non-nearest uses 3x3 averaging, so values differ from nearest.
        nearest_pyr = model.build_pyr(x, nearest=True)
        self.assertFalse(torch.equal(pyr[1][0, 0, 0], nearest_pyr[1][0, 0, 0]))

    def test_build_pyr_none_input(self):
        model = BaseModel()
        model.n_scales = 2
        out = model.build_pyr(None)
        self.assertEqual(out, [None, None])


if __name__ == '__main__':
    unittest.main()
