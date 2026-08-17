import unittest

import torch

from models.base_model import BaseModel


class ConcatValidationTest(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(BaseModel)

    def test_concat_with_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.concat([])

    def test_concat_with_one_tensor_raises_value_error(self):
        tensor = torch.tensor([1, 2])
        with self.assertRaises(ValueError):
            self.model.concat([tensor])

    def test_concat_with_three_tensors_raises_value_error(self):
        tensor = torch.tensor([1, 2])
        with self.assertRaises(ValueError):
            self.model.concat([tensor, tensor, tensor])

    def test_concat_with_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.concat(None)

    def test_concat_valid_two_tensors(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([3, 4])
        result = self.model.concat([a, b], dim=0)
        expected = torch.cat([a, b], dim=0)
        self.assertTrue(torch.equal(result, expected))

    def test_concat_accepts_tuple(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([3, 4])
        result = self.model.concat((a, b), dim=0)
        expected = torch.cat([a, b], dim=0)
        self.assertTrue(torch.equal(result, expected))
