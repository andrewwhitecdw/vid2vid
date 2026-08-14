import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import wrap_model


class TestWrapModel(unittest.TestCase):
    def _make_opt(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def _wrap(self, opt):
        modelG = MagicMock()
        modelD = MagicMock()
        flowNet = MagicMock()
        return wrap_model(opt, modelG, modelD, flowNet)

    def test_all_gpu_generator_uses_data_parallel_on_all_gpus(self):
        opt = self._make_opt(n_gpus_gen=2, gpu_ids=[0, 1], batchSize=4)
        with patch('models.models.nn.DataParallel') as mock_dp:
            self._wrap(opt)
        self.assertEqual(len(mock_dp.call_args_list), 3)
        for call in mock_dp.call_args_list:
            self.assertEqual(call[1]['device_ids'], [0, 1])

    def test_batch_size_one_restricts_generator_to_first_gpu(self):
        opt = self._make_opt(n_gpus_gen=1, gpu_ids=[0, 1, 2], batchSize=1)
        with patch('models.models.nn.DataParallel') as mock_dp:
            self._wrap(opt)
        self.assertEqual(mock_dp.call_args_list[0][1]['device_ids'], [0])
        for call in mock_dp.call_args_list[1:]:
            self.assertEqual(call[1]['device_ids'], [0, 2])

    def test_multi_batch_splits_remaining_gpus_for_discriminator(self):
        opt = self._make_opt(n_gpus_gen=2, gpu_ids=[0, 1, 2, 3], batchSize=4)
        with patch('models.models.nn.DataParallel') as mock_dp:
            self._wrap(opt)
        self.assertEqual(mock_dp.call_args_list[0][1]['device_ids'], [0, 1])
        for call in mock_dp.call_args_list[1:]:
            self.assertEqual(call[1]['device_ids'], [0, 2, 3])


if __name__ == '__main__':
    unittest.main()
