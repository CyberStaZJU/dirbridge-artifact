import unittest

import torch
from torch import nn

from utils.state_dict_ops import (
    apply_buffer_endpoints_,
    capture_buffer_endpoint,
    init_buffer_endpoints,
    model_param_dict,
)


class BatchNormEndpointRepairTests(unittest.TestCase):
    def _state(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model._asyncbuffer_trainable_state_only = True
        state = {'net_glob': model}
        init_buffer_endpoints(state, 2)
        return model, state

    def test_delta_export_excludes_batchnorm_buffers(self):
        model, _ = self._state()
        exported = model_param_dict(model)
        self.assertEqual(set(exported), {'0.weight', '0.bias', '1.weight', '1.bias'})

    def test_float_endpoints_are_averaged_and_integer_endpoints_use_maximum(self):
        model, state = self._state()
        for idx, mean, var, count in ((0, 1.0, 3.0, 4), (1, 5.0, 7.0, 9)):
            model[1].running_mean.fill_(mean)
            model[1].running_var.fill_(var)
            model[1].num_batches_tracked.fill_(count)
            capture_buffer_endpoint(state, idx, model)

        apply_buffer_endpoints_(state, [0, 1])
        self.assertTrue(torch.allclose(model[1].running_mean, torch.full((2,), 3.0)))
        self.assertTrue(torch.allclose(model[1].running_var, torch.full((2,), 5.0)))
        self.assertEqual(int(model[1].num_batches_tracked), 9)

    def test_missing_endpoint_is_rejected(self):
        _, state = self._state()
        with self.assertRaises(ValueError):
            apply_buffer_endpoints_(state, [0, 1])

    def test_negative_running_variance_is_rejected(self):
        model, state = self._state()
        model[1].running_var.fill_(-1.0)
        with self.assertRaises(FloatingPointError):
            capture_buffer_endpoint(state, 0, model)


if __name__ == '__main__':
    unittest.main()
