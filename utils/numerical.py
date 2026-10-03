"""Fail-fast numerical validation without modifying model state."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

__all__ = ['NumericalError', 'require_finite_tensor', 'validate_model']


def _json_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_value(value.detach().item())
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


class NumericalError(FloatingPointError):
    def __init__(self, message, stage, key=None, **context):
        self.details = {'stage': str(stage)}
        if key is not None:
            self.details['key'] = str(key)
        self.details.update({
            name: _json_value(value)
            for name, value in context.items() if value is not None
        })
        super().__init__(f'{stage}: {message}' + (f' ({key})' if key is not None else ''))


def require_finite_tensor(name, value, stage, **context):
    """Return the unchanged tensor or raise with serializable context."""
    checked = value.detach()
    if checked.is_sparse:
        checked = checked.coalesce().values()
    if not torch.isfinite(checked).all().item():
        raise NumericalError('nonfinite tensor', stage, key=name, **context)
    return value


def validate_model(model, stage, **context):
    """Check parameters, buffers, and BatchNorm variance without copying."""
    for name, param in model.named_parameters():
        require_finite_tensor(name, param, stage, **context)
    for name, buffer in model.named_buffers():
        require_finite_tensor(name, buffer, stage, **context)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            variance = module.running_var
            if variance is not None and (variance.detach() < 0).any().item():
                key = f'{name}.running_var' if name else 'running_var'
                raise NumericalError('negative BatchNorm running variance', stage, key=key, **context)
