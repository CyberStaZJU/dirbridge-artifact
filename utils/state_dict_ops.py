import torch
from typing import Mapping
try:
    import torch_npu
except ImportError:
    pass
import math
import torch.nn as nn
from utils.numerical import NumericalError, require_finite_tensor, validate_model

@torch.no_grad()
def sd_average(dicts):
    """
    对一组同结构的参数字典求平均。
    输入: List[Dict[str, Tensor]]
    输出: Dict[str, Tensor]
    """
    if dicts is None or len(dicts) == 0:
        raise ValueError("sd_average expects a non-empty list of dicts.")

    avg = {k: torch.zeros_like(v) for k, v in dicts[0].items()}

    for d in dicts:
        for k in avg:
            avg[k].add_(d[k])

    scale = 1.0 / float(len(dicts))
    for k in avg:
        avg[k].mul_(scale)

    return avg

def model_param_dict(model: nn.Module, device=None):
    """
    导出异步算法需要同步的 state。
    默认保持旧行为：导出 state_dict 中的参数和 buffer；如果模型标记了
    _asyncbuffer_trainable_state_only，则只导出 requires_grad=True 的参数。
    返回: Dict[str, Tensor]
    """
    out = {}

    if getattr(model, '_asyncbuffer_trainable_state_only', False):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            v = param.detach().clone()
            if device is not None:
                v = v.to(device)
            out[name] = v
        return out

    for name, t in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        v = t.detach().clone()
        if device is not None:
            v = v.to(device)
        out[name] = v
    return out


@torch.no_grad()
def load_param_dict_(model: nn.Module, param_dict):
    """
    把参数字典写回模型。
    只覆盖 named_parameters() 中出现的参数。
    """
    cur = model.state_dict()
    for name, v in param_dict.items():
        if name in cur:
            cur[name].copy_(v)
    model.load_state_dict(cur, strict=False)


def validate_state_dict(state_dict, context='model state', check_running_var=True):
    """Reject invalid state without changing tensors."""
    for name, value in state_dict.items():
        require_finite_tensor(name, value, context)
        if check_running_var and name.endswith('running_var') and (value < 0).any().item():
            raise NumericalError('negative running variance', context, key=name)


def validate_model_state(model, context='model state'):
    validate_model(model, context)


def uses_buffer_endpoints(model):
    return bool(getattr(model, '_asyncbuffer_trainable_state_only', False))


def model_buffer_dict(model, device=None):
    """Snapshot absolute persistent buffers, including BN batch counters."""
    buffers = dict(model.named_buffers())
    out = {
        name: value.detach().clone().to(device=device or value.device)
        for name, value in model.state_dict().items() if name in buffers
    }
    validate_state_dict(out, context='local buffer endpoint')
    return out


def init_buffer_endpoints(state, num_users):
    if uses_buffer_endpoints(state['net_glob']):
        state['buffer_endpoints'] = [None] * int(num_users)


def capture_buffer_endpoint(state, idx, model):
    if uses_buffer_endpoints(state['net_glob']):
        state['buffer_endpoints'][idx] = model_buffer_dict(model)


@torch.no_grad()
def apply_buffer_endpoints_(state, selected_clients):
    """Uniform endpoint mean; integer buffers use the selected maximum."""
    model = state['net_glob']
    if not uses_buffer_endpoints(model):
        return
    if not selected_clients:
        raise ValueError('buffer endpoints require selected clients')
    endpoints = [state['buffer_endpoints'][idx] for idx in selected_clients]
    buffers = model_buffer_dict(model)
    for endpoint in endpoints:
        if endpoint is None or endpoint.keys() != buffers.keys():
            raise ValueError('missing or mismatched buffer endpoint')
        validate_state_dict(endpoint, context='selected buffer endpoint')
        for name, value in endpoint.items():
            if value.shape != buffers[name].shape or value.dtype != buffers[name].dtype:
                raise ValueError(f'mismatched buffer endpoint {name}')
    aggregated = {}
    for name, buffer in buffers.items():
        values = [endpoint[name].to(buffer.device) for endpoint in endpoints]
        if buffer.is_floating_point() or buffer.is_complex():
            mean = torch.zeros_like(buffer)
            for value in values:
                mean.add_(value / len(values))
            aggregated[name] = mean
        else:
            maximum = values[0].clone()
            for value in values[1:]:
                maximum = torch.maximum(maximum, value)
            aggregated[name] = maximum
    validate_state_dict(aggregated, context='aggregated buffer endpoints')
    load_param_dict_(model, aggregated)


def sd_zero_like(sd):
    return {k: torch.zeros_like(v) for k, v in sd.items()}


def sd_copy(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


def sd_sub(a, b):
    return {k: a[k] - b[k] for k in a}


def sd_axpy(y, a, x):
    for k in y:
        y[k].add_(x[k], alpha=a)


def sd_scale(sd, a):
    return {k: a * v for k, v in sd.items()}


def sd_lerp(a, b, weight_b):
    weight_a = 1.0 - weight_b
    return {k: weight_a * a[k] + weight_b * b[k] for k in a}


def sd_div_(sd, a: float):
    for k in sd:
        sd[k].div_(a)


def dict_l2_norm(d):
    if d is None:
        return float("nan")
    s = 0.0
    for v in d.values():
        if v is None:
            continue
        s += v.detach().float().pow(2).sum().item()
    return math.sqrt(s)
