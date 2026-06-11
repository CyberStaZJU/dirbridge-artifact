import torch
try:
    import torch_npu
except ImportError:
    pass
import math
import torch.nn as nn

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
