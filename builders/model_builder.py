# builders/model_builder.py
try:
    import torch_npu
except ImportError:
    pass

from models.Nets import CNNEMnist, MelCNN
from models.resnet import ResNet18
from utils.state_dict_ops import model_param_dict


def build_model(args):
    model_name = str(args.model).lower()
    dataset_name = str(args.dataset).lower()

    if model_name == 'resnet' and dataset_name == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif model_name == 'resnet' and dataset_name == 'cifar100':
        net_glob = ResNet18(num_classes=100).to(args.device)
    elif model_name == 'resnet' and dataset_name == 'tinyimagenet':
        net_glob = ResNet18(num_classes=200).to(args.device)
    elif model_name == 'cnn' and dataset_name == 'femnist':
        net_glob = CNNEMnist(args=args).to(args.device)
    elif model_name == 'melcnn' and dataset_name == 'gspeech':
        net_glob = MelCNN(args=args).to(args.device)
    else:
        raise ValueError(f'Unrecognized model/dataset combination: model={args.model}, dataset={args.dataset}')

    policy = getattr(args, 'bn_buffer_policy', 'endpoint_mean')
    if policy not in ('endpoint_mean', 'legacy_delta'):
        raise ValueError(f'Unsupported BN buffer policy: {policy}')
    supported = str(getattr(args, 'algo', '')).lower() in (
        'fedbuff',
        'ca2fl',
        'fadas',
        'dirbridge',
    )
    endpoint_mean = supported and policy == 'endpoint_mean'
    args.resolved_bn_buffer_policy = 'endpoint_mean' if endpoint_mean else 'legacy_delta'
    args.bn_buffer_policy_qualification = (
        'unqualified' if not supported else
        ('endpoint_repair' if endpoint_mean else 'diagnostic_reference_only')
    )
    net_glob._asyncbuffer_trainable_state_only = endpoint_mean
    net_glob.train()
    w_glob = model_param_dict(net_glob, device=args.device)
    return net_glob, w_glob
