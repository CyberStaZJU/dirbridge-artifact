# builders/dataset_builder.py
import json
import os

import numpy as np

from data_reader import femnist
from data_reader.gspeech import FedScaleGSpeech
from utils.sampling import (
    extract_labels,
    generate_iid,
    generate_label_block,
    generate_label_correlated,
    generate_noniid,
)


def _load_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("Vision datasets require the optional package `torchvision`.") from exc
    return datasets, transforms


def _attach_dirichlet_direction_groups(dataset_train, dict_users, args):
    labels = extract_labels(dataset_train)
    classes = np.sort(np.unique(labels))
    if getattr(args, 'random_cost', '') == 'dir-skew':
        num_groups = 3
    else:
        num_groups = int(getattr(args, 'label_block_groups', 5))
    if num_groups <= 0:
        return
    num_groups = min(num_groups, len(classes))
    label_groups = [group.astype(int).tolist() for group in np.array_split(classes, num_groups)]

    client_group_ids = []
    users_per_group = [0] * num_groups
    for user_idx in range(int(args.num_users)):
        user_indices = np.asarray(dict_users[user_idx], dtype=np.int64)
        user_labels = labels[user_indices]
        group_counts = [
            int(np.isin(user_labels, np.asarray(group_labels, dtype=labels.dtype)).sum())
            for group_labels in label_groups
        ]
        group_id = int(np.argmax(group_counts))
        client_group_ids.append(group_id)
        users_per_group[group_id] += 1

    args.label_correlated_num_groups = int(num_groups)
    args.label_correlated_label_groups = label_groups
    args.label_correlated_group_ids = client_group_ids
    args.label_correlated_users_per_group = users_per_group
    args.label_correlated_primary_fraction = 0.0


def _partition_vision_dataset(dataset_train, args):
    distribution = str(args.distribution).lower()
    if distribution == 'iid':
        return generate_iid(dataset_train, args.num_users)
    if distribution == 'noniid':
        dict_users = generate_noniid(dataset_train, args.num_users, args.alpha)
        if getattr(args, 'random_cost', '') == 'dir-skew':
            _attach_dirichlet_direction_groups(dataset_train, dict_users, args)
        return dict_users
    if distribution in {'label_block', 'label_blocks', 'block_label', 'label_disjoint'}:
        num_groups = int(getattr(args, 'label_block_groups', 5))
        dict_users, metadata = generate_label_block(
            dataset_train,
            args.num_users,
            num_groups=num_groups,
            seed=int(getattr(args, 'seed', 0)),
            return_metadata=True,
        )
        args.distribution = 'label_block'
        args.label_block_num_groups = int(metadata['num_groups'])
        args.label_block_label_groups = metadata['label_groups']
        args.label_block_group_ids = metadata['client_group_ids']
        args.label_block_users_per_group = metadata['users_per_group']
        return dict_users
    if distribution in {'label_correlated', 'soft_label_block', 'label_soft_block'}:
        num_groups = int(getattr(args, 'label_block_groups', 5))
        primary_fraction = float(getattr(args, 'label_primary_fraction', 0.5))
        dict_users, metadata = generate_label_correlated(
            dataset_train,
            args.num_users,
            num_groups=num_groups,
            primary_fraction=primary_fraction,
            seed=int(getattr(args, 'seed', 0)),
            return_metadata=True,
        )
        args.distribution = 'label_correlated'
        args.label_correlated_num_groups = int(metadata['num_groups'])
        args.label_correlated_label_groups = metadata['label_groups']
        args.label_correlated_group_ids = metadata['client_group_ids']
        args.label_correlated_users_per_group = metadata['users_per_group']
        args.label_correlated_primary_fraction = float(metadata['primary_fraction'])
        return dict_users
    raise ValueError("distribution must be 'iid', 'noniid', 'label_block', or 'label_correlated'")


def _select_client_ids_from_leaf_files(data_path, args, dataset_label, min_samples_attr):
    train_dir = os.path.join(data_path, 'train')
    requested_users = int(args.num_users)
    min_samples = int(getattr(args, min_samples_attr, 0) or 0)
    if requested_users <= 0:
        raise ValueError(f"{dataset_label} --num_users must be positive")

    counts = {}
    for filename in sorted(os.listdir(train_dir)):
        file_path = os.path.join(train_dir, filename)
        if filename.endswith('.json'):
            with open(file_path, 'r') as handle:
                payload = json.load(handle)
            for user_id, user_data in payload.get('user_data', {}).items():
                counts[user_id] = len(user_data.get('x', []))
        elif filename.endswith('.pt'):
            import torch
            payload = torch.load(file_path, weights_only=True)
            for user_id, n in zip(payload.get('client_ids', []), payload.get('num_samples', [])):
                counts[user_id] = n

    eligible_client_ids = [
        user_id for user_id, count in counts.items()
        if min_samples <= 0 or count >= min_samples
    ]
    eligible_client_ids = sorted(eligible_client_ids)
    if requested_users > len(eligible_client_ids):
        raise ValueError(
            f"{dataset_label} requested {requested_users} users, but only "
            f"{len(eligible_client_ids)} eligible clients are available under {data_path} "
            f"with min_samples_per_client={min_samples}."
        )

    selected_client_ids = eligible_client_ids[:requested_users]
    selected_counts = [counts[user_id] for user_id in selected_client_ids]
    print(
        f'** using fixed {requested_users} {dataset_label} clients '
        f'out of {len(eligible_client_ids)} eligible / {len(counts)} available clients '
        f'(min_samples_per_client={min_samples}) **'
    )
    print(
        f'** selected {dataset_label} train samples: {sum(selected_counts)} '
        f'(min={min(selected_counts)}, max={max(selected_counts)}) **'
    )
    return selected_client_ids


def build_dataset(args):
    dataset_name = str(args.dataset).lower()
    if dataset_name == 'cifar':
        return _build_cifar(args)
    if dataset_name == 'cifar100':
        return _build_cifar100(args)
    if dataset_name == 'tinyimagenet':
        return _build_tinyimagenet(args)
    if dataset_name == 'femnist':
        return _build_femnist(args)
    if dataset_name == 'gspeech':
        return _build_gspeech(args)
    raise ValueError(f'Unrecognized dataset: {args.dataset}')


def _build_cifar(args):
    datasets, transforms = _load_torchvision()
    args.num_classes = 10
    args.num_channels = 3
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    dataset_train = datasets.CIFAR10(
        './data/cifar',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    dataset_test = datasets.CIFAR10(
        './data/cifar',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    dict_users = _partition_vision_dataset(dataset_train, args)
    num_samples = np.array([len(dict_users[i]) for i in dict_users])
    return dataset_train, dataset_test, dict_users, num_samples, {"img_size": dataset_train[0][0].shape}


def _build_cifar100(args):
    datasets, transforms = _load_torchvision()
    args.num_classes = 100
    args.num_channels = 3
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    dataset_train = datasets.CIFAR100(
        './data/cifar100',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    dataset_test = datasets.CIFAR100(
        './data/cifar100',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    dict_users = _partition_vision_dataset(dataset_train, args)
    num_samples = np.array([len(dict_users[i]) for i in dict_users])
    return dataset_train, dataset_test, dict_users, num_samples, {"img_size": dataset_train[0][0].shape}


def _build_tinyimagenet(args):
    datasets, transforms = _load_torchvision()
    args.num_classes = 200
    args.num_channels = 3
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    train_dir = './data/tiny-imagenet-200/train'
    val_dir = './data/tiny-imagenet-200/val'
    dataset_train = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    dataset_test = datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    dict_users = _partition_vision_dataset(dataset_train, args)
    num_samples = np.array([len(dict_users[i]) for i in dict_users])
    return dataset_train, dataset_test, dict_users, num_samples, {"img_size": dataset_train[0][0].shape}


def _build_femnist(args):
    _, transforms = _load_torchvision()
    args.dataset = 'femnist'
    args.num_classes = 62
    args.num_channels = 1

    data_path = os.path.abspath(getattr(args, 'femnist_data_dir', '') or './data/femnist_pt')
    if not (os.path.isdir(os.path.join(data_path, 'train')) and os.path.isdir(os.path.join(data_path, 'test'))):
        raise RuntimeError(
            "FEMNIST dataset not found. Expected train/ and test/ subdirs "
            f"with .json or .pt data files under {data_path}."
        )

    use_preselected_femnist = (
        os.path.basename(data_path) in ('femnist_truncated', 'femnist_pt')
        and int(args.num_users) == 1000
        and int(getattr(args, 'femnist_min_samples_per_client', 0) or 0) == 250
    )
    selected_client_ids = None if use_preselected_femnist else _select_client_ids_from_leaf_files(
        data_path,
        args,
        'FEMNIST',
        'femnist_min_samples_per_client',
    )
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = femnist.FEMNIST(data_path, train=True, download=False, transform=transform, selected_client_ids=selected_client_ids)
    dataset_test = femnist.FEMNIST(data_path, train=False, download=False, transform=transform, selected_client_ids=selected_client_ids)

    dict_users = dataset_train.get_dict_clients()
    if selected_client_ids is None:
        selected_client_ids = list(dataset_train.client_ids)
    args.num_users = len(selected_client_ids)
    num_samples = np.array([len(dict_users[i]) for i in range(args.num_users)])
    if getattr(args, 'random_cost', '') == 'dir-skew':
        _attach_dirichlet_direction_groups(dataset_train, dict_users, args)
    return dataset_train, dataset_test, dict_users, num_samples, {
        "img_size": dataset_train[0][0].shape,
        "selected_client_ids": selected_client_ids,
    }


def _build_gspeech(args):
    args.dataset = 'gspeech'
    args.model = str(getattr(args, 'model', 'melcnn') or 'melcnn')
    args.num_channels = 1

    data_path = os.path.abspath('./data/gspeech')
    selected_client_ids = _select_client_ids_from_leaf_files(
        data_path,
        args,
        'GSpeech',
        'gspeech_min_samples_per_client',
    )
    dataset_train = FedScaleGSpeech(data_path, train=True, selected_client_ids=selected_client_ids)
    dataset_test = FedScaleGSpeech(data_path, train=False, label_map=dataset_train.label_map)
    args.num_classes = int(dataset_train.num_classes)

    dict_users = dataset_train.get_dict_clients()
    args.num_users = len(selected_client_ids)
    num_samples = np.array([len(dict_users[i]) for i in range(args.num_users)])
    return dataset_train, dataset_test, dict_users, num_samples, {
        "img_size": dataset_train[0][0].shape,
        "selected_client_ids": selected_client_ids,
        "num_labels": int(dataset_train.num_classes),
        "label_map": dict(dataset_train.label_map),
    }
