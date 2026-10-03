#!/usr/bin/env python
"""Build the preselected FEMNIST .pt format used by the DirBridge artifact.

The script reads LEAF/FedScale-style FEMNIST JSON files under
<raw_root>/train and <raw_root>/test, selects a fixed client subset from the
training split, and writes compact .pt files that can be loaded by
`data_reader/femnist.py`.
"""

import argparse
import json
import os
from pathlib import Path

import torch


def read_user_data(split_dir):
    users = {}
    for path in sorted(Path(split_dir).glob("*.json")):
        with path.open("r") as handle:
            payload = json.load(handle)
        users.update(payload.get("user_data", {}))
    if not users:
        raise RuntimeError(f"No user_data entries found under {split_dir}")
    return users


def select_clients(train_users, num_clients, min_samples):
    counts = {
        user_id: len(user_data.get("x", []))
        for user_id, user_data in train_users.items()
    }
    eligible = sorted(
        user_id for user_id, count in counts.items()
        if count >= min_samples
    )
    if len(eligible) < num_clients:
        raise RuntimeError(
            f"Requested {num_clients} FEMNIST clients, but only {len(eligible)} "
            f"clients have at least {min_samples} training samples."
        )
    return eligible[:num_clients]


def build_split(users, client_ids, out_path):
    images = []
    targets = []
    clients = []
    num_samples = []

    for client_idx, user_id in enumerate(client_ids):
        user_data = users.get(user_id, {"x": [], "y": []})
        xs = user_data.get("x", [])
        ys = user_data.get("y", [])
        if len(xs) != len(ys):
            raise RuntimeError(f"Client {user_id} has mismatched x/y lengths")
        num_samples.append(len(xs))
        images.extend(xs)
        targets.extend(ys)
        clients.extend([client_idx] * len(xs))

    if images:
        image_tensor = torch.tensor(images, dtype=torch.float32).reshape(len(images), -1)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        client_tensor = torch.tensor(clients, dtype=torch.long)
    else:
        image_tensor = torch.empty((0, 784), dtype=torch.float32)
        target_tensor = torch.empty((0,), dtype=torch.long)
        client_tensor = torch.empty((0,), dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_ids": list(client_ids),
            "num_samples": num_samples,
            "images": image_tensor,
            "targets": target_tensor,
            "clients": client_tensor,
        },
        out_path,
    )
    print(
        f"Wrote {out_path} with {len(client_ids)} clients and "
        f"{int(image_tensor.shape[0])} samples"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Raw FEMNIST root containing train/ and test/ JSON files.")
    parser.add_argument("--out-root", default="data/femnist_pt", help="Output root for train/test .pt files.")
    parser.add_argument("--num-clients", type=int, default=1000, help="Number of clients to select.")
    parser.add_argument("--min-samples-per-client", type=int, default=250, help="Minimum training samples per selected client.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(f"Expected train/ and test/ under {raw_root}")

    train_users = read_user_data(train_dir)
    test_users = read_user_data(test_dir)
    client_ids = select_clients(
        train_users,
        num_clients=int(args.num_clients),
        min_samples=int(args.min_samples_per_client),
    )

    out_root = Path(args.out_root)
    build_split(train_users, client_ids, out_root / "train" / "femnist.pt")
    build_split(test_users, client_ids, out_root / "test" / "femnist.pt")

    print(
        f"Selected the first {len(client_ids)} sorted eligible clients "
        f"with at least {args.min_samples_per_client} training samples."
    )


if __name__ == "__main__":
    main()
