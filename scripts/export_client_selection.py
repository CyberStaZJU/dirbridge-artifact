#!/usr/bin/env python
"""Export the concrete client ids and FedScale profile mapping used by the artifact."""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.fedscale_trace import _as_profile_mapping, _canonical_client_id, _numeric_sort_key, load_fedscale_payload


def count_clients_from_json(train_dir):
    counts = {}
    for path in sorted(Path(train_dir).glob("*.json")):
        with path.open("r") as handle:
            payload = json.load(handle)
        for user_id, user_data in payload.get("user_data", {}).items():
            counts[str(user_id)] = len(user_data.get("x", []))
    return counts


def count_clients_from_pt(train_dir):
    counts = {}
    for path in sorted(Path(train_dir).glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for user_id, n in zip(payload.get("client_ids", []), payload.get("num_samples", [])):
            counts[str(user_id)] = int(n)
    return counts


def select_dataset_clients(data_root, num_clients, min_samples):
    train_dir = Path(data_root) / "train"
    counts = count_clients_from_json(train_dir)
    if not counts:
        counts = count_clients_from_pt(train_dir)
    if not counts:
        raise RuntimeError(f"No JSON or PT client data found under {train_dir}")
    eligible = sorted(user_id for user_id, count in counts.items() if count >= min_samples)
    if len(eligible) < num_clients:
        raise RuntimeError(
            f"Requested {num_clients} clients, but only {len(eligible)} clients "
            f"have at least {min_samples} training samples under {data_root}."
        )
    selected = eligible[:num_clients]
    return selected, counts


def select_fedscale_profiles(profile_path, num_clients, sample_mode, seed):
    raw = load_fedscale_payload(profile_path)
    records = _as_profile_mapping(raw)
    available = sorted(records.keys(), key=_numeric_sort_key)
    if not available:
        raise RuntimeError(f"No FedScale profile records found in {profile_path}")

    zero_based = [_canonical_client_id(i) for i in range(num_clients)]
    one_based = [_canonical_client_id(i + 1) for i in range(num_clients)]
    profile_id_set = set(available)
    if all(client_id in profile_id_set for client_id in zero_based):
        selected = zero_based
        mode = "zero_based_direct"
    elif all(client_id in profile_id_set for client_id in one_based):
        selected = one_based
        mode = "one_based_direct"
    elif sample_mode == "sorted":
        selected = available[:num_clients]
        mode = "sorted"
    else:
        rng = np.random.default_rng(int(seed) + 7919)
        replace = len(available) < num_clients
        selected = rng.choice(available, size=num_clients, replace=replace).tolist()
        mode = f"random_seed_{seed}"
    selected = [_canonical_client_id(client_id) for client_id in selected]
    return selected, records, mode


def write_dataset_csv(path, dataset_client_ids, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_client_index", "dataset_client_id", "train_samples"])
        writer.writeheader()
        for idx, client_id in enumerate(dataset_client_ids):
            writer.writerow({
                "local_client_index": idx,
                "dataset_client_id": client_id,
                "train_samples": counts.get(client_id, 0),
            })


def write_mapping_csv(path, dataset_client_ids, counts, fedscale_client_ids, fedscale_records, mapping_mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "local_client_index",
                "dataset_client_id",
                "train_samples",
                "fedscale_profile_client_id",
                "fedscale_mapping_mode",
            ],
        )
        writer.writeheader()
        for idx, dataset_client_id in enumerate(dataset_client_ids):
            profile_id = fedscale_client_ids[idx] if idx < len(fedscale_client_ids) else ""
            writer.writerow({
                "local_client_index": idx,
                "dataset_client_id": dataset_client_id,
                "train_samples": counts.get(dataset_client_id, 0),
                "fedscale_profile_client_id": profile_id,
                "fedscale_mapping_mode": mapping_mode,
            })


def write_profile_subset(path, fedscale_client_ids, fedscale_records):
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = {
        client_id: fedscale_records[client_id]
        for client_id in fedscale_client_ids
        if client_id in fedscale_records
    }
    with path.open("wb") as handle:
        pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["femnist", "gspeech"], required=True)
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/ and test/.")
    parser.add_argument("--num-clients", type=int, default=1000)
    parser.add_argument("--min-samples-per-client", type=int, default=0)
    parser.add_argument("--fedscale-profile", default="", help="Optional FedScale client_device_capacity file.")
    parser.add_argument("--fedscale-profile-sample", choices=["random", "sorted"], default="random")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="data/client_selection")
    args = parser.parse_args()

    dataset_ids, counts = select_dataset_clients(
        args.data_root,
        num_clients=int(args.num_clients),
        min_samples=int(args.min_samples_per_client),
    )

    out_dir = Path(args.out_dir)
    dataset_csv = out_dir / f"{args.dataset}_selected_clients.csv"
    write_dataset_csv(dataset_csv, dataset_ids, counts)
    print(f"Wrote {dataset_csv}")

    if args.fedscale_profile:
        fedscale_ids, records, mode = select_fedscale_profiles(
            args.fedscale_profile,
            num_clients=len(dataset_ids),
            sample_mode=args.fedscale_profile_sample,
            seed=int(args.seed),
        )
        mapping_csv = out_dir / f"{args.dataset}_fedscale_client_mapping_seed{args.seed}.csv"
        profile_subset = out_dir / f"{args.dataset}_fedscale_profile_subset_seed{args.seed}.pkl"
        write_mapping_csv(mapping_csv, dataset_ids, counts, fedscale_ids, records, mode)
        write_profile_subset(profile_subset, fedscale_ids, records)
        print(f"Wrote {mapping_csv}")
        print(f"Wrote {profile_subset}")


if __name__ == "__main__":
    main()
