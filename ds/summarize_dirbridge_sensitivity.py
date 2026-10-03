#!/usr/bin/env python3
"""Summarize DirBridge hyperparameter sensitivity runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


SWEEPS: Dict[str, Tuple[str, List[str], str]] = {
    "beta": ("EMA decay beta", ["0.5", "0.7", "0.9", "0.95", "0.99"], "0.9"),
    "dim": ("Sketch dimension d", ["16", "32", "64", "128", "256", "512"], "64"),
}


def fraction_token(value: str) -> str:
    return str(value).replace(".", "p")


def is_default_label_primary_fraction(value: str) -> bool:
    return abs(float(value) - 0.5) < 1e-12


def env_dir(
    dataset: str,
    alpha: str,
    concurrency: int,
    buffer_size: int,
    distribution: str = "noniid",
    label_primary_fraction: str = "0.5",
) -> str:
    if distribution == "label_correlated":
        if is_default_label_primary_fraction(label_primary_fraction):
            return f"{dataset}-dir-skew-100users-{concurrency}Mc-{buffer_size}Buffer"
        return (
            f"{dataset}-label_correlated_p{fraction_token(label_primary_fraction)}"
            f"-dir-skew"
            f"-100users-{concurrency}Mc-{buffer_size}Buffer"
        )
    if distribution == "label_block":
        return f"{dataset}-label_block-100users-{concurrency}Mc-{buffer_size}Buffer"
    return f"{dataset}-alpha{alpha}-100users-{concurrency}Mc-{buffer_size}Buffer"


def first_result_file(seed_dir: Path) -> Optional[Path]:
    files = sorted(seed_dir.glob("*-test_acc.txt"))
    return files[-1] if files else None


def read_accuracy_file(path: Path) -> List[float]:
    values: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def sample_std(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def collect_runs(
    root: Path,
    env_name: str,
    variant_dir: str,
    gap: int,
) -> List[dict]:
    base_dir = root / env_name / variant_dir
    runs: List[dict] = []

    for seed_dir in sorted(base_dir.glob("seed=*")):
        try:
            seed = int(seed_dir.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue

        result_file = first_result_file(seed_dir)
        if result_file is None:
            continue

        values = read_accuracy_file(result_file)
        if not values:
            continue

        runs.append(
            {
                "seed": seed,
                "rounds": len(values),
                "tail": mean(values[-gap:]),
                "final": values[-1],
                "best": max(values),
                "path": str(result_file),
            }
        )

    return sorted(runs, key=lambda item: int(item["seed"]))


def choose_variant_dir(root: Path, env_name: str, sweep: str, value: str, base_value: str) -> str:
    variant_dir = f"DirBridge_{sweep}={value}"
    if (root / env_name / variant_dir).exists():
        return variant_dir
    if value == base_value:
        return "DirBridge_base"
    return variant_dir


def summarize_runs(sweep: str, title: str, value: str, variant_dir: str, runs: List[dict]) -> dict:
    if not runs:
        return {
            "sweep": sweep,
            "sweep_title": title,
            "value": value,
            "variant_dir": variant_dir,
            "tail_mean": "",
            "tail_std": "",
            "tail_mean_pm_std": "missing",
            "final_mean": "",
            "final_std": "",
            "final_mean_pm_std": "missing",
            "best_mean": "",
            "best_std": "",
            "best_mean_pm_std": "missing",
            "seed_count": 0,
            "seeds": "",
            "rounds_min": "",
            "rounds_max": "",
        }

    tail_values = [float(run["tail"]) for run in runs]
    final_values = [float(run["final"]) for run in runs]
    best_values = [float(run["best"]) for run in runs]
    seeds = [int(run["seed"]) for run in runs]
    rounds = [int(run["rounds"]) for run in runs]

    tail_mean = mean(tail_values)
    tail_std = sample_std(tail_values)
    final_mean = mean(final_values)
    final_std = sample_std(final_values)
    best_mean = mean(best_values)
    best_std = sample_std(best_values)

    return {
        "sweep": sweep,
        "sweep_title": title,
        "value": value,
        "variant_dir": variant_dir,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_mean_pm_std": f"{tail_mean:.2f} +/- {tail_std:.2f}",
        "final_mean": final_mean,
        "final_std": final_std,
        "final_mean_pm_std": f"{final_mean:.2f} +/- {final_std:.2f}",
        "best_mean": best_mean,
        "best_std": best_std,
        "best_mean_pm_std": f"{best_mean:.2f} +/- {best_std:.2f}",
        "seed_count": len(seeds),
        "seeds": " ".join(map(str, seeds)),
        "rounds_min": min(rounds),
        "rounds_max": max(rounds),
    }


def write_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "sweep",
        "sweep_title",
        "value",
        "variant_dir",
        "tail_mean",
        "tail_std",
        "tail_mean_pm_std",
        "final_mean",
        "final_std",
        "final_mean_pm_std",
        "best_mean",
        "best_std",
        "best_mean_pm_std",
        "seed_count",
        "seeds",
        "rounds_min",
        "rounds_max",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[dict], path: Path, args: argparse.Namespace) -> None:
    by_sweep: Dict[str, List[dict]] = {}
    for row in rows:
        by_sweep.setdefault(str(row["sweep"]), []).append(row)

    with path.open("w") as handle:
        handle.write("# DirBridge Hyperparameter Sensitivity\n\n")
        handle.write(
            f"Environment: {args.dataset} {args.distribution}, "
            f"alpha={args.alpha}, p={args.label_primary_fraction}, "
            f"{Path(args.root).name}, concurrency={args.concurrency}, "
            f"buffer={args.buffer_size}. Each seed is reduced to tail-{args.gap} mean.\n\n"
        )

        for sweep, (title, values, _base_value) in SWEEPS.items():
            handle.write(f"## {title}\n\n")
            handle.write("| Value | Tail mean +/- std | Final mean +/- std | Best mean +/- std | Seeds | Rounds |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|\n")
            sweep_rows = {str(row["value"]): row for row in by_sweep.get(sweep, [])}
            for value in values:
                row = sweep_rows[value]
                if row["rounds_min"] == "":
                    rounds = ""
                elif row["rounds_min"] == row["rounds_max"]:
                    rounds = str(row["rounds_min"])
                else:
                    rounds = f"{row['rounds_min']}-{row['rounds_max']}"
                handle.write(
                    f"| {value} | {row['tail_mean_pm_std']} | "
                    f"{row['final_mean_pm_std']} | {row['best_mean_pm_std']} | "
                    f"{row['seed_count']} ({row['seeds']}) | {rounds} |\n"
                )
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/processed_metrics/sensitivity/dir-skew")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--distribution", default="noniid")
    parser.add_argument("--alpha", default="0.1")
    parser.add_argument("--label-primary-fraction", default="0.5")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--out-prefix", default="dirbridge_sensitivity")
    args = parser.parse_args()

    root = Path(args.root)
    env_name = env_dir(
        args.dataset,
        args.alpha,
        args.concurrency,
        args.buffer_size,
        args.distribution,
        args.label_primary_fraction,
    )

    rows: List[dict] = []
    for sweep, (title, values, base_value) in SWEEPS.items():
        for value in values:
            variant_dir = choose_variant_dir(root, env_name, sweep, value, base_value)
            runs = collect_runs(root, env_name, variant_dir, args.gap)
            rows.append(summarize_runs(sweep, title, value, variant_dir, runs))

    out_prefix = Path(args.out_prefix)
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"), args)
    print(f"wrote {out_prefix.with_suffix('.csv')} ({len(rows)} rows)")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
