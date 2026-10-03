#!/usr/bin/env python3
"""Paired significance checks for DirBridge main-result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from scipy.stats import ttest_rel


ALGOS = ["FedBuff", "CA2FL", "FedBuffMALight", "CASA", "FADAS", "DirBridge"]
DISPLAY = {"FedBuffMALight": "MA-Light"}


def read_tail10(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"no accuracy values in {path}")
    return mean(values[-10:])


def collect_algo(base: Path, algo: str) -> dict[int, float]:
    values: dict[int, float] = {}
    algo_dir = base / algo
    if not algo_dir.exists():
        return values
    for seed_dir in sorted(algo_dir.glob("seed=*")):
        try:
            seed = int(seed_dir.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        files = sorted(seed_dir.glob("*test_acc.txt"))
        if files:
            values[seed] = read_tail10(files[0])
    return values


def main_settings(root: Path) -> list[tuple[str, str, str, Path]]:
    settings: list[tuple[str, str, str, Path]] = []
    for dataset, alphas in [
        ("cifar", ["0.1", "0.5"]),
        ("cifar100", ["0.01", "0.1", "0.5"]),
        ("tinyimagenet", ["0.01", "0.1"]),
        ("femnist", ["0.5"]),
    ]:
        for alpha in alphas:
            base = root / "dir-skew" / f"{dataset}-alpha{alpha}-100users-40Mc-10Buffer"
            if base.exists():
                settings.append(("dir-skew", dataset, alpha, base))

    return settings


def summarize(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for latency, dataset, alpha, base in main_settings(root):
        data = {algo: collect_algo(base, algo) for algo in ALGOS if (base / algo).exists()}
        if "DirBridge" not in data:
            continue
        algo_means = {algo: mean(vals.values()) for algo, vals in data.items() if len(vals) >= 2}
        if "DirBridge" not in algo_means:
            continue
        best_baseline = max((algo for algo in algo_means if algo != "DirBridge"), key=algo_means.__getitem__)
        shared_seeds = sorted(set(data["DirBridge"]) & set(data[best_baseline]))
        dirbridge_values = [data["DirBridge"][seed] for seed in shared_seeds]
        baseline_values = [data[best_baseline][seed] for seed in shared_seeds]
        diffs = [a - b for a, b in zip(dirbridge_values, baseline_values)]
        test = ttest_rel(dirbridge_values, baseline_values)
        rows.append(
            {
                "latency": latency,
                "dataset": dataset,
                "alpha": alpha,
                "best_baseline": DISPLAY.get(best_baseline, best_baseline),
                "dirbridge_mean": f"{mean(dirbridge_values):.4f}",
                "baseline_mean": f"{mean(baseline_values):.4f}",
                "mean_diff": f"{mean(diffs):.4f}",
                "paired_t_pvalue": f"{test.pvalue:.6f}",
                "seeds": " ".join(map(str, shared_seeds)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/processed_metrics", help="Experiment root.")
    parser.add_argument("--out", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = summarize(Path(args.root))
    fieldnames = [
        "latency",
        "dataset",
        "alpha",
        "best_baseline",
        "dirbridge_mean",
        "baseline_mean",
        "mean_diff",
        "paired_t_pvalue",
        "seeds",
    ]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
