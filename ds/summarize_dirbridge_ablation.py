#!/usr/bin/env python3
"""Summarize DirBridge ablation runs for Table 4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


DEFAULT_VARIANTS = [
    ("DirBridge_full", "Full"),
    ("DirBridge_wo_ema_cache", "w/o EMA cache"),
    ("DirBridge_wo_direction_grouping", "w/o direction grouping"),
    ("DirBridge_wo_staleness_filter", "w/o staleness filter"),
]


def read_accuracy_file(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


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


def first_result_file(seed_dir: Path) -> Path | None:
    files = sorted(seed_dir.glob("*-test_acc.txt"))
    return files[-1] if files else None


def collect_variant(
    root: Path,
    variant_dir: str,
    dataset: str,
    alpha: str,
    concurrency: int,
    buffer_size: int,
    distribution: str,
    label_primary_fraction: str,
    gap: int,
) -> list[dict[str, object]]:
    base_dir = root / env_dir(
        dataset,
        alpha,
        concurrency,
        buffer_size,
        distribution,
        label_primary_fraction,
    ) / variant_dir
    runs: list[dict[str, object]] = []

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

        tail_values = values[-gap:]
        runs.append(
            {
                "seed": seed,
                "rounds": len(values),
                "tail": mean(tail_values),
                "final": values[-1],
                "best": max(values),
                "path": str(result_file),
            }
        )

    return sorted(runs, key=lambda item: int(item["seed"]))


def summarize_runs(label: str, variant_dir: str, runs: list[dict[str, object]]) -> dict[str, object]:
    tail_values = [float(run["tail"]) for run in runs]
    final_values = [float(run["final"]) for run in runs]
    best_values = [float(run["best"]) for run in runs]
    seeds = [int(run["seed"]) for run in runs]
    rounds = [int(run["rounds"]) for run in runs]

    if not runs:
        return {
            "variant": label,
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

    tail_mean = mean(tail_values)
    tail_std = sample_std(tail_values)
    final_mean = mean(final_values)
    final_std = sample_std(final_values)
    best_mean = mean(best_values)
    best_std = sample_std(best_values)

    return {
        "variant": label,
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


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "variant",
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


def write_markdown(rows: list[dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    with path.open("w") as handle:
        handle.write("# DirBridge Ablation Summary\n\n")
        handle.write(
            f"Environment: {args.dataset} {args.distribution}, "
            f"alpha={args.alpha}, p={args.label_primary_fraction}, "
            f"{Path(args.ablation_root).name}, concurrency={args.concurrency}, "
            f"buffer={args.buffer_size}. Each seed is reduced to tail-{args.gap} mean.\n\n"
        )
        handle.write("| Variant | Tail mean +/- std | Final mean +/- std | Best mean +/- std | Seeds | Rounds |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if row["rounds_min"] == "":
                rounds = ""
            elif row["rounds_min"] == row["rounds_max"]:
                rounds = str(row["rounds_min"])
            else:
                rounds = f"{row['rounds_min']}-{row['rounds_max']}"
            handle.write(
                f"| {row['variant']} | {row['tail_mean_pm_std']} | "
                f"{row['final_mean_pm_std']} | {row['best_mean_pm_std']} | "
                f"{row['seed_count']} ({row['seeds']}) | {rounds} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", default="artifacts/processed_metrics/ablation/dir-skew")
    parser.add_argument("--baseline-root", default="artifacts/processed_metrics/dir-skew")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--distribution", default="noniid")
    parser.add_argument("--alpha", default="0.1")
    parser.add_argument("--label-primary-fraction", default="0.5")
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--out-prefix", default="dirbridge_ablation_table4")
    args = parser.parse_args()

    ablation_root = Path(args.ablation_root)
    baseline_root = Path(args.baseline_root)

    rows: list[dict[str, object]] = []
    for variant_dir, label in DEFAULT_VARIANTS:
        root = baseline_root if variant_dir == "DirBridge_full" else ablation_root
        source_dir = "DirBridge" if variant_dir == "DirBridge_full" else variant_dir
        runs = collect_variant(
            root=root,
            variant_dir=source_dir,
            dataset=args.dataset,
            alpha=args.alpha,
            concurrency=args.concurrency,
            buffer_size=args.buffer_size,
            distribution=args.distribution,
            label_primary_fraction=args.label_primary_fraction,
            gap=args.gap,
        )
        rows.append(summarize_runs(label, variant_dir, runs))

    out_prefix = Path(args.out_prefix)
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"), args)
    print(f"wrote {out_prefix.with_suffix('.csv')} ({len(rows)} rows)")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
