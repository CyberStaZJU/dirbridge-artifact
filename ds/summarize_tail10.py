#!/usr/bin/env python3
"""Summarize tail-k test accuracy by environment and algorithm."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PATTERN = re.compile(
    r"^(?P<dataset>.+)-alpha(?P<alpha>[^-]+)-100users-"
    r"(?P<concurrency>\d+)Mc-(?P<buffer>\d+)Buffer/"
    r"(?P<algo>[^/]+)/seed=(?P<seed>\d+)/.*-test_acc\.txt$"
)


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


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


def format_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.2f} +/- {std_value:.2f}"


def default_metric_name(gap: int) -> str:
    return f"tail{gap}"


def default_metric_title(gap: int) -> str:
    return f"Tail-{gap}"


def collect_runs(root: Path, gap: int) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(root.glob("*-alpha*-100users-*Mc-*Buffer/*/seed=*/*test_acc.txt")):
        rel = path.relative_to(root).as_posix()
        match = PATTERN.match(rel)
        if not match:
            continue

        values = read_accuracy_file(path)
        if not values:
            continue

        item = match.groupdict()
        final_window = values[-gap:]
        runs.append(
            {
                **item,
                "seed": int(item["seed"]),
                "alpha_num": float(item["alpha"]),
                "concurrency": int(item["concurrency"]),
                "buffer": int(item["buffer"]),
                "rounds": len(values),
                "window": mean(final_window),
                "final": values[-1],
                "best": max(values),
                "path": str(path),
            }
        )
    return runs


def summarize(runs: list[dict[str, object]], metric_name: str) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        key = (
            run["dataset"],
            run["alpha"],
            run["alpha_num"],
            run["concurrency"],
            run["buffer"],
            run["algo"],
        )
        groups[key].append(run)

    rows: list[dict[str, object]] = []
    for key, group_runs in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][3], item[0][4], item[0][5])
    ):
        dataset, alpha, _alpha_num, concurrency, buffer, algo = key
        group_runs = sorted(group_runs, key=lambda run: int(run["seed"]))

        metric_values = [float(run["window"]) for run in group_runs]
        final_values = [float(run["final"]) for run in group_runs]
        best_values = [float(run["best"]) for run in group_runs]
        rounds = [int(run["rounds"]) for run in group_runs]
        seeds = [int(run["seed"]) for run in group_runs]

        metric_mean = mean(metric_values)
        metric_std = sample_std(metric_values)
        final_mean = mean(final_values)
        final_std = sample_std(final_values)
        best_mean = mean(best_values)
        best_std = sample_std(best_values)

        rows.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "concurrency": concurrency,
                "buffer_size": buffer,
                "algo": algo,
                f"{metric_name}_mean": metric_mean,
                f"{metric_name}_std": metric_std,
                f"{metric_name}_mean_pm_std": format_pm(metric_mean, metric_std),
                "final_mean": final_mean,
                "final_std": final_std,
                "final_mean_pm_std": format_pm(final_mean, final_std),
                "best_mean": best_mean,
                "best_std": best_std,
                "best_mean_pm_std": format_pm(best_mean, best_std),
                "seed_count": len(seeds),
                "seeds": " ".join(map(str, seeds)),
                "rounds_min": min(rounds),
                "rounds_max": max(rounds),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path, metric_name: str) -> None:
    fieldnames = [
        "dataset",
        "alpha",
        "concurrency",
        "buffer_size",
        "algo",
        f"{metric_name}_mean",
        f"{metric_name}_std",
        f"{metric_name}_mean_pm_std",
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


def write_markdown(
    rows: list[dict[str, object]],
    path: Path,
    label: str,
    metric_name: str,
    metric_title: str,
    gap: int,
) -> None:
    by_env: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_env[
            (
                row["dataset"],
                row["alpha"],
                float(str(row["alpha"])),
                row["concurrency"],
                row["buffer_size"],
            )
        ].append(row)

    with path.open("w") as handle:
        handle.write(f"# {metric_title} Mean +/- Std by Environment and Algorithm ({label})\n\n")
        handle.write(
            f"Each seed is first reduced to the mean of its actual last {gap} accuracy values. "
            "The table reports mean +/- sample std across seeds.\n"
        )
        for dataset, alpha, _alpha_num, concurrency, buffer in sorted(
            by_env, key=lambda env: (env[0], env[2], env[3], env[4])
        ):
            handle.write(f"\n## {dataset} alpha={alpha} concurrency={concurrency} buffer={buffer}\n\n")
            handle.write(
                f"| Algorithm | {metric_title} mean +/- std | Final mean +/- std | Best mean +/- std | Seeds | Rounds |\n"
            )
            handle.write("|---|---:|---:|---:|---:|---:|\n")
            env_rows = by_env[(dataset, alpha, _alpha_num, concurrency, buffer)]
            for row in sorted(env_rows, key=lambda item: str(item["algo"])):
                rounds = (
                    str(row["rounds_min"])
                    if row["rounds_min"] == row["rounds_max"]
                    else f"{row['rounds_min']}-{row['rounds_max']}"
                )
                handle.write(
                    f"| {row['algo']} | {row[f'{metric_name}_mean_pm_std']} | "
                    f"{row['final_mean_pm_std']} | {row['best_mean_pm_std']} | "
                    f"{row['seed_count']} ({row['seeds']}) | {rounds} |\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/processed_metrics/dir-skew", help="Experiment root to scan.")
    parser.add_argument("--gap", type=int, default=10, help="Number of final rounds per seed.")
    parser.add_argument("--metric-name", help="CSV column prefix for the final-window metric.")
    parser.add_argument("--metric-title", help="Markdown display title for the final-window metric.")
    parser.add_argument(
        "--out-prefix",
        default="tail10_mean_std_by_env_algo_dirskew",
        help="Output path prefix without extension.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    metric_name = args.metric_name or default_metric_name(args.gap)
    metric_title = args.metric_title or default_metric_title(args.gap)
    rows = summarize(collect_runs(root, args.gap), metric_name)
    out_prefix = Path(args.out_prefix)
    write_csv(rows, out_prefix.with_suffix(".csv"), metric_name)
    write_markdown(rows, out_prefix.with_suffix(".md"), root.name, metric_name, metric_title, args.gap)
    print(f"wrote {out_prefix.with_suffix('.csv')} ({len(rows)} rows)")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
