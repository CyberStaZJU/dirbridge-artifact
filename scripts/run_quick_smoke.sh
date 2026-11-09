#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 数据及缓存相对外部 STATE_ROOT/data；OUT_DIR 显式路径仍由用户控制。
STATE_ROOT="${DIRBRIDGE_STATE_ROOT:-${HOME}/.local/state/dirbridge}"
mkdir -p "${STATE_ROOT}"
cd "${STATE_ROOT}"
STATE_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_DIR="${OUT_DIR:-${STATE_ROOT}/processed_metrics/quick_smoke}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(mktemp -d "${OUT_DIR}/run.XXXXXXXX")"

for seed in 1; do
  for algo in FedBuff DirBridge; do
    "${PYTHON_BIN}" "${ROOT_DIR}/main_fed.py" --dataset cifar --algo "${algo}" --distribution noniid --alpha 0.5 --random_cost dir-skew --num_users 100 --concurrency 40 --buffer_size 10 --total_rounds 80 --model resnet --lr 0.01 --local_bs 100 --local_period 10 --interval 1 --bn_buffer_policy endpoint_mean --seed "${seed}" --output_dir "${OUT_DIR}/acc" --system_metrics_log_dir "${OUT_DIR}/metrics" --run_tag quick_smoke
  done
done
