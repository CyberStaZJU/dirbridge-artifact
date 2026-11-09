#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 数据及缓存相对外部 STATE_ROOT/data；OUT_DIR 显式路径仍由用户控制。
STATE_ROOT="${DIRBRIDGE_STATE_ROOT:-${HOME}/.local/state/dirbridge}"
mkdir -p "${STATE_ROOT}"
cd "${STATE_ROOT}"
STATE_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_DIR="${OUT_DIR:-${STATE_ROOT}/processed_metrics/fedscale_femnist}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ALGORITHMS="${ALGORITHMS:-FedBuff CA2FL FedBuffMALight CASA FADAS FedASMU DirBridge}"
FED_PROFILE="${FED_PROFILE:-fedscale_device_info/client_device_capacity}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(mktemp -d "${OUT_DIR}/run.XXXXXXXX")"

for seed in ${SEEDS}; do
  for algo in ${ALGORITHMS}; do
    extra=""
    if [ "${algo}" = "DirBridge" ]; then
      extra="--dirbridge_sketch_dim 2048"
    fi
    "${PYTHON_BIN}" "${ROOT_DIR}/main_fed.py" --dataset femnist --algo "${algo}" --distribution noniid --alpha 0.5 --random_cost fedscale_trace --fedscale_client_profile_path "${FED_PROFILE}" --num_users 1000 --concurrency 400 --buffer_size 100 --total_rounds 500 --model cnn --lr 0.01 --local_bs 50 --local_period 10 --interval 1 --seed "${seed}" --output_dir "${OUT_DIR}/acc" --system_metrics_log_dir "${OUT_DIR}/metrics" --run_tag fedscale_femnist ${extra}
  done
done
