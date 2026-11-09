#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 数据及缓存相对外部 STATE_ROOT/data；OUT_ROOT 等显式路径仍由用户控制。
STATE_ROOT="${DIRBRIDGE_STATE_ROOT:-${HOME}/.local/state/dirbridge}"
mkdir -p "${STATE_ROOT}"
cd "${STATE_ROOT}"
STATE_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SEEDS="${SEEDS:-1 2 3 4 5}"
OUT_ROOT="${OUT_ROOT:-${STATE_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
ACC_ROOT="${ACC_ROOT:-${OUT_ROOT}/test_acc}"
mkdir -p "${LOG_ROOT}" "${ACC_ROOT}"

run_one() {
  local table="$1"; shift
  local setting="$1"; shift
  local dataset="$1"; shift
  local algo="$1"; shift
  local label="$1"; shift
  local seed="$1"; shift
  local log_dir="${LOG_ROOT}/${table}/${setting}/${label}/seed=${seed}"
  local acc_dir="${ACC_ROOT}/${table}/${setting}/${label}/seed=${seed}"
  mkdir -p "${log_dir}" "${acc_dir}"
  for previous in "${acc_dir}"/*test_acc.txt "${log_dir}/stdout.log" "${log_dir}/stderr.log"; do
    if [[ -e "${previous}" ]]; then
      printf '输出已存在：%s\n' "${previous}" >&2
      exit 1
    fi
  done
  set -o noclobber
  echo "[${table}] setting=${setting} dataset=${dataset} algo=${algo} label=${label} seed=${seed}"
  "${PYTHON_BIN}" "${ROOT_DIR}/main_fed.py" "$@"     --dataset "${dataset}"     --algo "${algo}"     --seed "${seed}"     --output_dir "${acc_dir}"     > "${log_dir}/stdout.log" 2> "${log_dir}/stderr.log"
}

# Table 3: quantitative summary of raw-buffer direction skew Phi_t.
TABLE="table3"
SKEW_ROOT="${SKEW_ROOT:-${OUT_ROOT}/direction_skew}"
mkdir -p "${SKEW_ROOT}"
for seed in ${SEEDS}; do
  run_one "${TABLE}" "cifar_alpha0.1_dirskew" "cifar" "DirBridge" "DirBridge" "${seed}"     --distribution noniid --alpha 0.1 --random_cost dir-skew     --num_users 100 --concurrency 40 --buffer_size 10 --total_rounds 500     --model resnet --lr 0.01 --local_bs 100 --local_period 10 --interval 1 --bn_buffer_policy endpoint_mean     --direction_skew_log_dir "${SKEW_ROOT}/cifar_alpha0.1/seed=${seed}" --direction_skew_log_every 1
  run_one "${TABLE}" "femnist_fedscale" "femnist" "DirBridge" "DirBridge" "${seed}"     --distribution noniid --alpha 0.5 --random_cost fedscale_trace     --fedscale_client_profile_path "${FED_PROFILE:-fedscale_device_info/client_device_capacity}"     --num_users 1000 --concurrency 400 --buffer_size 100 --total_rounds 500     --model cnn --lr 0.01 --local_bs 50 --local_period 10 --interval 1 --bn_buffer_policy endpoint_mean     --direction_skew_log_dir "${SKEW_ROOT}/femnist_fedscale/seed=${seed}" --direction_skew_log_every 1
done
