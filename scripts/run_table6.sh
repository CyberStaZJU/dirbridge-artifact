#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/jczn2/.conda/envs/yibo/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SEEDS="${SEEDS:-1 2 3 4 5}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts}"
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
  echo "[${table}] setting=${setting} dataset=${dataset} algo=${algo} label=${label} seed=${seed}"
  "${PYTHON_BIN}" main_fed.py "$@"     --dataset "${dataset}"     --algo "${algo}"     --seed "${seed}"     --output_dir "${acc_dir}"     > "${log_dir}/stdout.log" 2> "${log_dir}/stderr.log"
}

# Table 6: DirBridge component ablation under Dir-Skew at alpha=0.5.
TABLE="table6"
DATASETS="${DATASETS:-cifar cifar100 tinyimagenet}"
VARIANTS="${VARIANTS:-none no_ema_cache random_grouping no_staleness_filter}"
for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    for variant in ${VARIANTS}; do
      run_one "${TABLE}" "${dataset}_alpha0.5" "${dataset}" "DirBridge" "DirBridge_${variant}" "${seed}"         --distribution noniid --alpha 0.5 --random_cost mild_label_correlated_hierarchical         --num_users 100 --concurrency 40 --buffer_size 10 --total_rounds 500         --model resnet --lr 0.01 --local_bs 100 --local_period 10 --interval 1         --dirbridge_ablation "${variant}"
    done
  done
done
