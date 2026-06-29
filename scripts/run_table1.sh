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

# Table 1: main 100-client image-dataset results under Dir-Skew latency.
TABLE="table1"
ALGOS="${ALGOS:-DirBridge FedBuff FedBuffMALight CASA CA2FL FADAS FedASMU}"
run_image_setting() {
  local dataset="$1" alpha="$2" setting="$1_alpha$2"
  for seed in ${SEEDS}; do
    for algo in ${ALGOS}; do
      run_one "${TABLE}" "${setting}" "${dataset}" "${algo}" "${algo}" "${seed}"         --distribution noniid --alpha "${alpha}" --random_cost mild_label_correlated_hierarchical         --num_users 100 --concurrency 40 --buffer_size 10 --total_rounds 500         --model resnet --lr 0.01 --local_bs 100 --local_period 10 --interval 1
    done
  done
}
run_image_setting cifar 0.1
run_image_setting cifar 0.5
run_image_setting cifar100 0.1
run_image_setting cifar100 0.5
run_image_setting tinyimagenet 0.5
