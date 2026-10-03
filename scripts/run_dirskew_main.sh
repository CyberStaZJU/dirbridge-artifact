#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-artifacts/processed_metrics/dirskew_main}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ALGORITHMS="${ALGORITHMS:-FedBuff CA2FL FedBuffMALight CASA FADAS FedASMU DirBridge}"
mkdir -p "${OUT_DIR}"

for seed in ${SEEDS}; do
  for algo in ${ALGORITHMS}; do
    extra=""
    if [ "${algo}" = "DirBridge" ]; then
      extra="--dirbridge_sketch_dim 2048"
    fi
    python main_fed.py --dataset cifar --algo "${algo}" --distribution noniid --alpha 0.5 --random_cost dir-skew --num_users 100 --concurrency 40 --buffer_size 10 --total_rounds 500 --model resnet --lr 0.01 --local_bs 100 --local_period 10 --interval 1 --seed "${seed}" --output_dir "${OUT_DIR}/acc" --system_metrics_log_dir "${OUT_DIR}/metrics" --run_tag dirskew_cifar_a05 ${extra}
  done
done
