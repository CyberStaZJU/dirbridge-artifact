# DirBridge Artifact

This repository provides the code artifact for **DirBridge: Scalable Direction-Group Memory for Latency-Biased Asynchronous Federated Data Streams**.

DirBridge studies buffered asynchronous federated learning under latency-induced direction skew. The artifact contains the core implementation, selected baselines, reproduction scripts, result summarization utilities, dataset preparation notes, and configuration records used to reproduce the paper's main empirical claims.

## Artifact Scope

This artifact is intended to support the following checks:

1. **Environment check**: verify that the main FedBuff and DirBridge code paths run on a small CIFAR-10 smoke test.
2. **Main-result reproduction**: rerun the direction-skewed latency experiments and the FedScale-trace FEMNIST/GSpeech experiments.
3. **Mechanism checks**: reproduce the direction-skew diagnostics, representation-module ablations, sensitivity analysis, and significance comparisons.
4. **Configuration audit**: inspect the scripts and configuration files that define client sampling, latency simulation, group-memory settings, and result summarization.

The repository does not include raw datasets. Some full experiments are compute-intensive; use the smoke test first to validate the environment.

## 1. Repository Layout

```text
dirbridge-artifact/
├── main_fed.py                       # Main experiment entry point
├── algorithm/                        # DirBridge and baseline algorithms
├── models/                           # Model definitions and local training utilities
├── builders/                         # Dataset/model construction
├── data_reader/                      # FEMNIST and GSpeech readers
├── utils/                            # Sampling, aggregation, FedScale trace, logging utilities
├── ds/                               # Result summarization scripts
├── scripts/                          # Reproduction, preprocessing, and artifact-check scripts
├── configs/                          # Human-readable experiment settings
└── artifacts/                        # Optional processed metrics and regenerated tables
```

## 2. Environment

The experiments were developed with Python 3.8+ and PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

A GPU is recommended for full reproduction. The quick smoke test can be used to verify the environment before launching the full multi-seed experiments.

## 3. Datasets

The paper uses five datasets:

- CIFAR-10
- CIFAR-100
- TinyImageNet
- FEMNIST
- Google Speech Commands (GSpeech)

CIFAR-10 and CIFAR-100 can be downloaded by the standard torchvision loaders. TinyImageNet, FEMNIST, and GSpeech may require manual preparation depending on the local environment. The artifact does not include raw datasets.

### FEMNIST preprocessed `.pt` format

The FEMNIST reader supports both LEAF/FedScale-style JSON files and a compact `.pt` format. The default path is:

```text
./data/femnist_pt/
├── train/femnist.pt
└── test/femnist.pt
```

Each `.pt` file is a dictionary with:

```python
{
    "client_ids": list[str],
    "num_samples": list[int],
    "images": torch.FloatTensor,   # shape: (N, 784)
    "targets": torch.LongTensor,   # shape: (N,)
    "clients": torch.LongTensor,   # shape: (N,), client index per sample
}
```

To build this format from raw FEMNIST JSON files, run:

```bash
python scripts/build_femnist_pt.py \
  --raw-root <raw_femnist_root> \
  --out-root data/femnist_pt \
  --num-clients 1000 \
  --min-samples-per-client 250
```

To export the exact selected client list and the FedScale profile mapping used by the simulator, run:

```bash
python scripts/export_client_selection.py \
  --dataset femnist \
  --data-root data/femnist_pt \
  --num-clients 1000 \
  --min-samples-per-client 250 \
  --fedscale-profile fedscale_device_info/client_device_capacity \
  --seed 1 \
  --out-dir data/client_selection
```

This writes:

```text
data/client_selection/femnist_selected_clients.csv
data/client_selection/femnist_fedscale_client_mapping_seed1.csv
data/client_selection/femnist_fedscale_profile_subset_seed1.pkl
```

The script scans `<raw_femnist_root>/train/*.json`, counts each client's training samples, keeps clients with at least 250 training samples, sorts the eligible client ids, and selects the first 1000. The same selected client ids are then used for both train and test `.pt` files.

When `--dataset femnist --num_users 1000 --femnist_min_samples_per_client 250` is used with the default `data/femnist_pt` directory, the training code treats `femnist_pt` as this preselected 1000-client subset. If a different FEMNIST directory is used, the same sorted-eligible-client selection is performed at load time.

### GSpeech client selection and feature cache

GSpeech uses the same client-selection rule at load time: it scans `data/gspeech/train/*.json`, counts each client's training samples, keeps clients satisfying `--gspeech_min_samples_per_client`, sorts eligible client ids, and selects the first `--num_users` clients. The FedScale GSpeech script uses `--num_users 1000`.

To export the exact GSpeech selected client list and FedScale mapping, run:

```bash
python scripts/export_client_selection.py \
  --dataset gspeech \
  --data-root data/gspeech \
  --num-clients 1000 \
  --min-samples-per-client 0 \
  --fedscale-profile fedscale_device_info/client_device_capacity \
  --seed 1 \
  --out-dir data/client_selection
```

This writes:

```text
data/client_selection/gspeech_selected_clients.csv
data/client_selection/gspeech_fedscale_client_mapping_seed1.csv
data/client_selection/gspeech_fedscale_profile_subset_seed1.pkl
```

The GSpeech reader converts waveform entries to log-mel features. By default these features are cached under:

```text
data/gspeech/feature_cache/clients_<hash>/
```

The `<hash>` is computed from the selected client ids, so different 1000-client subsets do not share the same feature cache. Set `GSPEECH_FEATURE_CACHE=0` to disable this cache.

For the FedScale-trace experiments, place the client device capacity file under:

```text
fedscale_device_info/client_device_capacity
```

The simulator maps local dataset clients to FedScale profile clients by local client index. With 1000 local clients, it first tries profile ids `0..999`, then `1..1000`; if neither range exists, it falls back to `--fedscale_profile_sample` (`random` by default, seeded by `--seed`/`--delay_seed`). The exported `*_fedscale_client_mapping_seed*.csv` files make this mapping explicit. The optional `*_fedscale_profile_subset_seed*.pkl` files contain the exact profile records used by those mappings, so reviewers can reproduce the latency assignment without depending on the full FedScale device-capacity file.

or pass a different path with:

```bash
--fedscale_client_profile_path <path>
```

## 4. Quick Start

Run a small smoke test:

```bash
bash scripts/run_quick_smoke.sh
```

This runs FedBuff and DirBridge on a short CIFAR-10 setting with one seed. It is intended to check installation, dataset loading, logging, and the aggregation path; it is not expected to match paper-level accuracy.

## 5. Reproducing Main Results

### Direction-skewed latency results

```bash
bash scripts/run_dirskew_main.sh
```

This script runs the main Dir-Skew latency setting with five seeds and the baselines used in the paper.

### FedScale-trace FEMNIST results

```bash
bash scripts/run_fedscale_femnist.sh
```

### FedScale-trace GSpeech results

```bash
bash scripts/run_fedscale_gspeech.sh
```

## 6. Diagnostics and Additional Analyses

The `ds/` directory contains scripts for:

- tail-10 accuracy summarization
- DirBridge ablation summarization
- sensitivity analysis summarization
- significance comparison

Example:

```bash
python ds/summarize_tail10.py --help
python ds/compare_dirbridge_significance.py --help
```

## 7. Mapping to Paper Results

| Paper result type | Artifact component |
|---|---|
| Main Dir-Skew accuracy tables | `scripts/run_dirskew_main.sh`, `ds/summarize_tail10.py` |
| FedScale FEMNIST/GSpeech tables | `scripts/build_femnist_pt.py`, `scripts/export_client_selection.py`, `scripts/run_fedscale_femnist.sh`, `scripts/run_fedscale_gspeech.sh` |
| Direction-skew diagnostic curves | training logs, processed metrics, and direction-skew summarization utilities in `ds/` |
| Ablation study | `ds/summarize_dirbridge_ablation.py` |
| Sensitivity study | `ds/summarize_dirbridge_sensitivity.py` |
| Significance checks | `ds/compare_dirbridge_significance.py` |
| Reproducibility configuration | `configs/`, shell scripts in `scripts/`, and dataset/client-selection exports |

## 8. Expected Outputs

By default, scripts write logs, metrics, and accuracy traces under:

```text
artifacts/processed_metrics/
```

Generated summary tables can be placed under:

```text
artifacts/paper_tables/
```

## 9. Notes on Runtime and Reproducibility

Full five-seed reproduction across all datasets and baselines is compute-intensive. The quick smoke test is intended only to check that the code path works; it is not expected to reproduce paper-level accuracy.

For paper-level reproduction, use the shell scripts in `scripts/` as the primary entry points and keep the default seeds and delay seeds unless intentionally running additional trials. When reporting regenerated results, record the commit hash, GPU type, PyTorch/CUDA versions, and any dataset path changes.

## 10. Citation

If this artifact is used, please cite the corresponding DirBridge paper.

```bibtex
@article{chen2026dirbridge,
  title   = {DirBridge: Scalable Direction-Group Memory for Latency-Biased Asynchronous Federated Data Streams},
  author  = {Chen, Yibo and Liu, Zhizhong and Qin, Yunchuan and Tang, Zhuo and Li, Kenli},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## 11. Contact

Please open an issue in this repository for artifact questions, missing script paths, or reproduction problems.
