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

The captured CUDA reference environment is recorded in `requirements.txt` and `environment.yml`. Create an environment from these files, then install the dependencies:

```bash
pip install -r requirements.txt
```

The captured reference versions are also recorded in `requirements.txt` and `environment.yml`:

- Python 3.9.23
- NumPy 2.0.2
- SciPy 1.13.1
- PyTorch 2.8.0+cu128
- TorchVision 0.23.0+cu128
- CUDA 12.8, cuDNN 91002
- Matplotlib 3.9.2
- scikit-learn 1.6.1
- PyYAML 6.0.3
- Pillow 11.3.0
- tqdm 4.67.1
- pandas 2.3.3
- GPU: NVIDIA GeForce RTX 5090 D

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
  --min-samples-per-client 30 \
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

## 6. Experiment Entry Points

### Main experiments

| Experiment | Entry point | Setting |
|---|---|---|
| Main image results | `scripts/run_table1.sh` | CIFAR-10/100 at alpha 0.1/0.5; TinyImageNet at alpha 0.5; seven methods, five seeds |
| Natural FedScale | `scripts/run_table2.sh` | FEMNIST and GSpeech; 1,000 clients, concurrency 400, buffer 100 |
| Raw-buffer skew | `scripts/run_table3.sh` | CIFAR-10 Dir-Skew and FEMNIST FedScale |
| Systems and simulated time | `scripts/run_table4.sh`, `scripts/run_table5.sh` | CIFAR-10 T50 and FEMNIST T80; per-round runtime and reclustering time |
| Component ablation | `scripts/run_table6.sh` | Full, no EMA cache, random grouping, no staleness filter |

```bash
bash scripts/run_table1.sh
SEEDS="1" ALGOS="DirBridge FedBuff" bash scripts/run_table1.sh
```

`PYTHON_BIN`, `SEEDS`, `ALGOS`, `OUT_ROOT`, and `FED_PROFILE` can be overridden. The default runtime directory is `${HOME}/.local/state/dirbridge`; set `DIRBRIDGE_STATE_ROOT` to use another location. Dataset paths are relative to this runtime directory. Keep datasets and generated outputs outside the source tree.

### Supplementary experiments

`scripts/run_revision_experiments.py` prints the resolved commands by default. Add `--execute` to run a suite. Each execution creates a separate batch directory with per-run manifests and logs.

| Suite | Protocol | Runs |
|---|---|---:|
| `tuning-ca2fl` | CIFAR-10 heldout search: 4 local rates x 3 server rates | 12 |
| `tuning-fadas` | 4 local rates x 4 server rates x delay adaptation on/off | 32 |
| `tuned-validation` | Two datasets x two alpha values x four methods x five seeds | 80 |
| `online-init` | Full/count, online/arrival-frequency, online/unique-client | 15 |
| `profile-coupling` | FEMNIST/GSpeech with controlled label-to-device-class coupling | 70 |
| `group-sketch` | K0 in {1,2,4,7,12,16}; d in {512,2048,8192}; shared default counted once | 40 |
| `beta-retention` | EMA retention on the three image datasets | 45 |
| `recluster-period` | Reclustering period on the three image datasets | 45 |

```bash
python scripts/run_revision_experiments.py --suite tuned-validation
python scripts/run_revision_experiments.py --suite tuned-validation --execute
python scripts/run_revision_experiments.py --suite profile-coupling \
  --fedscale-profile /absolute/path/to/client_device_capacity --execute
```

The beta/R suites reconstruct the figure grids with the current endpoint policy. The default beta/R point appears in both sweeps. Protocol values are in `configs/revision_experiments.json`.

Weak uses a fixed offline direction reference, client-indexed service trajectories, and the seven methods on CIFAR-10 at alpha 0.1/0.5. Its FIFO diagnostic selects from the same pre-update queue state.

```bash
python scripts/run_weak_experiments.py
python scripts/run_weak_experiments.py --execute --output-root /absolute/runtime/root --gpu 0
python scripts/summarize_weak.py --root /absolute/runtime/root/weak-UUID
```

The analytical server-state budget is available independently of training:

```bash
python scripts/server_state_budget.py
```

## 7. Results and Metrics

`ds/summarize_paper_runs.py` reads table-script outputs and revision manifests, checks the requested rounds and seed coverage, and reports tail-10 accuracy with sample standard deviation across seeds. It also summarizes simulated threshold time, relative per-round runtime, reclustering events, and raw-buffer skew when the corresponding logs are available.

```bash
python ds/summarize_paper_runs.py --root /absolute/runtime/root --include-direction-skew
python ds/summarize_paper_runs.py --root /absolute/runtime/root/revision/tuned-validation \
  --channel validation
python ds/summarize_paper_runs.py --root /absolute/runtime/root/revision/tuning-ca2fl \
  --expected-seeds 1 --channel validation
```

Use the shared `OUT_ROOT` as the table-summary root so the default accuracy and metric subdirectories can be joined. Each revision run's manifest supplies its metric paths and round count.

The existing `summarize_tail10.py`, ablation/sensitivity utilities, and significance script remain available for their original directory layouts.

## 8. Protocol Notes

- Heldout validation uses 5,000 CIFAR training examples selected without replacement by NumPy `default_rng(20260922)`, sorted before constructing the split. These examples are removed from client training partitions; test and validation metrics are recorded separately.
- FedBuff/DirBridge use effective server rate 1.0 in the tuned-validation suite; the historical configuration listed 0.5 while the implementation applied the full update. FADAS uses weight decay 1e-4 in this suite.
- Online initialization, profile coupling, and K0/d configurations retain their recorded `legacy_delta` BN policy. Some historical evaluations under this policy had nonfinite losses; evaluation logs expose loss, logits, inputs, and prediction validity. Endpoint-based suites use `endpoint_mean`.
- The raw-buffer monitor uses fixed label-histogram groups; Weak uses its fixed model-update direction reference. These are separate measurements.
- `round_runtime_sec` measures local-training computation and server processing, excluding evaluation, logging, and garbage collection. Simulated time and reclustering time are recorded separately. The analytical state budget counts tensor slots, whereas runtime memory fields measure process/device usage.

## 9. Verification

```bash
bash scripts/check_artifact.sh
python -m pytest tests -q
```

The first command checks Python and shell syntax; the tests exercise aggregation, BN endpoints, online initialization, validation splits, profile assignment, experiment definitions, and result handling. For numerical comparisons, retain the run manifest, dataset selection, software versions, and hardware details with the generated results.

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
