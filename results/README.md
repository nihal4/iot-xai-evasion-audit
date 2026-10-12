# results/

Result bundle of the reported run (Kaggle, sessions A, B, C1–C4).

| Path | Contents |
|---|---|
| `runs/<dataset>_split<k>_seed<s>/` | Seed-level results for all 60 runs (2 datasets × [20 seeds on partition 0 + 5 seeds on partitions 1 and 2]): `predictive*.csv`, `training.csv`, `attacks*.csv`, `explanations_seed.csv`, `explanations_runtime.csv`, `repeats.csv` |
| `data_audit/` | Raw-file SHA-256 hashes, duplicate and label-conflict audit (`audit_<dataset>.json`), partition summaries (`split_<dataset>_split<k>.json`) |
| `extras/` | Adversarial-training sweep, baseline ablation, original-pipeline replication path, multiclass analysis, operating-point analysis, resource profiling |
| `cross_seed_summary.csv` | Cross-seed attribution agreement (190 seed pairs per detector × explainer) |
| `logs/session_*.log` | Complete printed logs of each Kaggle session |

The tables in `runs/`, `data_audit/` and `extras/` were rebuilt from the session logs with
`python -m iotxai.report logs/session_X.log <out>`, which parses the `### BEGIN` / `### END` blocks every run prints.
They are identical to the CSV files the pipeline writes.

## Large per-run artefacts

The binary artefacts written by each session (`arrays.npz` with per-record attributions,
adversarial inputs and validation/test probabilities; `models/`; `explanations_records.csv.gz`;
`training_history.csv`; the partition manifests `partition_<dataset>_split<k>.csv.gz`;
and `environment.json`) are in the `results_session_<X>.zip` files from the Kaggle *Output* tab.
To add them, unzip each archive and copy the contents of its `results_v2/` folder into this folder;
existing files are not changed.
