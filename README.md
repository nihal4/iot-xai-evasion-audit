# iot-xai-evasion-audit

**Multi-seed audit of SHAP/LIME explanation faithfulness and stability under constrained and unconstrained adversarial evasion for IoT/IIoT intrusion detection.**

This repository contains the complete code, data-partition manifests and result artefacts for

> S. M. N. Ahmed, A. H. Khan, S. N. Sinthia.
> *A Multi-Seed Evaluation of Explanation Faithfulness and Stability under Adversarial Evasion in IoT/IIoT Intrusion Detection.*
> Discover Informatics (under revision), 2026.

## About

The study audits whether post hoc explanations of intrusion detectors stay faithful and stable when the detector is attacked, and whether such findings survive retraining.

- **Datasets.** ToN-IoT and Edge-IIoTset are analysed separately. Each is de-duplicated and screened for conflicting labels before partitioning.
- **Detectors.** Four detectors are trained with 20 independent seeds on a primary partition, and with 5 seeds on each of two additional partitions:
  - an MLP,
  - the MLP with FGSM adversarial training,
  - the MLP with PGD adversarial training,
  - XGBoost.
- **Threat models.** Each detector is attacked under three threat models:
  - *unconstrained*,
  - *numeric-only*,
  - *domain-constrained*: attacker-controllable, increase-only fields; frozen protocol fields; repair to integer and protocol invariants.
- **Attacks.** FGSM, PGD, a score-based black-box random-search attack, and transfer attacks. Every adversarial record passes a validity audit.
- **Faithfulness.** SHAP and LIME are evaluated with predicted-class comprehensiveness and sufficiency AOPC. This is computed under median, marginal and k-NN in-fill masking references, each compared with a random-ranking control.
- **Stability.** Measured between clean and attacked inputs, across repeated explainer calls, across background sets and across training seeds.
- **Statistics.** The training seed is the unit of inference. Tests are exact Wilcoxon signed-rank tests with Holm correction within pre-declared families, reported with Cohen's d_z and bootstrap 95% confidence intervals over seeds.

## Repository layout

| Path | Contents |
|---|---|
| `iotxai/` | Python package (see the module table below) |
| `notebooks/kaggle_revision_v2.ipynb` | Self-contained Kaggle notebook, generated from the package by `tools/build_notebook.py` |
| `KAGGLE_RUN_GUIDE.md` | Step-by-step guide to running the experiment on Kaggle |
| `results/` | Result bundle of the reported run (see *Result artefacts*) |
| `analysis/` | Tables, statistical tests and figures used in the paper |
| `legacy_v1/` | Aggregate outputs and notebook of the original five-seed submission, kept for traceability |
| `tests/` | Synthetic-data generator with the real column schemas, smoke test and notebook simulator |
| `tools/` | Notebook builder |

| Module | Purpose |
|---|---|
| `config.py` | All settings, dataset feature policies, mutability specification and session presets |
| `data.py` | Raw-file hashing, duplicate and label-conflict audit, de-duplication, stratified partitions, train-only preprocessing |
| `models.py` | MLP (clean / FGSM-AT / PGD-AT), XGBoost, validation-tuned threshold, calibration, metrics |
| `attacks.py` | FGSM, PGD, black-box random search, constraint projection and repair, validity audit |
| `explain.py` | SHAP and LIME, group aggregation, faithfulness (3 references + random control), stability metrics |
| `runner.py` | One run = (dataset, partition, seed); resumable; writes per-run outputs |
| `extras.py` | Adversarial-training sweep, baseline ablation, multiclass analysis, operating-point analysis, resource profiling |
| `analysis.py` | Seed-level statistics, tables and figures |
| `report.py` | Prints result tables into the log and rebuilds results from a pasted log |
| `cli.py` | Command-line entry point |

## Datasets

The raw datasets are third-party data and are **not redistributed** here. Download them from their original providers:

- **ToN-IoT**, network subset `train_test_network.csv` (UNSW Canberra): <https://research.unsw.edu.au/projects/toniot-datasets>
- **Edge-IIoTset**, file `Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv` (Ferrag et al., IEEE Access 2022, doi:10.1109/ACCESS.2022.3165809): <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot>

Before running, check that your copies are byte-identical to the ones used in the paper:

| File | SHA-256 |
|---|---|
| `train_test_network.csv` (ToN-IoT) | `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974` |
| `ML-EdgeIIoT-dataset.csv` (Edge-IIoTset) | `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` |

## Installation

```bash
git clone https://github.com/<account>/iot-xai-evasion-audit.git
cd iot-xai-evasion-audit
pip install -r requirements.txt
```

Python 3.10 or later is required. A CUDA GPU is optional, but it speeds up MLP training and SHAP considerably.

## Reproducing the experiment

Put the two CSV files anywhere below `./data` (file names are discovered automatically), or pass their paths with `--ton-csv` and `--edge-csv`.

```bash
python -m iotxai.cli --session A  --input-root ./data --out ./results   # ToN-IoT, 20 seeds
python -m iotxai.cli --session B  --input-root ./data --out ./results   # Edge-IIoTset, 20 seeds
python -m iotxai.cli --session C1 --input-root ./data --out ./results   # 2 extra partitions x 5 seeds
python -m iotxai.cli --session C2 --input-root ./data --out ./results   # AT sweep, ablation, multiclass, profiling
python -m iotxai.cli --session C3 --input-root ./data --out ./results   # FPR-matched operating points
python -m iotxai.analysis ./results --out ./analysis                    # all tables and figures
```

| Session | Content | Approx. time (Tesla T4) |
|---|---|---|
| A | ToN-IoT: 20 seeds × 4 detectors, all attacks, SHAP/LIME, repeated-call checks | 3–5 h |
| B | Edge-IIoTset: same as A | 3–5 h |
| C1 | Partition sensitivity: partitions 1 and 2 × 5 seeds, both datasets | 2–3 h |
| C2 | Adversarial-training sweep, baseline ablation, multiclass analysis, latency/memory profiling | 4–5 h |
| C3 | Attack success at false-positive-rate-matched thresholds | ~1 h |

**Resuming.** Every run writes `results/runs/<dataset>_split<k>_seed<s>/done.json`. Finished runs are skipped, so an interrupted session can simply be restarted.

**Running on Kaggle.** The reported run used Kaggle notebooks on an NVIDIA Tesla T4. To repeat it there, import `notebooks/kaggle_revision_v2.ipynb`, attach both datasets, and follow `KAGGLE_RUN_GUIDE.md`.

**Rebuilding results from a log.** Every run prints its result tables between `### BEGIN` / `### END` markers. If output files are lost, the seed-level results can be rebuilt from a saved log:

```bash
python -m iotxai.report saved_log.txt ./results_from_log
```

**Quick check without the real data** (CPU, about 3 minutes):

```bash
python tests/make_synthetic.py /tmp/synth 6000
python tests/smoke.py /tmp/synth /tmp/smoke_out
```

## Result artefacts

For every run, `results/runs/<dataset>_split<k>_seed<s>/` contains:

| File | Contents |
|---|---|
| `predictive.csv`, `predictive_family.csv` | Clean metrics at the fixed 0.5 and validation-tuned thresholds; per-family detection |
| `training.csv`, `training_history.csv` | Training time, tuned threshold, temperature, per-epoch history |
| `attacks.csv`, `attacks_family.csv` | Attack success, validity audit and perturbation size for every detector × attack × threat model × budget |
| `explanations_records.csv.gz` | Per-record faithfulness and stability metrics |
| `explanations_seed.csv` | Seed-level aggregates of the explanation metrics |
| `explanations_runtime.csv`, `repeats.csv` | Explainer runtime; repeated-call LIME and SHAP background stability |
| `arrays.npz` | Group attributions for every detector × explainer × view, adversarial inputs of the explained records, validation and test probabilities |
| `models/` | Trained MLP weights (`.pt`) and XGBoost models (`.json`) |

`results/data_audit/` contains, for each dataset:

- raw-file hashes, and the duplicate and label-conflict audit;
- preprocessing parameters;
- **partition manifests** (`partition_<dataset>_split<k>.csv.gz`: raw-file row number → train/validation/test);
- the fixed attack and explanation subsets.

Together with the raw CSVs, these files rebuild every model input exactly.

## Citation

If you use this code or the artefacts, please cite the paper (see `CITATION.cff`). Please also cite the ToN-IoT and Edge-IIoTset dataset papers.

## Licence

- **Code**: MIT licence. See [`LICENSE`](LICENSE).
- **Result artefacts** in `results/`, `analysis/` and `legacy_v1/`: Creative Commons Attribution 4.0 International. See [`LICENSE-DATA`](LICENSE-DATA).
- **Raw datasets**: not included; they remain under the terms of their original providers.

## Contact

S. M. Nihal Ahmed (corresponding author), Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh. Email: sm.nihal4@gmail.com
