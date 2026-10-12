"""Generate the self-contained Kaggle notebook from the package sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "config", "data", "models", "attacks", "explain", "runner", "extras", "analysis", "report", "cli"]

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [md("""
# Revision v2: multi-seed XAI / adversarial-evasion IDS experiment (Kaggle)

This notebook carries out the revised experiment for *"A Multi-Seed Evaluation of Explanation Faithfulness
and Stability under Adversarial Evasion in IoT/IIoT Intrusion Detection"* (major revision).

**Before running**
1. *Add Input* → the **ToN-IoT network** dataset (`train_test_network.csv`) and the **Edge-IIoTset** dataset
   (`ML-EdgeIIoT-dataset.csv`). These are the same inputs used for the original run.
2. *Settings* → Accelerator **GPU T4 x1** (or P100). Internet **On** (only needed to `pip install lime`).
3. In the **SESSION** cell, choose `SESSION = "TEST"` first (~15 min). Check that it finishes and that the printed
   macro-F1 values look reasonable. Then make one notebook version for each of `"A"`, `"B"`, `"C1"` and `"C2"`.
4. Use **Save Version → Save & Run All (Commit)** for A/B/C so that each one runs in the background (≤12 h).
   Download `results_session_<X>.zip` from each version's *Output* tab and send all four zips back.

| Session | Content | Expected GPU time |
|---|---|---|
| A | ToN-IoT: 20 seeds × {MLP, MLP+FGSM-AT, MLP+PGD-AT, XGBoost}, all attacks, SHAP/LIME, repeated-call | see log |
| B | Edge-IIoTset: same as A | see log |
| C1 | Partition sensitivity: 2 extra stratified partitions × 5 seeds, both datasets | see log |
| C2 | Adversarial-training trade-off sweep, baseline ablation, multiclass attack-family analysis, memory/latency profiling | see log |
| C3 | Attack success at false-positive-rate-matched operating points (short) | see log |
| C4 | Original-pipeline replication path (feature list, scaling) + absolute-memory profiling (short) | see log |

Each run writes `runs/<dataset>_split<k>_seed<s>/done.json`. If a session is interrupted, attach the previous
version's output as an input and set `RESUME_FROM` so that the completed runs are skipped.
"""),
code("""
import importlib.util, subprocess, sys
for pkg in ["lime"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
import torch, shap, xgboost
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| shap", shap.__version__, "| xgboost", xgboost.__version__)
""")]
for m in MODULES:
    src = (ROOT / "iotxai" / f"{m}.py").read_text()
    cells.append(code(f"%%writefile /kaggle/working/iotxai/{m}.py\n" + src) if m != "__init__" else
                 code("import os; os.makedirs('/kaggle/working/iotxai', exist_ok=True)\n" +
                      "open('/kaggle/working/iotxai/__init__.py','w').write('''" + src + "''')"))
cells.append(code('''
# ============================ SESSION =====================================
SESSION = "TEST"          # "TEST" first, then "A", "B", "C1", "C2", "C3", "C4" (one notebook version each)
RESUME_FROM = ""          # e.g. "/kaggle/input/<previous-version-output>/results_v2" to skip finished runs
TON_CSV = ""              # leave "" for automatic discovery under /kaggle/input
EDGE_CSV = ""
# ==========================================================================
import sys, shutil, time
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
OUT = "/kaggle/working/results_v2"
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, OUT, dirs_exist_ok=True)
    for d in Path(OUT, "runs").glob("*"):
        if not (d / "done.json").exists():
            shutil.rmtree(d)
from iotxai.cli import run_session
cfg = run_session(SESSION, OUT, "/kaggle/working/cache_v2", "/kaggle/input", TON_CSV, EDGE_CSV)
'''))
cells.append(code('''
# Quick look (also serves as a sanity check for the TEST session)
import pandas as pd
pd.set_option("display.width", 200)
pred = pd.concat([pd.read_csv(f) for f in Path(OUT).glob("runs/*/predictive.csv")])
display(pred[pred.threshold_rule=="val_tuned"].groupby(["dataset","model"])[["macro_f1","mal_recall","ben_precision","fpr","pr_auc"]].mean().round(4))
att = pd.concat([pd.read_csv(f) for f in Path(OUT).glob("runs/*/attacks.csv")])
display(att[att.eps==att.eps.max()].groupby(["dataset","model","attack","threat"])[["asr","syntactic_valid_rate"]].mean().round(3))
for f in sorted(Path(OUT).glob("data_audit/audit_*.json")):
    print(f.name, open(f).read()[:1500])
'''))
cells.append(md("""
## Print every result table (copy-paste fallback)
Every run already printed its tables when it finished (blocks starting with `### BEGIN`). This cell reprints
**all** of them. If the output files are ever lost, copy the whole output of this cell (or of the full log) into
a .txt file and send it. The results can be rebuilt from that text with `python -m iotxai.report`.
"""))
cells.append(code('''
from iotxai.report import print_all
print_all(OUT)
with open("/kaggle/working/printed_tables.txt", "w") as fh:
    print_all(OUT, fh)
print("### ALL TABLES PRINTED ###")
'''))
cells.append(code('''
import shutil
from IPython.display import FileLink, display
zip_base = f"/kaggle/working/results_session_{SESSION}"
shutil.make_archive(zip_base, "zip", "/kaggle/working", "results_v2")
print("size MB:", round(Path(zip_base + ".zip").stat().st_size / 2**20, 1))
display(FileLink(f"results_session_{SESSION}.zip"))
'''))
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks" / "kaggle_revision_v2.ipynb").write_text(json.dumps(nb, indent=1))
print("written", len(cells), "cells")
