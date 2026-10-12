"""Print result tables into the notebook log as delimited CSV blocks, and rebuild
result folders from a pasted log.  This is a fallback for sessions whose output
files are lost: everything the analysis needs at the seed level is printed.

Block format:
    ### BEGIN <run-id> <table>
    <csv>
    ### END <run-id> <table>
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd

FLOAT_FMT = "%.6g"
ATTACK_COLS = ["dataset", "split", "seed", "model", "attack", "threat", "eps", "eligible", "success", "asr",
               "asr_at_0p5", "mal_recall_clean", "mal_recall_adv", "mean_linf", "mean_features_changed",
               "syntactic_valid_rate", "threat_model_valid_rate", "onehot_valid_rate", "integer_valid_rate",
               "relation_valid_rate", "asr_syntactic_valid", "asr_threat_model_valid", "mean_queries", "seconds",
               "n_attack", "mean_p_drop"]


def print_block(run_id, name, df, out=sys.stdout):
    if df is None or len(df) == 0:
        return
    out.write(f"### BEGIN {run_id} {name}\n")
    df.to_csv(out, index=False, float_format=FLOAT_FMT)
    out.write(f"### END {run_id} {name}\n")
    out.flush()


def run_tables(rdir: Path):
    """Seed-level tables of one finished run (compact)."""
    rdir = Path(rdir)
    t = {}
    for name in ["training", "predictive", "explanations_seed", "explanations_runtime", "repeats"]:
        f = rdir / f"{name}.csv"
        if f.exists():
            t[name] = pd.read_csv(f)
    if (rdir / "repeats.csv").exists():
        r = t["repeats"]
        keys = [c for c in ["dataset", "split", "seed", "model", "explainer", "source"] if c in r]
        t["repeats"] = r.groupby(keys).mean(numeric_only=True).reset_index().drop(columns=["rep_a", "rep_b"], errors="ignore")
    f = rdir / "attacks.csv"
    if f.exists():
        a = pd.read_csv(f)
        t["attacks"] = a[[c for c in ATTACK_COLS if c in a.columns]]
    f = rdir / "attacks_family.csv"
    if f.exists():   # per-family results for the primary attack of each model
        af = pd.read_csv(f)
        main = ((af.attack == "pgd") & af.model.str.startswith("mlp")) | ((af.attack == "bb") & (af.model == "xgb"))
        keep = main & (af.eps.isin([0.05, 1.0]))
        t["attacks_family"] = af[keep]
    f = rdir / "predictive_family.csv"
    if f.exists():
        pf = pd.read_csv(f)
        t["predictive_family"] = pf[pf.threshold_rule == "val_tuned"]
    return t


def print_run(rdir: Path, out=sys.stdout):
    rid = Path(rdir).name
    for name, df in run_tables(rdir).items():
        print_block(rid, name, df, out)


def print_all(results_dir, out=sys.stdout):
    """Reprint every finished run, the data audit and the supplementary experiments."""
    results_dir = Path(results_dir)
    for f in sorted((results_dir / "data_audit").glob("*.json")) if (results_dir / "data_audit").exists() else []:
        if f.name.startswith(("audit_", "split_")):
            out.write(f"### BEGIN data_audit {f.name}\n{f.read_text()}\n### END data_audit {f.name}\n")
    for rdir in sorted((results_dir / "runs").glob("*")):
        if (rdir / "done.json").exists():
            print_run(rdir, out)
    for f in sorted((results_dir / "extras").glob("*.csv")) if (results_dir / "extras").exists() else []:
        print_block("extras", f.stem, pd.read_csv(f), out)
    cs = results_dir / "cross_seed_summary.csv"
    if cs.exists():
        print_block("session", "cross_seed_summary", pd.read_csv(cs), out)


def cross_seed_summary(results_dir):
    """Cross-seed attribution agreement (needs arrays.npz; computed in-session and printed)."""
    from .analysis import cross_seed_stability
    cs = cross_seed_stability([results_dir])
    if len(cs) == 0:
        return cs
    s = cs.groupby(["dataset", "split", "model", "explainer"]).agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.drop(columns=[c for c in s.columns if c.startswith(("seed_a", "seed_b"))]).reset_index()
    s.to_csv(Path(results_dir) / "cross_seed_summary.csv", index=False)
    return s


# ------------------------------------------------------------------ parsing a pasted log
BEGIN = re.compile(r"^### BEGIN (\S+) (\S+)\s*$")


def parse_log(text, out_dir):
    """Rebuild results folders from a printed log. Duplicate blocks (per-run print and
    the final reprint) are written once; later copies overwrite earlier ones."""
    out_dir = Path(out_dir)
    lines = text.splitlines()
    i, n_blocks = 0, 0
    while i < len(lines):
        m = BEGIN.match(lines[i].strip())
        if not m:
            i += 1
            continue
        rid, name = m.group(1), m.group(2)
        end = f"### END {rid} {name}"
        j = i + 1
        body = []
        while j < len(lines) and lines[j].strip() != end:
            body.append(lines[j]); j += 1
        if j >= len(lines):
            print(f"warning: block {rid} {name} not terminated (truncated paste?)")
        if rid == "data_audit":
            d = out_dir / "data_audit"; d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text("\n".join(body))
        elif rid == "extras":
            d = out_dir / "extras"; d.mkdir(parents=True, exist_ok=True)
            pd.read_csv(io.StringIO("\n".join(body))).to_csv(d / f"{name}.csv", index=False)
        elif rid == "session":
            pd.read_csv(io.StringIO("\n".join(body))).to_csv(out_dir / f"{name}.csv", index=False)
        else:
            d = out_dir / "runs" / rid; d.mkdir(parents=True, exist_ok=True)
            pd.read_csv(io.StringIO("\n".join(body))).to_csv(d / f"{name}.csv", index=False)
            (d / "done.json").write_text("{}")
        n_blocks += 1
        i = j + 1
    return n_blocks


if __name__ == "__main__":
    # python -m iotxai.report <pasted_log.txt> <out_dir>
    n = parse_log(Path(sys.argv[1]).read_text(errors="replace"), sys.argv[2])
    print("blocks parsed:", n)
