"""Summaries, LaTeX table bodies and figures for the second-round analyses.

usage: python tools/round2_report.py <main_results_dir> <round2_results_dir> <out_dir>
  main_results_dir   : merged bundle of the primary runs (runs/, extras/)
  round2_results_dir : merged bundle of sessions R1-R4 (extras/r2_*.csv, runs/*_split{10,20,30}_*)
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAIN, R2, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
DS = {"ton": "ToN-IoT", "edge": "Edge-IIoTset"}
MODELS = ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"]
LAB = {"mlp_clean": "MLP", "mlp_fgsmat": "MLP+FGSM-AT", "mlp_pgdat": "MLP+PGD-AT", "xgb": "XGBoost"}
COL = {"mlp_clean": "#0072B2", "mlp_fgsmat": "#E69F00", "mlp_pgdat": "#CC79A7", "xgb": "#009E73"}
MK = {"mlp_clean": "o", "mlp_fgsmat": "s", "mlp_pgdat": "^", "xgb": "D"}
NUM = {}


def cat(pattern):
    fs = glob.glob(str(R2 / pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def ms(x, d=2):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "--"
    return f"{x.mean():.{d}f}" + (f"\\,$\\pm$\\,{x.std(ddof=1):.{d}f}" if len(x) > 1 else "")


def m(x, d=2):
    x = pd.Series(x).dropna()
    return "--" if len(x) == 0 else f"{x.mean():.{d}f}"


main_attack = lambda model: "bb300" if model == "xgb" else "pgd"

# ------------------------------------------------------------------ robustness (R1/R2)
op = cat("extras/r2_operating_*.csv")
mg = cat("extras/r2_margin_*.csv")
if len(op):
    op.to_csv(OUT / "r2_operating_all.csv", index=False)
    mg.to_csv(OUT / "r2_margin_all.csv", index=False)
    key = ["dataset", "model", "attack", "form", "threat", "eps", "threshold_rule"]
    S = op.groupby(key).agg(n_seeds=("seed", "nunique"), asr=("asr", "mean"), asr_sd=("asr", "std"),
                            det=("mal_recall_adv", "mean"), det_sd=("mal_recall_adv", "std"),
                            clean_det=("mal_recall_clean", "mean"), test_fpr=("test_fpr", "mean"),
                            asr_tmv=("asr_threat_model_valid", "mean")).reset_index()
    S.to_csv(OUT / "r2_operating_summary.csv", index=False)
    M = mg.groupby(["dataset", "model", "attack", "form", "threat", "eps"]).mean(numeric_only=True).reset_index()
    M.to_csv(OUT / "r2_margin_summary.csv", index=False)

    def g(ds, model, attack, form, threat, eps, rule, col):
        r = S[(S.dataset == ds) & (S.model == model) & (S.attack == attack) & (S.form == form) & (S.threat == threat)
              & np.isclose(S.eps, eps) & (S.threshold_rule == rule)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    # Table: attacks at the tuned threshold incl. projection and strongest attacks
    lines = []
    for ds in ["ton", "edge"]:
        for i, mdl in enumerate(MODELS):
            a = main_attack(mdl)
            cells = [DS[ds] if i == 0 else "", LAB[mdl]]
            for threat, eps in (("unconstrained", 0.05), ("numeric", 0.05)):
                cells.append(f"{g(ds, mdl, a, 'raw', threat, eps, 'val_tuned', 'asr'):.2f} / "
                             f"{g(ds, mdl, a, 'projected', threat, eps, 'val_tuned', 'asr'):.2f}")
            for eps in (0.05, 1.0):
                cells.append(f"{g(ds, mdl, a, 'raw', 'constrained', eps, 'val_tuned', 'asr'):.3f} / "
                             f"{g(ds, mdl, 'best', 'raw', 'constrained', eps, 'val_tuned', 'asr'):.3f}")
            lines.append(" & ".join(cells) + r"\\")
        if ds == "ton":
            lines.append(r"\midrule")
    (OUT / "tab_attacks_r2.tex").write_text("\n".join(lines) + "\n")

    # Table: matched-FPR detection under attack (strongest attack) + threshold-free AUC
    lines = []
    for ds in ["ton", "edge"]:
        for i, mdl in enumerate(MODELS):
            cells = [DS[ds] if i == 0 else "", LAB[mdl]]
            cells.append(f"{g(ds, mdl, 'best', 'raw', 'constrained', 1.0, 'fpr_0.01', 'clean_det'):.2f}")
            for (threat, eps, form) in (("unconstrained", 0.05, "raw"), ("unconstrained", 0.05, "projected"),
                                        ("constrained", 1.0, "raw")):
                cells.append(f"{g(ds, mdl, 'best', form, threat, eps, 'fpr_0.01', 'det'):.2f}")
            cells.append(f"{g(ds, mdl, 'best', 'raw', 'constrained', 1.0, 'val_tuned', 'det'):.2f}")
            r = M[(M.dataset == ds) & (M.model == mdl) & (M.attack == "best") & (M.form == "raw")]
            auc_c = r.auc_clean.mean() if len(r) else np.nan
            auc_p = r[(r.threat == "constrained") & np.isclose(r.eps, 1.0)].auc_adv.mean()
            cells.append(f"{auc_c:.3f} / {auc_p:.3f}")
            lines.append(" & ".join(cells) + r"\\")
        if ds == "ton":
            lines.append(r"\midrule")
    (OUT / "tab_operating_r2.tex").write_text("\n".join(lines) + "\n")

    # Numbers for the text
    for ds in ["ton", "edge"]:
        for mdl in MODELS:
            for rule in ["val_tuned", "fpr_0.01", "fpr_0.005", "fpr_0.05", "fixed_0.5"]:
                for threat, eps in (("unconstrained", 0.05), ("numeric", 0.05), ("constrained", 0.05), ("constrained", 1.0)):
                    for att in ["pgd", "bb300", "bb3000", "coord", "best"]:
                        for form in ["raw", "projected"]:
                            for col in ["asr", "det", "clean_det", "test_fpr"]:
                                v = g(ds, mdl, att, form, threat, eps, rule, col)
                                if not np.isnan(v):
                                    NUM[f"{ds}|{mdl}|{att}|{form}|{threat}|{eps:g}|{rule}|{col}"] = round(v, 4)

    # Figure: detection under attack vs FPR budget (strongest attack, 20 seeds)
    fpr_rules = ["fpr_0.001", "fpr_0.005", "fpr_0.01", "fpr_0.02", "fpr_0.05"]
    fpr_x = np.array([0.001, 0.005, 0.01, 0.02, 0.05])
    DODGE = {"mlp_clean": -1.5, "mlp_fgsmat": -0.5, "mlp_pgdat": 0.5, "xgb": 1.5}
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True})
    MM = 1 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 118 * MM), sharex=True, sharey=True)
    for i, ds in enumerate(["ton", "edge"]):
        for j, (threat, eps, title) in enumerate([("unconstrained", 0.05, r"Unconstrained, $\epsilon$=0.05"),
                                                  ("constrained", 1.0, "Constrained, maximum padding")]):
            ax = axes[i, j]
            for mdl in MODELS:
                o = op[(op.dataset == ds) & (op.model == mdl) & (op.attack == "best") & (op.form == "raw")
                       & (op.threat == threat) & np.isclose(op.eps, eps)]
                mu = [o[o.threshold_rule == r].mal_recall_adv.mean() for r in fpr_rules]
                sd = [o[o.threshold_rule == r].mal_recall_adv.std(ddof=1) for r in fpr_rules]
                ax.errorbar(fpr_x * (1.13 ** DODGE[mdl]), mu, yerr=sd, color=COL[mdl], marker=MK[mdl], ms=4,
                            capsize=1.5, elinewidth=0.7, label=LAB[mdl], zorder=3 + DODGE[mdl])
            ax.set_xscale("log"); ax.set_xticks(fpr_x); ax.set_xticklabels(["0.1%", "0.5%", "1%", "2%", "5%"])
            ax.minorticks_off(); ax.set_ylim(-0.04, 1.04); ax.set_xlim(0.00065, 0.075)
            if i == 0:
                ax.set_title(title)
            if i == 1:
                ax.set_xlabel("false-positive budget")
            if j == 0:
                ax.set_ylabel(f"{DS[ds]}\ndetection under attack")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.045))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_operating_point.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Bimodality: per-seed clean-MLP outcomes (ToN), joined with primary-run ASR
    pa = pd.concat([pd.read_csv(f) for f in glob.glob(str(MAIN / "runs" / "*_split0_*" / "attacks.csv"))])
    pa = pa[(pa.model == "mlp_clean") & (pa.attack == "pgd") & (pa.threat == "unconstrained") & np.isclose(pa.eps, 0.05)]
    rows = []
    for ds in ["ton", "edge"]:
        mm = mg[(mg.dataset == ds) & (mg.model == "mlp_clean") & (mg.form == "raw")]
        for seed, gg in mm.groupby("seed"):
            pr = pa[(pa.dataset == ds) & (pa.seed == seed)]
            r = {"dataset": ds, "seed": seed, "asr_tuned_primary": float(pr.asr.iloc[0]) if len(pr) else np.nan,
                 "threshold": gg.val_tuned_threshold.iloc[0], "median_clean_margin_tuned": gg.median_clean_margin_tuned.iloc[0],
                 "median_clean_margin_0p5": gg.median_clean_margin_0p5.iloc[0]}
            for threat, eps in (("unconstrained", 0.05), ("constrained", 1.0)):
                q = gg[(gg.attack == "best") & (gg.threat == threat) & np.isclose(gg.eps, eps)]
                if len(q):
                    r[f"logodds_drop_{threat[0]}{eps:g}"] = q.mean_logodds_drop.iloc[0]
                    r[f"auc_adv_{threat[0]}{eps:g}"] = q.auc_adv.iloc[0]
                o = op[(op.dataset == ds) & (op.model == "mlp_clean") & (op.seed == seed) & (op.attack == "best")
                       & (op.form == "raw") & (op.threat == threat) & np.isclose(op.eps, eps)]
                for rule in ["val_tuned", "fpr_0.01", "fixed_0.5"]:
                    oo = o[o.threshold_rule == rule]
                    if len(oo):
                        r[f"asr_{threat[0]}{eps:g}_{rule}"] = oo.asr.iloc[0]
                        r[f"det_{threat[0]}{eps:g}_{rule}"] = oo.mal_recall_adv.iloc[0]
            rows.append(r)
    B = pd.DataFrame(rows)
    B["vulnerable_group"] = B.asr_tuned_primary > 0.3
    B.to_csv(OUT / "r2_bimodality_by_seed.csv", index=False)
    if len(B):
        NUM["bimodality_group_means"] = B.groupby(["dataset", "vulnerable_group"]).mean(numeric_only=True).round(4).reset_index().to_dict("records")

# ------------------------------------------------------------------ explainer fairness (R3)
fa = cat("extras/r2_explainer_fairness_*.csv")
if len(fa):
    fa.to_csv(OUT / "r2_fairness_all.csv", index=False)
    F = fa.groupby(["dataset", "model", "explainer", "config"]).agg(
        n=("seed", "nunique"), dcomp_marginal=("dcomp_marginal", "mean"), dcomp_knn=("dcomp_knn", "mean"),
        dcomp_median=("dcomp_median", "mean"), rep_spearman=("rep_spearman", "mean"),
        ms_per_record=("ms_per_record", "mean")).reset_index()
    F.to_csv(OUT / "r2_fairness_summary.csv", index=False)
    lines = []
    for ds in ["ton", "edge"]:
        for i, mdl in enumerate(MODELS):
            f = F[(F.dataset == ds) & (F.model == mdl)]
            main = f[f.explainer.str.startswith("SHAP_")]
            ker = f[f.explainer == "KernelSHAP"]
            lime = f[f.explainer == "LIME"]
            ldef = lime[lime.config.str.startswith("samples=1000,kernel=0.75") & lime.config.str.endswith("discretize=0")]
            lbest = lime.loc[[lime.dcomp_knn.idxmax()]] if len(lime) else lime
            def c(r, col):
                return f"{r[col].iloc[0]:.3f}" if len(r) else "--"
            cells = [DS[ds] if i == 0 else "", LAB[mdl]]
            for r in (main, ker, ldef, lbest):
                cells += [c(r, "dcomp_marginal"), c(r, "dcomp_knn")]
            cells.append(lbest.config.iloc[0].replace("samples=", "").replace(",kernel=", ", ").replace("sqrt(d),discretize=1", "$\\sqrt{p}$, disc.").replace("sqrt(d),discretize=0", "$\\sqrt{p}$") if len(lbest) else "--")
            lines.append(" & ".join(cells) + r"\\")
        if ds == "ton":
            lines.append(r"\midrule")
    (OUT / "tab_fairness_r2.tex").write_text("\n".join(lines) + "\n")
    NUM["fairness"] = F.round(4).to_dict("records")

# ------------------------------------------------------------------ data variants (R4)
vr = sorted(glob.glob(str(R2 / "runs" / "*_split[123]0_*")))
if vr:
    P, A, X = [], [], []
    for d in vr:
        d = Path(d)
        for nm, lst in (("predictive.csv", P), ("attacks.csv", A), ("explanations_seed.csv", X)):
            if (d / nm).exists():
                lst.append(pd.read_csv(d / nm))
    P = pd.concat(P); A = pd.concat(A); X = pd.concat(X) if X else pd.DataFrame()
    # primary (split 0, first 5 seeds) for comparison
    seeds5 = sorted(P.seed.unique())
    P0 = pd.concat([pd.read_csv(f) for f in glob.glob(str(MAIN / "runs" / "*_split0_*" / "predictive.csv"))])
    P0 = P0[P0.seed.isin(seeds5)]
    A0 = pd.concat([pd.read_csv(f) for f in glob.glob(str(MAIN / "runs" / "*_split0_*" / "attacks.csv"))])
    A0 = A0[A0.seed.isin(seeds5)]
    X0 = pd.concat([pd.read_csv(f) for f in glob.glob(str(MAIN / "runs" / "*_split0_*" / "explanations_seed.csv"))])
    X0 = X0[X0.seed.isin(seeds5)]
    P, A = pd.concat([P0, P]), pd.concat([A0, A])
    X = pd.concat([X0, X]) if len(X) else X0
    VN = {0: "primary (stratified)", 10: "label conflicts retained", 20: "temporal, global", 30: "temporal, within family"}
    rows, lines = [], []
    for ds in ["ton", "edge"]:
        for k, (sid, vname) in enumerate(VN.items()):
            p = P[(P.dataset == ds) & (P.split == sid) & (P.threshold_rule == "val_tuned")]
            a = A[(A.dataset == ds) & (A.split == sid)]
            x = X[(X.dataset == ds) & (X.split == sid) & (X.view == "clean")] if len(X) else X
            if not len(p):
                continue
            r = {"dataset": ds, "variant": vname, "n_seeds": p.seed.nunique()}
            for mdl in ["mlp_clean", "xgb"]:
                pp = p[p.model == mdl]
                r[f"f1_{mdl}"] = pp.macro_f1.mean(); r[f"fpr_{mdl}"] = pp.fpr.mean(); r[f"auc_{mdl}"] = pp.roc_auc.mean()
            aa = a[(a.model == "mlp_clean") & (a.attack == "pgd")]
            r["asr_mlp_u"] = aa[(aa.threat == "unconstrained") & np.isclose(aa.eps, 0.05)].asr.mean()
            r["asr_mlp_c1"] = aa[(aa.threat == "constrained") & np.isclose(aa.eps, 1.0)].asr.mean()
            bb = a[(a.model == "xgb") & (a.attack == "bb")]
            r["asr_xgb_c1"] = bb[(bb.threat == "constrained") & np.isclose(bb.eps, 1.0)].asr.mean()
            if len(x):
                for ex in ["SHAP", "LIME"]:
                    r[f"dcomp_marg_{ex}"] = x[x.explainer == ex].dcomp_marginal.mean()
                    r[f"dcomp_knn_{ex}"] = x[x.explainer == ex].dcomp_knn.mean()
            rows.append(r)
            lines.append(" & ".join([DS[ds] if k == 0 else "", vname,
                                     f"{r['f1_mlp_clean']:.3f}", f"{r['f1_xgb']:.3f}",
                                     f"{r['fpr_mlp_clean']:.3f}", f"{r['fpr_xgb']:.3f}",
                                     f"{r['asr_mlp_u']:.2f}", f"{r['asr_mlp_c1']:.2f}", f"{r['asr_xgb_c1']:.3f}",
                                     f"{r.get('dcomp_marg_SHAP', np.nan):.2f} / {r.get('dcomp_marg_LIME', np.nan):.2f}"]) + r"\\")
        if ds == "ton":
            lines.append(r"\midrule")
    V = pd.DataFrame(rows); V.to_csv(OUT / "r2_variants_summary.csv", index=False)
    (OUT / "tab_variants_r2.tex").write_text("\n".join(lines) + "\n")
    NUM["variants"] = V.round(4).to_dict("records")

(OUT / "numbers_r2.json").write_text(json.dumps(NUM, indent=1, default=float))
print("round-2 outputs written to", OUT)
