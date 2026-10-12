"""Publication figures for the revised manuscript.

usage: python tools/make_figures.py <results_dir> <analysis_dir> <out_dir>
(results_dir = merged result bundle; analysis_dir = output of python -m iotxai.analysis)
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, AN, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "axes.axisbelow": True})
# Okabe-Ito (CVD-safe); every series also has its own marker (secondary encoding)
MODELS = ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"]
LAB = {"mlp_clean": "MLP", "mlp_fgsmat": "MLP+FGSM-AT", "mlp_pgdat": "MLP+PGD-AT", "xgb": "XGBoost"}
COL = {"mlp_clean": "#0072B2", "mlp_fgsmat": "#E69F00", "mlp_pgdat": "#CC79A7", "xgb": "#009E73"}
MK = {"mlp_clean": "o", "mlp_fgsmat": "s", "mlp_pgdat": "^", "xgb": "D"}
EXC = {"SHAP": "#0072B2", "LIME": "#D55E00"}
DS = {"ton": "ToN-IoT", "edge": "Edge-IIoTset"}
MM = 1 / 25.4


def load(name):
    return pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "runs" / "*" / f"{name}.csv"))], ignore_index=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


att = load("attacks")
att = att[att.split == 0]
main = ((att.model.str.startswith("mlp")) & (att.attack == "pgd")) | ((att.model == "xgb") & (att.attack == "bb"))
A = att[main]

# ---------------------------------------------------------------- Fig: ASR vs eps, 3 threat models
fig, axes = plt.subplots(2, 3, figsize=(174 * MM, 98 * MM), sharey=True, sharex=True)
for i, ds in enumerate(["ton", "edge"]):
    for j, th in enumerate(["unconstrained", "numeric", "constrained"]):
        ax = axes[i, j]
        for m in MODELS:
            g = A[(A.dataset == ds) & (A.model == m) & (A.threat == th) & (A.eps <= 0.05 + 1e-9)].groupby("eps")["asr"]
            mu, sd = g.mean(), g.std(ddof=1)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, color=COL[m], marker=MK[m], ms=3.5, capsize=1.5,
                        lw=1.1, label=LAB[m])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.01, 0.03, 0.05])
        if i == 0:
            ax.set_title({"unconstrained": "Unconstrained", "numeric": "Numeric-only",
                          "constrained": "Domain-constrained"}[th])
        if j == 0:
            ax.set_ylabel(f"{DS[ds]}\nattack success rate")
        if i == 1:
            ax.set_xlabel(r"budget $\epsilon$ (normalized $\ell_\infty$)")
axes[0, 0].legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig_asr")

# ---------------------------------------------------------------- Fig: padding budget (constrained)
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 56 * MM), sharey=True)
for ax, ds in zip(axes, ["ton", "edge"]):
    for m in MODELS:
        g = A[(A.dataset == ds) & (A.model == m) & (A.threat == "constrained")].groupby("eps")["asr"]
        mu, sd = g.mean(), g.std(ddof=1)
        ax.errorbar(mu.index, mu.values, yerr=sd.values, color=COL[m], marker=MK[m], ms=3.5, capsize=1.5, label=LAB[m])
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.03, 0.1, 0.25, 0.5, 1.0]); ax.set_xticklabels(["0.01", "0.03", "0.1", "0.25", "0.5", "1"])
    ax.set_title(DS[ds]); ax.set_xlabel(r"padding budget $\epsilon$ (constrained; 1 = training maximum)")
    ax.set_ylim(-0.03, 0.75)
axes[0].set_ylabel("attack success rate"); axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig_padding")

# ---------------------------------------------------------------- Fig: operating point
op = pd.read_csv(R / "extras" / "operating_point.csv")
fpr_rules = ["fpr_0.001", "fpr_0.005", "fpr_0.01", "fpr_0.02", "fpr_0.05"]
fpr_x = [0.001, 0.005, 0.01, 0.02, 0.05]
fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 96 * MM), sharex=True, sharey=True)
for i, ds in enumerate(["ton", "edge"]):
    for j, (th, eps, title) in enumerate([("unconstrained", 0.05, r"unconstrained, $\epsilon$=0.05"),
                                          ("constrained", 1.0, r"domain-constrained, maximum padding")]):
        ax = axes[i, j]
        for m in MODELS:
            g = op[(op.dataset == ds) & (op.model == m) & (op.threat == th) & (op.eps == eps)]
            mu = [g[g.threshold_rule == r].mal_recall_adv.mean() for r in fpr_rules]
            sd = [g[g.threshold_rule == r].mal_recall_adv.std(ddof=1) for r in fpr_rules]
            ax.errorbar(fpr_x, mu, yerr=sd, color=COL[m], marker=MK[m], ms=3.5, capsize=1.5, label=LAB[m])
        ax.set_xscale("log"); ax.set_xticks(fpr_x); ax.set_xticklabels(["0.1%", "0.5%", "1%", "2%", "5%"])
        ax.set_ylim(-0.03, 1.03)
        if i == 0:
            ax.set_title(title)
        if i == 1:
            ax.set_xlabel("false-positive-rate budget (matched on validation)")
        if j == 0:
            ax.set_ylabel(f"{DS[ds]}\ndetection under attack")
axes[1, 0].legend(frameon=False, loc="lower right")
fig.tight_layout()
save(fig, "fig_operating_point")

# ---------------------------------------------------------------- Fig: faithfulness by reference
sl = pd.read_csv(AN / "seed_level_explanations.csv")
s0 = sl[(sl.split == 0) & (sl.view == "clean")]
refs = [("median", "Med"), ("marginal", "Marg"), ("knn", "kNN")]
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 64 * MM), sharey=True)
for ax, ds in zip(axes, ["ton", "edge"]):
    xt, xl = [], []
    for mi, m in enumerate(MODELS):
        for ri, (r, rl) in enumerate(refs):
            x0 = mi * 4 + ri
            for ei, ex in enumerate(["SHAP", "LIME"]):
                v = s0[(s0.dataset == ds) & (s0.model == m) & (s0.explainer == ex)][f"dcomp_{r}"]
                ax.errorbar(x0 + (ei - 0.5) * 0.32, v.mean(), yerr=v.std(ddof=1), fmt="o" if ex == "SHAP" else "s",
                            ms=3.2, capsize=1.2, color=EXC[ex], label=ex if (mi == 0 and ri == 0) else None)
            xt.append(x0); xl.append(rl)
        ax.text(mi * 4 + 1, -0.21, LAB[m], ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
        if mi:
            ax.axvline(mi * 4 - 1, color="#dddddd", lw=0.6)
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6, rotation=45, ha="right")
    ax.set_title(DS[ds]); ax.grid(axis="x", visible=False); ax.set_ylim(-0.03, 0.47)
axes[0].set_ylabel(r"$\Delta$Comp (AOPC $-$ random control)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save(fig, "fig_faithfulness")

# ---------------------------------------------------------------- Fig: stability axes (Spearman)
rep = load("repeats")
cs = pd.read_csv(R / "cross_seed_summary.csv")
cs = cs[cs.split == 0]
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), sharey=True)
axis_defs = [("call / background", None), ("training seed", None), ("input: max. padding", "c"), ("input: numeric", "n"),
             ("input: unconstrained", "u")]
for ax, ds in zip(axes, ["ton", "edge"]):
    for mi, m in enumerate(MODELS):
        for ei, ex in enumerate(["SHAP", "LIME"]):
            vals = []
            r = rep[(rep.dataset == ds) & (rep.model == m) & (rep.explainer == ex)]
            vals.append(r.groupby("seed")["spearman"].mean())
            c = cs[(cs.dataset == ds) & (cs.model == m) & (cs.explainer == ex)]
            vals.append(pd.Series([c.spearman_mean.iloc[0]]) if len(c) else pd.Series(dtype=float))
            k = "bb" if m == "xgb" else "pgd"
            for t in ["c_1", "n_0.05", "u_0.05"]:
                v = sl[(sl.split == 0) & (sl.dataset == ds) & (sl.model == m) & (sl.explainer == ex) &
                       (sl.view == f"{k}_{t}")]["stab_spearman"]
                vals.append(v)
            for ai, v in enumerate(vals):
                x = ai * 9 + mi * 2 + ei * 0.7
                if len(v):
                    ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else None, fmt=MK[m], ms=3.2,
                                capsize=1, color=EXC[ex], mfc=EXC[ex] if ex == "SHAP" else "white", mew=0.9)
    ax.set_xticks([ai * 9 + 3.4 for ai in range(len(axis_defs))])
    ax.set_xticklabels([a for a, _ in axis_defs], rotation=20, ha="right", fontsize=6.5)
    ax.set_title(DS[ds]); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.04)
axes[0].set_ylabel(r"Spearman $\rho$ of |group attribution|")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=EXC["SHAP"], marker="o", ls="", label="SHAP (filled)"),
     Line2D([], [], color=EXC["LIME"], marker="o", mfc="white", ls="", label="LIME (open)")]
h += [Line2D([], [], color="#555555", marker=MK[m], ls="", label=LAB[m], ms=4) for m in MODELS]
axes[1].legend(handles=h, frameon=False, loc="lower left", fontsize=6, ncol=2)
fig.tight_layout()
save(fig, "fig_stability")

# ---------------------------------------------------------------- Fig: AT sweep trade-off
sw = pd.read_csv(R / "extras" / "at_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 58 * MM))
for ax, ds in zip(axes, ["ton", "edge"]):
    g = sw[(sw.dataset == ds) & (sw.eps == 0.05)]
    for th, mk, c in (("unconstrained", "o", "#0072B2"), ("numeric", "s", "#E69F00"), ("constrained", "^", "#009E73")):
        h = g[g.threat == th].groupby("model")[["clean_macro_f1", "asr"]].mean()
        ax.scatter(h["asr"], h["clean_macro_f1"], marker=mk, s=16, color=c, label=th, zorder=3)
        if th == "unconstrained":
            for name, r in h.iterrows():
                lab = "clean" if name == "mlp_clean" else name.replace("sweep_", "").replace("_", "/")
                ax.annotate(lab, (r["asr"], r["clean_macro_f1"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_title(DS[ds]); ax.set_xlabel(r"PGD attack success rate ($\epsilon$=0.05)")
axes[0].set_ylabel("clean macro-F1 (validation-tuned threshold)")
axes[0].legend(frameon=False, title=r"labels: $\epsilon_t/\lambda$", title_fontsize=6)
fig.tight_layout()
save(fig, "fig_at_tradeoff")
print("figures written to", OUT)
