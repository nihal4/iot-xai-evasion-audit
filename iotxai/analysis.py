"""Seed-level statistics, tables and figures for the revised manuscript.

Usage:  python -m iotxai.analysis <results_dir> [<results_dir> ...] --out <dir>

Inference unit = training seed (one aggregated value per seed and cell).
Primary outcomes (pre-declared in the revision):
  P1 attack success rate (ASR), PGD for MLPs / black-box for XGBoost, both threat models;
  P2 faithfulness = comprehensiveness AOPC minus the random-ranking control,
     marginal-sampling reference, evaluated class = predicted class;
  P3 stability = Spearman rank correlation of |group attribution| between the
     clean record and its attacked version (constrained threat model, eps = max).
Tests: two-sided exact Wilcoxon signed-rank on seed-paired differences, Holm
correction within each predeclared family, Cohen's d_z, percentile bootstrap CI
over seeds (10 000 resamples).
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODEL_LABEL = {"mlp_clean": "MLP", "mlp_fgsmat": "MLP+FGSM-AT", "mlp_pgdat": "MLP+PGD-AT", "xgb": "XGBoost"}
DS_LABEL = {"ton": "ToN-IoT", "edge": "Edge-IIoTset"}
EPS_MAIN = 0.05          # primary l_inf budget (pre-declared)
THREAT_ABBR = {"unconstrained": "U", "numeric": "N", "constrained": "C"}


# ------------------------------------------------------------------ loading
def load(dirs):
    frames = {}
    names = ["predictive", "predictive_family", "attacks", "attacks_family", "training",
             "explanations_runtime", "repeats", "training_history"]
    for nm in names + ["explanations_records", "explanations_seed"]:
        parts = []
        for d in dirs:
            for f in sorted(Path(d).glob(f"runs/*/{nm}.csv*")):
                parts.append(pd.read_csv(f))
        frames[nm] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    extras = {}
    for d in dirs:
        for f in Path(d).glob("extras/*.csv"):
            extras.setdefault(f.stem, []).append(pd.read_csv(f))
    frames["extras"] = {k: pd.concat(v, ignore_index=True) for k, v in extras.items()}
    audits = {}
    for d in dirs:
        for f in Path(d).glob("data_audit/*.json"):
            audits[f.stem] = json.loads(f.read_text())
    frames["audit"] = audits
    return frames


# ------------------------------------------------------------------ stats helpers
def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def paired(diff):
    d = np.asarray(diff, float); d = d[np.isfinite(d)]
    out = {"n": len(d), "mean_diff": d.mean() if len(d) else np.nan,
           "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan}
    out["dz"] = out["mean_diff"] / out["sd_diff"] if out["sd_diff"] and out["sd_diff"] > 0 else np.nan
    out["ci_low"], out["ci_high"] = boot_ci(d)
    out["n_positive"] = int((d > 0).sum())
    if len(d) >= 2 and np.any(d != 0):
        try:
            out["p_wilcoxon"] = stats.wilcoxon(d, method="exact" if len(d) <= 25 else "auto").pvalue
        except Exception:
            out["p_wilcoxon"] = stats.wilcoxon(d).pvalue
        out["p_ttest"] = stats.ttest_1samp(d, 0).pvalue
    else:
        out["p_wilcoxon"] = out["p_ttest"] = np.nan
    return out


def holm(p):
    p = np.asarray(p, float); m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full_like(p, np.nan); running = 0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, min(1, (m - rank) * p[i]))
        adj[i] = running
    return adj


def mde_dz(n, alpha=0.05, power=0.8):
    """Minimum detectable paired effect (d_z) for a two-sided one-sample t-test."""
    from statsmodels.stats.power import TTestPower
    return TTestPower().solve_power(nobs=n, alpha=alpha, power=power, alternative="two-sided")


def msd(x, digits=4):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{digits}f} $\\pm$ {np.nanstd(x, ddof=1):.{digits}f}"


# ------------------------------------------------------------------ seed-level tables
def seed_level_explanations(rec):
    rec = rec.copy()
    for r in ["median", "marginal", "knn"]:
        rec[f"dcomp_{r}"] = rec[f"comp_{r}"] - rec[f"comp_rand_{r}"]
        rec[f"dsuff_{r}"] = rec[f"suff_{r}"] - rec[f"suff_rand_{r}"]
    rec["neg_v1"] = (rec["comp_v1style"] < 0).astype(float)
    rec["neg_pred_median"] = (rec["comp_median"] < 0).astype(float)
    rec["neg_pred_marginal"] = (rec["comp_marginal"] < 0).astype(float)
    keys = ["dataset", "split", "seed", "model", "explainer", "view"]
    vals = [c for c in rec.columns if c.startswith(("comp", "suff", "dcomp", "dsuff", "stab_", "neg_"))]
    g = rec.groupby(keys)[vals].mean().reset_index()
    ev = rec[rec["view"] != "clean"].copy()
    if len(ev):
        sub = ev[ev["evaded"] == 1].groupby(keys)[[c for c in vals if c.startswith("stab_")]].mean()
        sub.columns = [c + "_evaded" for c in sub.columns]
        g = g.merge(sub.reset_index(), on=keys, how="left")
        g = g.merge(ev.groupby(keys)["evaded"].mean().rename("evaded_rate").reset_index(), on=keys, how="left")
    return g


def cross_seed_stability(dirs, ks=(3, 5)):
    from .explain import stability
    rows = []
    for d in dirs:
        runs = sorted(Path(d).glob("runs/*/arrays.npz"))
        by = {}
        for f in runs:
            ds, sp, sd = f.parent.name.split("_")
            by.setdefault((ds, sp), []).append((int(sd.replace("seed", "")), f))
        for (ds, sp), lst in by.items():
            arrs = {s: np.load(f) for s, f in lst}
            keys = [k for k in next(iter(arrs.values())).files if k.startswith("attr__") and k.endswith("__clean")]
            for k in keys:
                _, model, ex, _ = k.split("__")
                mats = {s: a[k].astype(np.float32) for s, a in arrs.items() if k in a.files}
                for s1, s2 in combinations(sorted(mats), 2):
                    st = stability(mats[s1], mats[s2], list(ks))
                    rows.append({"dataset": ds, "split": int(sp.replace("split", "")), "model": model,
                                 "explainer": ex, "seed_a": s1, "seed_b": s2,
                                 **{kk: float(np.nanmean(v)) for kk, v in st.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main analysis
def analyse(dirs, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    F = load(dirs)
    res = {}
    pred = F["predictive"]; att = F["attacks"]; rec = F["explanations_records"]
    main_split = 0
    # ---- predictive
    p = pred[(pred.split == main_split)]
    cols = ["macro_f1", "balanced_accuracy", "mal_recall", "mal_precision", "ben_precision", "ben_recall",
            "fpr", "fnr", "pr_auc", "roc_auc", "brier", "ece", "threshold"]
    summ = p.groupby(["dataset", "model", "threshold_rule"])[cols].agg(["mean", "std", "count"])
    summ.to_csv(out / "T_predictive_summary.csv")
    res["predictive"] = summ
    # ---- attacks (seed level already)
    a = att[att.split == main_split]
    asumm = a.groupby(["dataset", "model", "attack", "threat", "eps"])[
        ["asr", "asr_at_0p5", "mal_recall_clean", "mal_recall_adv", "mean_linf", "mean_features_changed",
         "syntactic_valid_rate", "threat_model_valid_rate", "onehot_valid_rate", "integer_valid_rate",
         "relation_valid_rate", "asr_syntactic_valid", "asr_threat_model_valid", "mean_queries", "seconds",
         "eligible"]].agg(["mean", "std", "count"])
    asumm.to_csv(out / "T_attacks_summary.csv")
    # ASR CIs
    ci_rows = []
    for k, g in a.groupby(["dataset", "model", "attack", "threat", "eps"]):
        lo, hi = boot_ci(g["asr"])
        ci_rows.append(dict(zip(["dataset", "model", "attack", "threat", "eps"], k), asr_mean=g["asr"].mean(),
                            asr_sd=g["asr"].std(ddof=1), ci_low=lo, ci_high=hi, n=len(g)))
    pd.DataFrame(ci_rows).to_csv(out / "T_attacks_ci.csv", index=False)
    # ---- explanations
    have_expl = len(rec) or len(F["explanations_seed"])
    if have_expl:
        # record-level files when available, otherwise the printed seed-level summaries
        sl = seed_level_explanations(rec) if len(rec) else F["explanations_seed"]
        if len(rec) and len(F["explanations_seed"]):   # runs recovered from a printed log
            have = set(map(tuple, sl[["dataset", "split", "seed"]].values))
            extra = F["explanations_seed"][[tuple(r) not in have for r in F["explanations_seed"][["dataset", "split", "seed"]].values]]
            sl = pd.concat([sl, extra], ignore_index=True)
        sl.to_csv(out / "seed_level_explanations.csv", index=False)
        esumm = sl[sl.split == main_split].groupby(["dataset", "model", "explainer", "view"]).agg(["mean", "std"])
        esumm.to_csv(out / "T_explanations_summary.csv")
        # share of records with negative comprehensiveness: v1-style vs predicted-class definitions
        sl.groupby(["dataset", "model", "explainer", "view"])[["neg_v1", "neg_pred_median", "neg_pred_marginal"]].mean().to_csv(
            out / "T_negative_comprehensiveness_share.csv")
        # ---- primary tests: SHAP vs LIME (paired over seeds)
        tests = []
        emax = EPS_MAIN
        prim_views = ["clean", f"pgd_c_{emax:g}", f"bb_c_{emax:g}", f"pgd_n_{emax:g}", f"bb_n_{emax:g}"]
        outcomes = {"P2_dcomp_marginal": "dcomp_marginal", "dcomp_knn": "dcomp_knn", "dcomp_median": "dcomp_median",
                    "dsuff_marginal": "dsuff_marginal", "P3_stab_spearman": "stab_spearman",
                    "stab_jaccard@3": "stab_jaccard@3", "stab_signed_cosine": "stab_signed_cosine"}
        s0 = sl[sl.split == main_split]
        for (ds, model), g in s0.groupby(["dataset", "model"]):
            for view in g.view.unique():
                if view not in prim_views and not view.startswith(("pgd_", "bb_")):
                    continue
                for oname, col in outcomes.items():
                    if col not in g or g[col].isna().all():
                        continue
                    w = g[g.view == view].pivot_table(index="seed", columns="explainer", values=col)
                    if not {"SHAP", "LIME"} <= set(w.columns):
                        continue
                    t = paired(w["SHAP"] - w["LIME"])
                    tests.append(dict(family=f"{ds}|{model}|{oname}", dataset=ds, model=model, view=view,
                                      outcome=oname, comparison="SHAP-LIME", shap_mean=w["SHAP"].mean(),
                                      lime_mean=w["LIME"].mean(), **t))
                # explanation vs random control (is each explainer better than random?)
                for ex in ["SHAP", "LIME"]:
                    gg = g[(g.view == view) & (g.explainer == ex)]
                    if len(gg):
                        t = paired(gg["dcomp_marginal"])
                        tests.append(dict(family=f"{ds}|{model}|vs_random", dataset=ds, model=model, view=view,
                                          outcome="P2_dcomp_marginal", comparison=f"{ex}-random", **t))
        T = pd.DataFrame(tests)
        if len(T):
            T["p_holm"] = np.nan
            for fam, idx in T.groupby("family").groups.items():
                T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_wilcoxon"].values)
            T.to_csv(out / "T_tests_explainers.csv", index=False)
            res["tests_explainers"] = T
    # ---- condition contrasts (paired over seeds)
    ctr = []
    pv = p[p.threshold_rule == "val_tuned"]
    for ds, g in pv.groupby("dataset"):
        w = g.pivot_table(index="seed", columns="model", values="macro_f1")
        for m in [c for c in w.columns if c != "mlp_clean"]:
            ctr.append(dict(family=f"{ds}|clean_macro_f1", dataset=ds, outcome="clean_macro_f1",
                            comparison=f"{m}-mlp_clean", **paired(w[m] - w["mlp_clean"])))
    emax = EPS_MAIN
    for (ds, threat), g in a[(a.eps == emax)].groupby(["dataset", "threat"]):
        mlp = g[g.attack == "pgd"].pivot_table(index="seed", columns="model", values="asr")
        for m in ["mlp_fgsmat", "mlp_pgdat"]:
            if m in mlp:
                ctr.append(dict(family=f"{ds}|{threat}|ASR_PGD", dataset=ds, outcome=f"ASR_PGD_{threat}",
                                comparison=f"{m}-mlp_clean", **paired(mlp[m] - mlp["mlp_clean"])))
        bb = g[g.attack == "bb"].pivot_table(index="seed", columns="model", values="asr")
        for m in [c for c in bb.columns if c != "mlp_clean"]:
            ctr.append(dict(family=f"{ds}|{threat}|ASR_BB", dataset=ds, outcome=f"ASR_BB_{threat}",
                            comparison=f"{m}-mlp_clean", **paired(bb[m] - bb["mlp_clean"])))
        for mdl in mlp.columns:   # constrained vs unconstrained effect is computed below
            pass
    # threat-model effect
    for (ds, model, attack), g in a[(a.eps == emax) & a.attack.isin(["pgd", "bb"])].groupby(["dataset", "model", "attack"]):
        w = g.pivot_table(index="seed", columns="threat", values="asr")
        for lo, hi in (("constrained", "unconstrained"), ("constrained", "numeric"), ("numeric", "unconstrained")):
            if {lo, hi} <= set(w.columns):
                ctr.append(dict(family=f"{ds}|threat_effect", dataset=ds, outcome=f"ASR_{attack}_{model}",
                                comparison=f"{lo}-{hi}", **paired(w[lo] - w[hi])))
    C = pd.DataFrame(ctr)
    if len(C):
        C["p_holm"] = np.nan
        for fam, idx in C.groupby("family").groups.items():
            C.loc[idx, "p_holm"] = holm(C.loc[idx, "p_wilcoxon"].values)
        C.to_csv(out / "T_tests_conditions.csv", index=False)
    # ---- repeats / cross-seed
    if len(F["repeats"]):
        F["repeats"].groupby(["dataset", "model", "explainer", "source"]).mean(numeric_only=True).to_csv(
            out / "T_repeated_call_stability.csv")
    cs = cross_seed_stability(dirs)
    if not len(cs):   # fall back to printed in-session summaries
        parts = [pd.read_csv(f) for d in dirs for f in Path(d).glob("cross_seed_summary.csv")]
        if parts:
            pd.concat(parts).to_csv(out / "T_cross_seed_stability.csv", index=False)
    if len(cs):
        cs.to_csv(out / "cross_seed_pairs.csv", index=False)
        cs.groupby(["dataset", "split", "model", "explainer"]).mean(numeric_only=True).to_csv(out / "T_cross_seed_stability.csv")
    # ---- split sensitivity (variance components across split x seed)
    if pred.split.nunique() > 1:
        rows = []
        common = sorted(set.intersection(*[set(g.seed) for _, g in pred.groupby("split")]))
        pp = pred[(pred.threshold_rule == "val_tuned") & pred.seed.isin(common)]
        for (ds, m), g in pp.groupby(["dataset", "model"]):
            rows.append(dict(dataset=ds, model=m, outcome="macro_f1", **variance_components(g, "macro_f1")))
        aa = att[(att.eps == emax) & att.seed.isin(common) & att.attack.isin(["pgd", "bb"])]
        for (ds, m, ak, th), g in aa.groupby(["dataset", "model", "attack", "threat"]):
            rows.append(dict(dataset=ds, model=m, outcome=f"asr_{ak}_{th}", **variance_components(g, "asr")))
        if have_expl:
            ss = sl[sl.seed.isin(common)]
            for (ds, m, ex), g in ss[ss.view == "clean"].groupby(["dataset", "model", "explainer"]):
                rows.append(dict(dataset=ds, model=m, outcome=f"dcomp_marginal_{ex}", **variance_components(g, "dcomp_marginal")))
        pd.DataFrame(rows).to_csv(out / "T_split_sensitivity.csv", index=False)
    # ---- power
    n = pred[pred.split == main_split].seed.nunique()
    pw = {"n_seeds": int(n), "mde_dz_alpha05": mde_dz(n, 0.05) if n > 2 else None,
          "mde_dz_alpha05_over_4": mde_dz(n, 0.0125) if n > 2 else None,
          "min_exact_wilcoxon_p_two_sided": 2 / 2 ** n if n else None}
    (out / "power.json").write_text(json.dumps(pw, indent=2))
    # ---- runtime
    if len(F["explanations_runtime"]):
        F["explanations_runtime"].groupby(["dataset", "model", "explainer"])[["ms_per_record"]].agg(
            ["mean", "std"]).to_csv(out / "T_runtime.csv")
    for k, v in F["extras"].items():
        v.to_csv(out / f"extras_{k}.csv", index=False)
    (out / "audit_all.json").write_text(json.dumps(F["audit"], indent=2, default=str))
    figures(F, out, sl if have_expl else None, cs)
    return res


def variance_components(g, col):
    """One-way decomposition: between-split vs within-split (seed) variance of means."""
    by = g.groupby("split")[col]
    means, k = by.mean(), by.size().mean()
    within = by.var(ddof=1).mean()
    between = max(0.0, means.var(ddof=1) - within / k) if len(means) > 1 else np.nan
    return {"grand_mean": g[col].mean(), "split_means": "; ".join(f"{s}:{v:.4f}" for s, v in means.items()),
            "var_between_split": between, "var_within_split_seed": within,
            "share_split": between / (between + within) if (between + within) > 0 else np.nan}


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.1, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


COL = {"mlp_clean": "#0072B2", "mlp_fgsmat": "#E69F00", "mlp_pgdat": "#CC79A7", "xgb": "#009E73",
       "SHAP": "#0072B2", "LIME": "#D55E00"}


def figures(F, out, sl, cs):
    plt = _style()
    att = F["attacks"]; att = att[att.split == 0]
    if len(att):
        fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 62 / 25.4), sharey=True)
        for ax, ds in zip(axes, ["ton", "edge"]):
            g = att[att.dataset == ds]
            for m in ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"]:
                kind = "bb" if m == "xgb" else "pgd"
                for th, ls in (("unconstrained", "-"), ("numeric", ":"), ("constrained", "--")):
                    h = g[(g.model == m) & (g.attack == kind) & (g.threat == th) & (g.eps <= EPS_MAIN + 1e-9)].groupby("eps")["asr"]
                    if not len(h.mean()):
                        continue
                    mu, sd = h.mean(), h.std(ddof=1)
                    ax.errorbar(mu.index, mu.values, yerr=sd.values, ls=ls, marker="o", ms=2.5, capsize=1.5,
                                color=COL[m], label=f"{MODEL_LABEL[m]} ({THREAT_ABBR[th]})")
            ax.set_title(DS_LABEL.get(ds, ds)); ax.set_xlabel(r"$\epsilon$ (normalized $\ell_\infty$)")
            ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("Attack success rate")
        axes[1].legend(ncol=2, frameon=False, loc="upper left", fontsize=5.5)
        fig.tight_layout(); fig.savefig(out / "fig_asr.pdf", bbox_inches="tight"); plt.close(fig)
        # padding budget curve (constrained threat model)
        fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 58 / 25.4), sharey=True)
        for ax, ds in zip(axes, ["ton", "edge"]):
            g = att[(att.dataset == ds) & (att.threat == "constrained")]
            for m in ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"]:
                kind = "bb" if m == "xgb" else "pgd"
                h = g[(g.model == m) & (g.attack == kind)].groupby("eps")["asr"]
                if len(h.mean()):
                    ax.errorbar(h.mean().index, h.mean().values, yerr=h.std(ddof=1).values, marker="o", ms=2.5,
                                capsize=1.5, color=COL[m], label=MODEL_LABEL[m])
            ax.set_xscale("log"); ax.set_title(DS_LABEL.get(ds, ds))
            ax.set_xlabel(r"padding budget $\epsilon$ (constrained, log scale)")
        axes[0].set_ylabel("Attack success rate"); axes[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig_asr_padding.pdf", bbox_inches="tight"); plt.close(fig)
    if sl is not None and len(sl):
        s0 = sl[(sl.split == 0) & (sl.view == "clean")]
        fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 64 / 25.4), sharey=False)
        refs = ["median", "marginal", "knn"]
        for ax, ds in zip(axes, ["ton", "edge"]):
            g = s0[s0.dataset == ds]
            xt, xl = [], []
            x = 0
            for m in ["mlp_clean", "mlp_pgdat", "xgb"]:
                for ri, r in enumerate(refs):
                    for ei, ex in enumerate(["SHAP", "LIME"]):
                        v = g[(g.model == m) & (g.explainer == ex)][f"dcomp_{r}"]
                        if len(v):
                            ax.errorbar(x + ei * 0.3, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=2.5, capsize=1.5,
                                        color=COL[ex], label=ex if (x == 0 and ri == 0) else None)
                    xt.append(x + 0.15); xl.append(f"{MODEL_LABEL[m]}\n{r}")
                    x += 1
                x += 0.5
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=90, fontsize=5.5)
            ax.set_title(DS_LABEL.get(ds, ds))
        axes[0].set_ylabel("Comprehensiveness AOPC $-$ random control")
        axes[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig_faithfulness.pdf", bbox_inches="tight"); plt.close(fig)
        # stability vs eps
        sv = sl[(sl.split == 0) & sl.view.str.startswith("pgd_c")].copy()
        sv = sv[sv.view.str.split("_").str[-1].astype(float) <= EPS_MAIN + 1e-9]
        if len(sv):
            sv["eps"] = sv.view.str.split("_").str[-1].astype(float)
            rep = F["repeats"]
            fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 60 / 25.4), sharey=True)
            for ax, ds in zip(axes, ["ton", "edge"]):
                g = sv[sv.dataset == ds]
                for m, ls in (("mlp_clean", "-"), ("mlp_pgdat", "--")):
                    for ex in ["SHAP", "LIME"]:
                        h = g[(g.model == m) & (g.explainer == ex)].groupby("eps")["stab_spearman"]
                        if len(h.mean()):
                            ax.errorbar(h.mean().index, h.mean().values, yerr=h.std(ddof=1).values, ls=ls, marker="o",
                                        ms=2.5, capsize=1.5, color=COL[ex], label=f"{ex}, {MODEL_LABEL[m]}")
                if len(rep):
                    r = rep[(rep.dataset == ds) & (rep.model == "mlp_clean")]
                    for ex in ["SHAP", "LIME"]:
                        v = r[r.explainer == ex]["spearman"]
                        if len(v):
                            ax.axhline(v.mean(), color=COL[ex], lw=0.6, ls=":")
                ax.set_title(DS_LABEL.get(ds, ds)); ax.set_xlabel(r"$\epsilon$ (constrained PGD)")
            axes[0].set_ylabel(r"Spearman $\rho$ (clean vs. attacked)")
            axes[1].legend(frameon=False, fontsize=6)
            fig.tight_layout(); fig.savefig(out / "fig_stability.pdf", bbox_inches="tight"); plt.close(fig)
    sw = F["extras"].get("at_sweep")
    if sw is not None and len(sw):
        fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 60 / 25.4))
        emax = sw.eps.max()
        for ax, ds in zip(axes, ["ton", "edge"]):
            g = sw[(sw.dataset == ds) & (sw.eps == emax)]
            for th, mk in (("unconstrained", "o"), ("constrained", "s")):
                h = g[g.threat == th].groupby("model")[["clean_macro_f1", "asr"]].mean()
                ax.scatter(h["asr"], h["clean_macro_f1"], marker=mk, s=14, label=th,
                           color="#0072B2" if th[0] == "u" else "#D55E00")
                for name, r in h.iterrows():
                    lab = "clean" if name == "mlp_clean" else name.replace("sweep_", r"$\epsilon_t$=").replace("_", r", $\lambda$=")
                    ax.annotate(lab, (r["asr"], r["clean_macro_f1"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
            ax.set_title(DS_LABEL.get(ds, ds)); ax.set_xlabel(f"PGD ASR at $\\epsilon$={emax:g}")
            ax.set_ylabel("Clean macro-F1 (val-tuned threshold)")
        axes[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig_at_tradeoff.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    analyse(a.dirs, a.out)
    print("analysis written to", a.out)
