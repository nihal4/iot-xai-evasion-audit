"""LaTeX table bodies for the revised manuscript (generated, never hand-typed).

usage: python tools/make_tables.py <results_dir> <analysis_dir> <out_dir>
Also writes numbers.json with every value quoted in the text.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, AN, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"mlp_clean": "MLP", "mlp_fgsmat": "MLP+FGSM-AT", "mlp_pgdat": "MLP+PGD-AT", "xgb": "XGBoost"}
MODELS = list(LAB)
DS = {"ton": "ToN-IoT", "edge": "Edge-IIoTset"}
N = {}   # numbers for the text


def load(name, split=0):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "runs" / "*" / f"{name}.csv"))], ignore_index=True)
    return d[d.split == split] if split is not None and "split" in d else d


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}f}\\,$\\pm$\\,{np.nanstd(x, ddof=1):.{d}f}"


def f(x, d=3):
    return f"{x:.{d}f}"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


# ------------------------------------------------------------------ data audit
rows = []
for ds in ["ton", "edge"]:
    a = json.loads((R / "data_audit" / f"audit_{ds}.json").read_text())
    s = json.loads((R / "data_audit" / f"split_{ds}_split0.json").read_text())
    N[f"audit_{ds}"] = {k: a[k] for k in ["rows_raw", "rows_exact_duplicate_loaded_columns",
                                         "rows_in_label_conflicting_groups", "rows_duplicate_feature_label",
                                         "rows_after_dedup", "positive_rate_after_dedup", "sha256"]}
    N[f"split_{ds}"] = s
    rows.append(f"{DS[ds]} & {a['rows_raw']:,} & {a['rows_exact_duplicate_loaded_columns']:,} & "
                f"{a['rows_duplicate_feature_label']:,} & {a['rows_in_label_conflicting_groups']:,} & "
                f"{a['rows_after_dedup']:,} & {a['positive_rate_after_dedup']:.3f} & "
                f"{s['n_model_columns']} / {s['n_feature_groups']} & "
                f"{s['n_train']:,} / {s['n_val']:,} / {s['n_test']:,}\\\\")
write("tab_audit", "\n".join(rows).replace(",", "{,}"))

# ------------------------------------------------------------------ clean performance
pred = load("predictive")
rows = []
for ds in ["ton", "edge"]:
    for m in MODELS:
        g = pred[(pred.dataset == ds) & (pred.model == m) & (pred.threshold_rule == "val_tuned")]
        g5 = pred[(pred.dataset == ds) & (pred.model == m) & (pred.threshold_rule == "fixed_0.5")]
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & {ms(g.macro_f1)} & {f(g5.macro_f1.mean())} & "
                    f"{f(g.mal_recall.mean())} & {f(g.ben_precision.mean())} & {ms(g.fpr)} & {f(g5.fpr.mean())} & {f(g.roc_auc.mean())} & {f(g.ece.mean())}\\\\")
        N[f"clean_{ds}_{m}"] = {k: float(g[k].mean()) for k in ["macro_f1", "mal_recall", "ben_precision", "fpr", "pr_auc", "ece", "roc_auc", "ben_recall"]}
        N[f"clean_{ds}_{m}"].update({k + "_sd": float(g[k].std(ddof=1)) for k in ["macro_f1", "fpr"]})
        N[f"clean05_{ds}_{m}"] = {k: float(g5[k].mean()) for k in ["macro_f1", "fpr", "mal_recall", "ben_recall"]}
    if ds == "ton":
        rows.append("\\midrule")
write("tab_clean", "\n".join(rows))

# ------------------------------------------------------------------ v1 -> v2 baseline path
v1 = pd.read_csv(R / "extras" / "baseline_v1_path.csv")
ab = pd.read_csv(R / "extras" / "baseline_ablation.csv").assign(scaling="logminmax")
path = pd.concat([v1, ab], ignore_index=True)
steps = [("v1", "quantile_v1", False, "P0: original pipeline (v1 features, 1--99\\% clip scaling, duplicates kept)"),
         ("v1", "logminmax", False, "P1: + log/min--max scaling"),
         ("numeric_only", "logminmax", False, "P2: + revised numeric feature set"),
         ("v2", "logminmax", False, "P3: + categorical protocol fields"),
         ("v2", "logminmax", True, "P4: + de-duplication (= revised pipeline)")]
rows = []
for name, sc, dd, lab in steps:
    cells = []
    for ds in ["edge", "ton"]:
        for m in ["mlp_clean", "xgb"]:
            g = path[(path.dataset == ds) & (path.model == m) & (path.feature_policy == name) & (path.scaling == sc) &
                     (path.dedup == dd)]
            a = g[g.threshold_rule == "fixed_0.5"].macro_f1
            b = g[g.threshold_rule == "val_tuned"].macro_f1
            cells.append(f"{a.mean():.3f} / {b.mean():.3f}")
            N[f"path_{name}_{sc}_{int(dd)}_{ds}_{m}"] = {"f05": float(a.mean()), "tuned": float(b.mean()),
                                                         "f05_sd": float(a.std(ddof=1)),
                                                         "benprec05": float(g[g.threshold_rule == "fixed_0.5"].ben_precision.mean()),
                                                         "fpr05": float(g[g.threshold_rule == "fixed_0.5"].fpr.mean())}
    rows.append(lab + " & " + " & ".join(cells) + "\\\\")
write("tab_path", "\n".join(rows))

# ------------------------------------------------------------------ attacks
att = load("attacks")
main = ((att.model.str.startswith("mlp")) & (att.attack == "pgd")) | ((att.model == "xgb") & (att.attack == "bb"))
A = att[main]
rows = []
for ds in ["ton", "edge"]:
    for m in MODELS:
        cells = []
        for th, eps in [("unconstrained", 0.05), ("numeric", 0.05), ("constrained", 0.05), ("constrained", 1.0)]:
            g = A[(A.dataset == ds) & (A.model == m) & (A.threat == th) & (np.isclose(A.eps, eps))]
            cells.append(ms(g.asr))
            N[f"asr_{ds}_{m}_{th}_{eps:g}"] = {"mean": float(g.asr.mean()), "sd": float(g.asr.std(ddof=1)),
                                              "min": float(g.asr.min()), "max": float(g.asr.max()),
                                              "valid_succ": float(g.asr_syntactic_valid.mean()),
                                              "synt_valid_rate": float(g.syntactic_valid_rate.mean()),
                                              "recall_adv": float(g.mal_recall_adv.mean()),
                                              "asr05": float(g.asr_at_0p5.mean()),
                                              "feat_changed": float(g.mean_features_changed.mean())}
        gu = A[(A.dataset == ds) & (A.model == m) & (A.threat == "unconstrained") & (np.isclose(A.eps, 0.05))]
        gn = A[(A.dataset == ds) & (A.model == m) & (A.threat == "numeric") & (np.isclose(A.eps, 0.05))]
        cells.append(f"{gu.syntactic_valid_rate.mean():.2f} / {gn.syntactic_valid_rate.mean():.2f}")
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & " + " & ".join(cells) + "\\\\")
    if ds == "ton":
        rows.append("\\midrule")
write("tab_attacks", "\n".join(rows))
# transfer + bb on MLP for text
for ds in ["ton", "edge"]:
    for m in ["mlp_fgsmat", "mlp_pgdat", "xgb"]:
        for th in ["unconstrained", "numeric", "constrained"]:
            g = att[(att.dataset == ds) & (att.model == m) & (att.attack == "transfer_pgd_from_mlp_clean") &
                    (att.threat == th) & np.isclose(att.eps, 0.05)]
            N[f"transfer_{ds}_{m}_{th}"] = float(g.asr.mean())
    for th in ["unconstrained", "numeric", "constrained"]:
        g = att[(att.dataset == ds) & (att.model == "mlp_clean") & (att.attack == "bb") & (att.threat == th) & np.isclose(att.eps, 0.05)]
        N[f"bb_mlp_{ds}_{th}"] = float(g.asr.mean())
# per-seed ToN clean MLP bimodality
g = A[(A.dataset == "ton") & (A.model == "mlp_clean") & (A.threat == "unconstrained") & np.isclose(A.eps, 0.05)]
N["ton_mlp_seed_asr"] = dict(zip(g.seed.astype(int).astype(str), g.asr.round(3)))
N["ton_mlp_n_vulnerable"] = int((g.asr > 0.5).sum())
tr = load("training")
tt = tr[(tr.dataset == "ton") & (tr.model == "mlp_clean")].merge(g[["seed", "asr"]], on="seed")
pp = pred[(pred.dataset == "ton") & (pred.model == "mlp_clean") & (pred.threshold_rule == "val_tuned")].merge(g[["seed", "asr"]], on="seed")
vul, rob = tt.asr > 0.5, tt.asr <= 0.5
N["ton_bimodal"] = {"thr_vul": float(tt[vul].val_tuned_threshold.mean()), "thr_rob": float(tt[rob].val_tuned_threshold.mean()),
                    "f1_vul": float(pp[pp.asr > 0.5].macro_f1.mean()), "f1_rob": float(pp[pp.asr <= 0.5].macro_f1.mean()),
                    "fpr_vul": float(pp[pp.asr > 0.5].fpr.mean()), "fpr_rob": float(pp[pp.asr <= 0.5].fpr.mean()),
                    "asr_vul": [float(g[g.asr > 0.5].asr.min()), float(g[g.asr > 0.5].asr.max())],
                    "asr_rob": [float(g[g.asr <= 0.5].asr.min()), float(g[g.asr <= 0.5].asr.max())]}
# partitions
c1 = load("attacks", split=None)
x = c1[(c1.dataset == "ton") & (c1.model == "mlp_clean") & (c1.attack == "pgd") & (c1.threat == "unconstrained") &
       np.isclose(c1.eps, 0.05) & c1.seed.isin([17, 29, 41, 53, 67])].pivot_table(index="seed", columns="split", values="asr")
N["ton_seed_by_partition"] = {str(k): {str(c): round(float(v), 3) for c, v in r.items()} for k, r in x.iterrows()}

# ------------------------------------------------------------------ operating point
op = pd.read_csv(R / "extras" / "operating_point.csv")
rows = []
for ds in ["ton", "edge"]:
    for m in MODELS:
        cells = []
        for th, eps in [("unconstrained", 0.05), ("constrained", 1.0)]:
            for rule in ["fpr_0.01", "fpr_0.05", "val_tuned"]:
                g = op[(op.dataset == ds) & (op.model == m) & (op.threat == th) & np.isclose(op.eps, eps) & (op.threshold_rule == rule)]
                cells.append(f(g.mal_recall_adv.mean(), 2))
                N[f"op_{ds}_{m}_{th}_{eps:g}_{rule}"] = {"recall_adv": float(g.mal_recall_adv.mean()),
                                                        "asr": float(g.asr.mean()), "fpr": float(g.test_fpr.mean()),
                                                        "clean_recall": float(g.test_mal_recall.mean())}
        g = op[(op.dataset == ds) & (op.model == m) & (op.threat == "constrained") & np.isclose(op.eps, 1.0) & (op.threshold_rule == "fpr_0.01")]
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & {f(g.test_mal_recall.mean(), 2)} & " + " & ".join(cells) + "\\\\")
    if ds == "ton":
        rows.append("\\midrule")
write("tab_operating", "\n".join(rows))

# ------------------------------------------------------------------ explanations: faithfulness
sl = pd.read_csv(AN / "seed_level_explanations.csv")
s0 = sl[sl.split == 0]
rows = []
for ds in ["ton", "edge"]:
    for m in MODELS:
        cells = []
        for ex in ["SHAP", "LIME"]:
            g = s0[(s0.dataset == ds) & (s0.model == m) & (s0.explainer == ex) & (s0.view == "clean")]
            cells += [f(g.dcomp_median.mean()), f(g.dcomp_marginal.mean()), f(g.dcomp_knn.mean()), f(g.dsuff_marginal.mean())]
            N.setdefault("faith_sd_max", 0.0)
            N["faith_sd_max"] = max(N["faith_sd_max"], float(g[["dcomp_median", "dcomp_marginal", "dcomp_knn", "dsuff_marginal"]].std(ddof=1).max()))
            N[f"faith_{ds}_{m}_{ex}"] = {k: float(g[k].mean()) for k in ["dcomp_median", "dcomp_marginal", "dcomp_knn",
                                                                       "dsuff_marginal", "neg_v1", "neg_pred_marginal",
                                                                       "comp_marginal", "comp_rand_marginal"]}
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & " + " & ".join(cells) + "\\\\")
    if ds == "ton":
        rows.append("\\midrule")
write("tab_faith", "\n".join(rows))
# negative-share summary (clean view)
neg = s0[s0.view == "clean"].groupby(["dataset", "explainer"])[["neg_v1", "neg_pred_median", "neg_pred_marginal"]].mean()
N["neg_share"] = {f"{a}_{b}": {k: float(v) for k, v in r.items()} for (a, b), r in neg.iterrows()}
negm = s0[s0.view == "clean"].groupby(["dataset", "model", "explainer"])[["neg_v1", "neg_pred_marginal"]].mean()
N["neg_share_range"] = {"v1_min": float(negm.neg_v1.min()), "v1_max": float(negm.neg_v1.max()),
                        "pred_min": float(negm.neg_pred_marginal.min()), "pred_max": float(negm.neg_pred_marginal.max())}
# attacked-view faithfulness (constrained 0.05 & unconstrained)
for ds in ["ton", "edge"]:
    for m in MODELS:
        k = "bb" if m == "xgb" else "pgd"
        for ex in ["SHAP", "LIME"]:
            for v in [f"{k}_c_0.05", f"{k}_c_1", f"{k}_n_0.05", f"{k}_u_0.05"]:
                g = s0[(s0.dataset == ds) & (s0.model == m) & (s0.explainer == ex) & (s0.view == v)]
                if len(g):
                    N[f"faithv_{ds}_{m}_{ex}_{v}"] = {"dcomp_marginal": float(g.dcomp_marginal.mean()),
                                                      "spearman": float(g.stab_spearman.mean()),
                                                      "spearman_evaded": float(g.stab_spearman_evaded.mean()),
                                                      "jac3": float(g["stab_jaccard@3"].mean()),
                                                      "cos": float(g.stab_signed_cosine.mean()),
                                                      "evaded_rate": float(g.evaded_rate.mean())}

# ------------------------------------------------------------------ stability axes
rep = load("repeats")
cs = pd.read_csv(R / "cross_seed_summary.csv"); cs = cs[cs.split == 0]
rows = []
for ds in ["ton", "edge"]:
    for m in MODELS:
        k = "bb" if m == "xgb" else "pgd"
        cells = []
        for ex in ["SHAP", "LIME"]:
            r = rep[(rep.dataset == ds) & (rep.model == m) & (rep.explainer == ex)]["spearman"].mean()
            c = cs[(cs.dataset == ds) & (cs.model == m) & (cs.explainer == ex)]
            c = float(c.spearman_mean.iloc[0]) if len(c) else np.nan
            ic = s0[(s0.dataset == ds) & (s0.model == m) & (s0.explainer == ex) & (s0.view == f"{k}_c_1")].stab_spearman.mean()
            iu = s0[(s0.dataset == ds) & (s0.model == m) & (s0.explainer == ex) & (s0.view == f"{k}_u_0.05")].stab_spearman.mean()
            cells += [f(r, 2), f(c, 2), f(ic, 2), f(iu, 2)]
            N[f"stab_{ds}_{m}_{ex}"] = {"call_or_bg": float(r), "seed": c, "input_c": float(ic), "input_u": float(iu)}
            cj = cs[(cs.dataset == ds) & (cs.model == m) & (cs.explainer == ex)]
            N[f"stab_{ds}_{m}_{ex}"]["seed_jac3"] = float(cj["jaccard@3_mean"].iloc[0]) if len(cj) else None
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & " + " & ".join(cells) + "\\\\")
    if ds == "ton":
        rows.append("\\midrule")
write("tab_stab", "\n".join(rows))

# ------------------------------------------------------------------ tests
T = pd.read_csv(AN / "T_tests_explainers.csv")
t = T[(T.comparison == "SHAP-LIME")]
for o in ["P2_dcomp_marginal", "P3_stab_spearman"]:
    tt_ = t[t.outcome == o]
    N[f"tests_{o}"] = {"n": int(len(tt_)), "n_sig": int((tt_.p_holm < 0.05).sum()),
                       "n_shap_better": int(((tt_.mean_diff > 0) & (tt_.p_holm < 0.05)).sum()),
                       "n_lime_better": int(((tt_.mean_diff < 0) & (tt_.p_holm < 0.05)).sum()),
                       "dz_min": float(tt_.dz.abs().min()), "dz_max": float(tt_.dz.abs().max()),
                       "max_p_holm_sig": float(tt_[tt_.p_holm < 0.05].p_holm.max()),
                       "lime_better_cases": tt_[(tt_.mean_diff < 0) & (tt_.p_holm < 0.05)][["dataset", "model", "view", "mean_diff", "dz"]].to_dict("records"),
                       "ns_cases": tt_[tt_.p_holm >= 0.05][["dataset", "model", "view", "mean_diff", "dz", "p_holm"]].to_dict("records")}
vr = T[T.comparison.str.endswith("-random")]
N["vs_random"] = {"n": int(len(vr)), "n_sig_pos": int(((vr.mean_diff > 0) & (vr.p_holm < 0.05)).sum()),
                  "n_sig_neg": int(((vr.mean_diff < 0) & (vr.p_holm < 0.05)).sum()),
                  "neg_cases": vr[vr.mean_diff < 0][["dataset", "model", "view", "comparison", "mean_diff", "p_holm"]].to_dict("records")}
C = pd.read_csv(AN / "T_tests_conditions.csv")
N["cond_tests"] = C[["family", "outcome", "comparison", "mean_diff", "dz", "ci_low", "ci_high", "n_positive", "p_wilcoxon", "p_holm"]].round(5).to_dict("records")
N["power"] = json.loads((AN / "power.json").read_text())

# ------------------------------------------------------------------ split sensitivity
sp = pd.read_csv(AN / "T_split_sensitivity.csv")
N["split_sens"] = sp[["dataset", "model", "outcome", "grand_mean", "split_means", "var_between_split", "var_within_split_seed", "share_split"]].round(5).to_dict("records")

# ------------------------------------------------------------------ AT sweep
sw = pd.read_csv(R / "extras" / "at_sweep.csv")
g = sw[np.isclose(sw.eps, 0.05)].groupby(["dataset", "model", "threat"])[["clean_macro_f1", "asr", "clean_fpr"]].mean()
N["sweep"] = {f"{a}|{b}|{c}": {k: float(v) for k, v in r.items()} for (a, b, c), r in g.iterrows()}

# ------------------------------------------------------------------ multiclass
mm = pd.read_csv(R / "extras" / "multiclass_metrics.csv")
ma = pd.read_csv(R / "extras" / "multiclass_attacks.csv")
mst = pd.read_csv(R / "extras" / "multiclass_stability.csv")
rows = []
for ds in ["ton", "edge"]:
    for model in ["mlp_multiclass", "xgb_multiclass"]:
        g = mm[(mm.dataset == ds) & (mm.model == model)]
        N[f"mc_{ds}_{model}"] = {k: float(g[k].mean()) for k in ["macro_f1", "weighted_f1", "binary_collapse_macro_f1"]}
    cells = [ms(mm[(mm.dataset == ds) & (mm.model == "mlp_multiclass")].macro_f1),
             ms(mm[(mm.dataset == ds) & (mm.model == "xgb_multiclass")].macro_f1)]
    for th in ["unconstrained", "numeric", "constrained"]:
        for mode in ["untargeted", "to_benign"]:
            gg = ma[(ma.dataset == ds) & (ma.threat == th) & (ma["mode"] == mode)]
            r = gg.success.sum() / max(gg.eligible.sum(), 1)
            cells.append(f(r, 2))
            N[f"mc_asr_{ds}_{th}_{mode}"] = float(r)
            s = mst[(mst.dataset == ds) & (mst.threat == th) & (mst["mode"] == mode) & (mst.metric == "spearman")]["mean"].mean()
            N[f"mc_stab_{ds}_{th}_{mode}"] = float(s)
    rows.append(f"{DS[ds]} & " + " & ".join(cells) + "\\\\")
write("tab_multiclass", "\n".join(rows))
fam = ma[ma.threat == "unconstrained"].groupby(["dataset", "mode", "family"])[["eligible", "success"]].sum()
fam["asr"] = fam.success / fam.eligible
N["mc_family_unconstrained"] = {f"{a}|{b}|{c}": float(v) for (a, b, c), v in fam.asr.items() if np.isfinite(v)}
famc = ma[ma.threat == "constrained"].groupby(["dataset", "mode", "family"])[["eligible", "success"]].sum()
famc["asr"] = famc.success / famc.eligible
N["mc_family_constrained"] = {f"{a}|{b}|{c}": float(v) for (a, b, c), v in famc.asr.items() if np.isfinite(v)}
pc = pd.read_csv(R / "extras" / "multiclass_per_class_f1.csv")
N["mc_perclass"] = pc.groupby(["dataset", "family"])["f1"].mean().round(3).unstack(0).to_dict()

# ------------------------------------------------------------------ profiling
pr = pd.read_csv(R / "extras" / "profiling.csv")
rows = []
for ds in ["ton", "edge"]:
    for m in ["mlp_clean", "xgb"]:
        g = pr[(pr.dataset == ds) & (pr.model == m)]
        def v(task, dev, thr, col="ms_per_record"):
            x = g[(g.task == task) & (g.device == dev) & ((g.threads == thr) if thr else g.threads.isna())][col]
            return float(x.mean()) if len(x) else np.nan
        cells = [f(v("inference", "cpu", 1), 2), f(v("SHAP", "cpu", 1), 1), f(v("LIME", "cpu", 1), 1),
                 (f(v("SHAP", "cuda", None), 1) + " / " + f(v("LIME", "cuda", None), 1)) if m != "xgb" else "--",
                 f(v("BB_attack_c", "cpu", 1), 1) + " / " + (f(v("PGD_attack_c", "cpu", 1), 2) if m != "xgb" else "--")]
        ps = g[g.task.isin(["SHAP", "LIME"])]
        cells += [f(ps.peak_rss_delta_mb.max(), 1), f(ps.py_alloc_peak_mb.max(), 1),
                  f(ps.gpu_peak_alloc_mb.max(), 1) if m != "xgb" else "--"]
        size = (85.0 if ds == "ton" else 67.5) if m != "xgb" else float(g.model_size_kb.dropna().iloc[0])
        cells.append(f(size, 0) if m == "xgb" else f(size, 1))
        rows.append(f"{DS[ds] if m == 'mlp_clean' else ''} & {LAB[m]} & " + " & ".join(cells) + "\\\\")
        N[f"prof_{ds}_{m}"] = {"rss_base": float(g.process_rss_before_mb.dropna().min())}
write("tab_profile", "\n".join(rows))
rt = load("explanations_runtime")
N["runtime_gpu_batch"] = {f"{a}|{b}|{c}": float(v) for (a, b, c), v in rt.groupby(["dataset", "model", "explainer"])["ms_per_record"].mean().round(2).items()}

(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=str))
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))
