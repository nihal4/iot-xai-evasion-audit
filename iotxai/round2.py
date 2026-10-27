"""Second-round analyses.

(1) robustness_r2 : all 20 seeds; attacks that minimise the malicious score without
    early stopping, scored at the validation-tuned threshold, the fixed 0.5 threshold
    and five validation-matched false-positive budgets; threshold-free measures (score
    and log-odds change, ROC-AUC of benign vs. attacked malicious records); projection of
    unconstrained / numeric-only adversarial records onto the valid input space;
    stronger attacks on XGBoost (black-box with 3,000 queries and a tree-aware
    coordinate search over the model's own split thresholds).
(2) explainer_fairness : LIME sample-size / kernel-width / discretisation grid and
    KernelSHAP as an estimator common to both model families.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import attacks as A
from . import explain as E
from .data import prepare
from .models import resolve_device
from .report import print_block
from .runner import build_models

FPR_TARGETS = [0.001, 0.005, 0.01, 0.02, 0.05]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ------------------------------------------------------------ projection onto the valid space
def project_valid(prep, Xa, X0):
    """Nearest syntactically valid record: each one-hot block -> the nearest valid
    pattern (level k if its value > 0.5, else the all-zero 'absent/unseen' pattern);
    integer fields -> nearest integer in raw units within the training range;
    data-supported relations restored by raising the left-hand side."""
    Xp = np.asarray(Xa, np.float32).copy()
    for g in prep.groups:
        if g["kind"] != "cat":
            continue
        B = Xp[:, g["cols"]]
        k = B.argmax(1)
        on = B[np.arange(len(B)), k] > 0.5
        P = np.zeros_like(B)
        P[np.arange(len(B))[on], k[on]] = 1.0
        Xp[:, g["cols"]] = P
    for j, c in enumerate(prep.columns):
        if prep.col_kind[j] == "num" and prep.params[c]["integer"]:
            changed = np.abs(Xp[:, j] - X0[:, j]) > 1e-7
            if not changed.any():
                continue
            p = prep.params[c]
            r = np.clip(np.round(prep.to_raw(Xp, j)), p["raw_min"], p["raw_max"])
            Xp[:, j] = np.where(changed, prep.from_raw(r, j), Xp[:, j])
    cols = prep.columns
    for rel in prep.relations:
        if not rel["kept"]:
            continue
        jl = cols.index(rel["lhs"])
        rhs = sum(cf * prep.to_raw(Xp, cols.index(f)) for f, cf in rel["rhs"].items())
        lhs = prep.to_raw(Xp, jl)
        viol = lhs < rhs - 1e-6
        if viol.any():
            cap = prep.params[rel["lhs"]]["raw_max"]
            Xp[:, jl] = np.where(viol, prep.from_raw(np.minimum(np.ceil(rhs), cap), jl), Xp[:, jl])
    return Xp


# ------------------------------------------------------------ coordinate search on split points
def _candidates(det, j, prep, grid=32):
    if det.kind == "xgb":
        if not hasattr(det, "_splits"):
            df = det.obj.get_booster().trees_to_dataframe()
            df = df[df.Feature != "Leaf"]
            det._splits = {}
            for f, s in zip(df.Feature.values, df.Split.values):
                jj = int(str(f).lstrip("f")) if str(f).startswith("f") else prep.columns.index(f)
                det._splits.setdefault(jj, set()).add(float(s))
            det._splits = {k: np.array(sorted(v), np.float32) for k, v in det._splits.items()}
        c = det._splits.get(j, np.zeros(0, np.float32))
        # a split s sends x >= s to the right branch: try s itself and a value just below it
        return np.unique(np.concatenate([c, np.nextafter(c, np.float32(-1))]).clip(0, 1))
    return np.linspace(0, 1, grid + 1, dtype=np.float32)


def coord_search(det, X, eps, prep, threat, passes=3, chunk=200_000):
    """Greedy coordinate descent on P(malicious): for one feature at a time, every
    candidate value (the detector's own split thresholds for XGBoost, a 33-point grid for
    the MLPs) that is feasible under the threat model is scored and the best is kept
    (an exact line search for trees). In the constrained model every proposal is
    repaired before it is scored."""
    X = np.asarray(X, np.float32)
    n, d = X.shape
    mut, up = A.masks(prep, threat, d)
    lo = np.where(up, X, X - eps).clip(0, 1)
    hi = (X + eps).clip(0, 1)
    cur = X.copy(); best = det.predict(cur)
    cols = [j for j in np.where(mut)[0] if prep.col_kind[j] == "num" or threat == "unconstrained"]
    cand = {j: _candidates(det, j, prep) for j in cols}
    for _ in range(passes):
        improved = False
        for j in cols:
            C = cand[j]
            if len(C) == 0:
                continue
            feas = (C[None, :] >= lo[:, j:j + 1] - 1e-7) & (C[None, :] <= hi[:, j:j + 1] + 1e-7) & \
                   (np.abs(cur[:, j:j + 1] - C[None, :]) > 1e-7)
            r_idx, c_idx = np.nonzero(feas)
            if len(r_idx) == 0:
                continue
            P = np.empty(len(r_idx)); Xs = None
            for s in range(0, len(r_idx), chunk):
                r, c = r_idx[s:s + chunk], c_idx[s:s + chunk]
                Xn = cur[r].copy(); Xn[:, j] = C[c]
                if threat == "constrained":
                    Xn = A.repair(prep, Xn, X[r])
                P[s:s + chunk] = det.predict(Xn)
            order = np.lexsort((P, r_idx))
            r_s, P_s, c_s = r_idx[order], P[order], c_idx[order]
            first = np.r_[True, r_s[1:] != r_s[:-1]]
            rr, pp, cc = r_s[first], P_s[first], c_s[first]
            imp = pp < best[rr] - 1e-9
            if imp.any():
                improved = True
                rr, cc = rr[imp], cc[imp]
                Xn = cur[rr].copy(); Xn[:, j] = C[cc]
                if threat == "constrained":
                    Xn = A.repair(prep, Xn, X[rr])
                cur[rr] = Xn; best[rr] = pp[imp]
        if not improved:
            break
    return cur


# ------------------------------------------------------------ (1) robustness at all seeds
def robustness_r2(cfg, log):
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    emax = max(cfg.epsilons)
    settings = [("unconstrained", emax), ("numeric", emax), ("constrained", emax), ("constrained", 1.0)]
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        idx = z["attack_idx"]; X, y = z["X_test"][idx], z["y_test"][idx]
        ben_val = z["X_val"][z["y_val"] == 0]
        ben_test = z["X_test"][z["y_test"] == 0]
        for seed in cfg.seeds:
            t_seed = time.time()
            op_rows, mg_rows = [], []
            dets, _, tr = build_models(z, cfg, seed, device, cfg.models, log)
            for name, det in dets.items():
                pt = det.predict(z["X_test"]); pb_val = det.predict(ben_val); pb_test = det.predict(ben_test)
                p0 = det.predict(X)
                thr = {"val_tuned": det.threshold, "fixed_0.5": 0.5}
                for t in FPR_TARGETS:
                    thr[f"fpr_{t:g}"] = float(np.quantile(pb_val, 1 - t))
                auc_clean = roc_auc_score(np.r_[np.zeros(len(pb_test)), np.ones(len(p0))], np.r_[pb_test, p0])
                for threat, eps in settings:
                    variants = {}
                    t0 = time.perf_counter()
                    if det.kind == "mlp":
                        variants["pgd"] = A.pgd(det, X, eps, prep, threat, steps=cfg.pgd_steps, seed=seed)
                    variants["bb300"], _ = A.blackbox(det, X, eps, prep, threat, cfg.bb_queries, seed=seed, thr=-1.0)
                    if det.kind == "xgb":
                        variants["bb3000"], _ = A.blackbox(det, X, eps, prep, threat, cfg.bb_queries_strong,
                                                           seed=seed + 1, thr=-1.0)
                    if threat != "unconstrained" and (det.kind == "xgb" or threat == "constrained"):
                        variants["coord"] = coord_search(det, X, eps, prep, threat, passes=cfg.coord_passes)
                    # an attacker never keeps a change that raises the score: fall back to the clean record
                    for k in list(variants):
                        worse = det.predict(variants[k]) > p0
                        variants[k] = np.where(worse[:, None], X, variants[k]).astype(np.float32)
                    # per-record strongest attack among those run (lowest malicious score)
                    keys = list(variants)
                    P = np.stack([det.predict(variants[k]) for k in keys])
                    pick = P.argmin(0)
                    variants["best"] = np.stack([variants[k] for k in keys])[pick, np.arange(len(X))]
                    secs = time.perf_counter() - t0
                    for aname, Xa in variants.items():
                        sets = [("raw", Xa)]
                        if threat in ("unconstrained", "numeric"):
                            sets.append(("projected", project_valid(prep, Xa, X)))
                        for form, Xf in sets:
                            p1 = det.predict(Xf)
                            v = A.validity(prep, Xf, X)
                            auc_adv = roc_auc_score(np.r_[np.zeros(len(pb_test)), np.ones(len(p1))], np.r_[pb_test, p1])
                            dl = _logit(p0) - _logit(p1)
                            mg_rows.append({"dataset": ds, "seed": seed, "model": name, "attack": aname, "form": form,
                                            "threat": threat, "eps": eps, "val_tuned_threshold": det.threshold,
                                            "auc_clean": auc_clean, "auc_adv": auc_adv,
                                            "mean_p_drop": float(np.mean(p0 - p1)), "median_p_drop": float(np.median(p0 - p1)),
                                            "mean_logodds_drop": float(np.mean(dl)), "median_logodds_drop": float(np.median(dl)),
                                            "share_logodds_drop_gt1": float(np.mean(dl > 1)),
                                            "median_clean_margin_tuned": float(np.median(_logit(p0) - _logit(det.threshold))),
                                            "median_clean_margin_0p5": float(np.median(_logit(p0))),
                                            "syntactic_valid_rate": float(v["syntactic_valid"].mean()),
                                            "threat_model_valid_rate": float(v["threat_model_valid"].mean()),
                                            "seconds_all_attacks": secs})
                            for tname, tv in thr.items():
                                pred0, pred1 = p0 >= tv, p1 >= tv
                                elig = (y == 1) & pred0
                                op_rows.append({"dataset": ds, "seed": seed, "model": name, "attack": aname, "form": form,
                                                "threat": threat, "eps": eps, "threshold_rule": tname, "threshold": tv,
                                                "test_fpr": float(np.mean(pt[z["y_test"] == 0] >= tv)),
                                                "mal_recall_clean": float(pred0.mean()), "mal_recall_adv": float(pred1.mean()),
                                                "eligible": int(elig.sum()),
                                                "asr": float((elig & ~pred1).sum() / max(elig.sum(), 1)),
                                                "asr_threat_model_valid": float((elig & ~pred1 & v["threat_model_valid"]).sum()
                                                                                / max(elig.sum(), 1))})
                log(f"  r2 robustness {ds} seed {seed} {name} done")
            op, mg = pd.DataFrame(op_rows), pd.DataFrame(mg_rows)
            op.to_csv(out / f"r2_operating_{ds}_seed{seed}.csv", index=False)
            mg.to_csv(out / f"r2_margin_{ds}_seed{seed}.csv", index=False)
            print_block("extras", f"r2_operating_{ds}_seed{seed}", op)
            print_block("extras", f"r2_margin_{ds}_seed{seed}", mg)
            log(f"r2 robustness {ds} seed {seed} finished in {time.time() - t_seed:.0f}s")


# ------------------------------------------------------------ (2) explainer fairness
def _kernel_shap(det, X, bg, nsamples, seed):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(lambda a: det.predict(np.asarray(a, np.float32)), np.asarray(bg, np.float32))
    v = ex.shap_values(np.asarray(X, np.float32), nsamples=nsamples, silent=True)
    v = np.asarray(v[-1] if isinstance(v, list) else v)
    return v[..., -1] if v.ndim == 3 else v


def explainer_fairness(cfg, log):
    from lime.lime_tabular import LimeTabularExplainer
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    refs = ["median", "marginal", "knn"]
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        groups = prep.groups; G = len(groups); d = z["X_train"].shape[1]
        cat_cols = [j for j, k in enumerate(prep.col_kind) if k == "cat"]
        Xtr = z["X_train"]; bg = Xtr[z["shap_bg_idx"]]; lbg = Xtr[z["lime_bg_idx"]]; pool = Xtr[z["knn_pool_idx"]]
        nb = cfg.fair_rows // 2
        Xv = np.concatenate([z["X_test"][z["explain_benign_idx"][:nb]], z["X_test"][z["explain_mal_idx"][:nb]]])
        dev_knn = str(device) if device.type == "cuda" else "cpu"
        lime_grid = [(ns, kw, False) for ns in cfg.fair_lime_samples for kw in cfg.fair_lime_kernel] + \
                    [(cfg.fair_lime_samples[0], 0.75, True)]
        for seed in cfg.seeds:
            rows = []
            dets, _, _ = build_models(z, cfg, seed, device, cfg.models, log)
            for name, det in dets.items():
                masker = E.Masker(groups, d, bg, pool, cfg.marginal_draws, cfg.knn_neighbors, seed=seed * 13 + 5, device=dev_knn)
                p = det.predict(Xv); cls = (p >= det.threshold).astype(int)
                rng = np.random.default_rng(seed + 4242)
                rand = {r: np.zeros(len(Xv)) for r in refs}; rand_s = {r: np.zeros(len(Xv)) for r in refs}
                for _ in range(cfg.random_order_draws):
                    fr = E.faithfulness(det, Xv, rng.random((len(Xv), G)), masker, cfg.k_fracs, refs, cls)
                    for r in refs:
                        rand[r] += fr[r][0] / cfg.random_order_draws; rand_s[r] += fr[r][1] / cfg.random_order_draws

                def score(Ag_list, label, secs, extra):
                    Ag = Ag_list[0]
                    evid = np.where(cls[:, None] == 1, Ag, -Ag)
                    fa = E.faithfulness(det, Xv, evid, masker, cfg.k_fracs, refs, cls)
                    row = {"dataset": ds, "seed": seed, "model": name, "explainer": label, "n": len(Xv),
                           "ms_per_record": 1000 * secs / len(Xv), **extra}
                    for r in refs:
                        row[f"dcomp_{r}"] = float(np.mean(fa[r][0] - rand[r]))
                        row[f"dsuff_{r}"] = float(np.mean(fa[r][1] - rand_s[r]))
                    if len(Ag_list) > 1:
                        st = E.stability(Ag_list[0], Ag_list[1], cfg.stability_ks)
                        for k, v in st.items():
                            row[f"rep_{k}"] = float(np.nanmean(v))
                    rows.append(row)

                # reference estimators used in the main analysis, two background draws
                A1, s1 = E.timed(E.shap_values, det, Xv, bg)
                rb = np.random.default_rng(seed * 31 + 7)
                A2 = E.shap_values(det, Xv, Xtr[rb.choice(len(Xtr), cfg.shap_background_rows, replace=False)])
                score([E.to_groups(A1, groups), E.to_groups(A2, groups)],
                      "SHAP_tree" if det.kind == "xgb" else "SHAP_deep", s1, {"config": "main"})
                # KernelSHAP: one estimator for both model families
                kb = bg[:cfg.kernel_bg_rows]
                K1, k1 = E.timed(_kernel_shap, det, Xv, kb, cfg.kernel_nsamples, seed)
                K2 = _kernel_shap(det, Xv, Xtr[rb.choice(len(Xtr), cfg.kernel_bg_rows, replace=False)], cfg.kernel_nsamples, seed + 1)
                score([E.to_groups(K1, groups), E.to_groups(K2, groups)], "KernelSHAP", k1,
                      {"config": f"nsamples={cfg.kernel_nsamples},bg={cfg.kernel_bg_rows}"})
                # LIME grid
                for ns, kw, disc in lime_grid:
                    width = kw * np.sqrt(d)
                    def run(sd):
                        ex = LimeTabularExplainer(np.asarray(lbg, np.float32), mode="classification",
                                                  categorical_features=cat_cols, discretize_continuous=disc,
                                                  sample_around_instance=not disc, kernel_width=width, random_state=int(sd))
                        fn = lambda a: np.column_stack([1 - det.predict(np.asarray(a, np.float32)),
                                                        det.predict(np.asarray(a, np.float32))])
                        Araw = np.zeros((len(Xv), d), np.float32)
                        for i in range(len(Xv)):
                            e = ex.explain_instance(Xv[i], fn, labels=(1,), num_features=d, num_samples=ns)
                            for jj, w in e.local_exp[1]:
                                Araw[i, jj] = w
                        return E.to_groups(Araw, groups)
                    L1, ls = E.timed(run, seed * 1000 + 1)
                    L2 = run(seed * 1000 + 2)
                    score([L1, L2], "LIME", ls, {"config": f"samples={ns},kernel={kw:g}sqrt(d),discretize={int(disc)}",
                                                 "lime_samples": ns, "lime_kernel_mult": kw, "lime_discretize": int(disc)})
                log(f"  fairness {ds} seed {seed} {name} done")
            df = pd.DataFrame(rows)
            df.to_csv(out / f"r2_explainer_fairness_{ds}_seed{seed}.csv", index=False)
            print_block("extras", f"r2_explainer_fairness_{ds}_seed{seed}", df)
