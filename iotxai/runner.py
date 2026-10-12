"""Orchestration: one *run* = (dataset, split, seed).  Each run writes its own
folder under ``results_v2/runs`` and a ``done.json`` marker, so an interrupted
Kaggle session can be resumed and several sessions can be merged."""
from __future__ import annotations

import json
import zlib
import platform
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import attacks as A
from . import explain as E
from .config import Config
from .data import prepare
from .models import (Detector, binary_metrics, family_recall, fit_temperature, resolve_device,
                     set_seed, train_mlp, train_xgb, tune_threshold, ece_score)


class Logger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def environment():
    import sklearn, scipy, shap, xgboost, lime
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "scipy": scipy.__version__, "shap": shap.__version__, "xgboost": xgboost.__version__,
           "lime": getattr(lime, "__version__", "unknown")}
    return env


# ---------------------------------------------------------------- model zoo
def build_models(z, cfg, seed, device, names, log):
    Xtr, ytr, Xva, yva = z["X_train"], z["y_train"], z["X_val"], z["y_val"]
    out, hist_rows, train_rows = {}, [], []
    for name in names:
        if name == "xgb":
            clf, secs = train_xgb(Xtr, ytr, Xva, yva, seed, cfg, device)
            det = Detector("xgb", clf, device); hist = []
        else:
            if name == "mlp_clean":
                kw = dict(mode="clean")
            elif name == "mlp_fgsmat":
                kw = dict(mode="at", eps=cfg.fgsmat_eps, lam=cfg.fgsmat_lambda, steps=1)
            elif name == "mlp_pgdat":
                kw = dict(mode="at", eps=cfg.pgdat_eps, lam=cfg.pgdat_lambda, steps=cfg.pgdat_steps)
            elif name.startswith("sweep_"):
                _, e, l = name.split("_")
                kw = dict(mode="at", eps=float(e), lam=float(l), steps=cfg.pgdat_steps)
            else:
                raise ValueError(name)
            net, hist, secs = train_mlp(Xtr, ytr, Xva, yva, seed, cfg, device, log=log, **kw)
            det = Detector("mlp", net, device)
            det.temperature = fit_temperature(det, Xva, yva)
        pv = det.predict(Xva)
        det.threshold, _ = tune_threshold(yva, pv)
        out[name] = det
        for h in hist:
            hist_rows.append(dict(model=name, **h))
        train_rows.append({"model": name, "train_seconds": secs, "val_tuned_threshold": det.threshold,
                           "temperature": det.temperature})
    return out, pd.DataFrame(hist_rows), pd.DataFrame(train_rows)


def predictive(dets, z, base):
    rows, fam_rows, probs = [], [], {}
    for name, det in dets.items():
        pv, pt = det.predict(z["X_val"]), det.predict(z["X_test"])
        probs[f"pval_{name}"] = pv.astype(np.float16); probs[f"ptest_{name}"] = pt.astype(np.float16)
        for tname, thr in (("fixed_0.5", 0.5), ("val_tuned", det.threshold)):
            m = binary_metrics(z["y_test"], pt, thr)
            rows.append(dict(base, model=name, threshold_rule=tname, **m))
            for fr in family_recall(z["y_test"], z["fam_test"], pt, thr):
                fam_rows.append(dict(base, model=name, threshold_rule=tname, **fr))
        if det.kind == "mlp":
            pc = det.predict_calibrated(z["X_test"])
            rows[-1]["ece_temperature_scaled"] = ece_score(z["y_test"], pc)
    return pd.DataFrame(rows), pd.DataFrame(fam_rows), probs


# ---------------------------------------------------------------- attacks
def run_attacks(dets, z, prep, cfg, base, seed, log, store_views):
    idx = z["attack_idx"]
    X, y, fam = z["X_test"][idx], z["y_test"][idx], z["fam_test"][idx]
    rows, fam_rows, advs = [], [], {}
    emax = max(cfg.epsilons)
    n_exp = len(z["explain_mal_idx"])
    for name, det in dets.items():
        for threat in cfg.threat_models:
            eps_list = list(cfg.epsilons) + (list(cfg.constrained_extra_eps) if threat == "constrained" else [])
            for eps in eps_list:
                kinds = ["bb"] if det.kind == "xgb" else ["fgsm", "pgd", "bb"]
                if eps > max(cfg.epsilons) and det.kind != "xgb":
                    kinds = ["pgd", "bb"]
                for kind in kinds:
                    t0 = time.perf_counter()
                    if kind == "bb":
                        Xa, q = A.blackbox(det, X, eps, prep, threat, cfg.bb_queries, seed=seed, thr=det.threshold)
                    else:
                        Xa = A.pgd(det, X, eps, prep, threat, steps=cfg.pgd_steps, seed=seed, fgsm=(kind == "fgsm"))
                        q = None
                    secs = time.perf_counter() - t0
                    s, elig, succ = A.attack_summary(det, X, y, Xa, det.threshold, prep)
                    s05, _, _ = A.attack_summary(det, X, y, Xa, 0.5, prep)
                    rows.append(dict(base, model=name, attack=kind, threat=threat, eps=eps, seconds=secs,
                                     mean_queries=float(q[elig].mean()) if q is not None and elig.any() else np.nan,
                                     asr_at_0p5=s05["asr"], **s))
                    for f in np.unique(fam):
                        m = fam == f
                        fam_rows.append(dict(base, model=name, attack=kind, threat=threat, eps=eps, family=f,
                                             eligible=int(elig[m].sum()), success=int(succ[m].sum())))
                    tag = f"{kind}_{threat[0]}_{eps:g}"
                    if store_views(name, tag):
                        advs[(name, tag)] = Xa[:n_exp]
                    if name == "mlp_clean" and kind == "pgd":   # transfer to the other detectors
                        for tgt, tdet in dets.items():
                            if tgt == name:
                                continue
                            st, _, _ = A.attack_summary(tdet, X, y, Xa, tdet.threshold, prep)
                            rows.append(dict(base, model=tgt, attack="transfer_pgd_from_mlp_clean", threat=threat,
                                             eps=eps, seconds=np.nan, mean_queries=np.nan, asr_at_0p5=np.nan, **st))
        log(f"  attacks done: {name}")
    return pd.DataFrame(rows), pd.DataFrame(fam_rows), advs


def view_plan(cfg, model_kind):
    emax = max(cfg.epsilons)
    cmax = max(cfg.constrained_extra_eps) if cfg.constrained_extra_eps else emax
    k = "bb" if model_kind == "xgb" else "pgd"
    if cfg.adv_views_reduced:
        return [f"{k}_c_{cmax:g}", f"{k}_n_{emax:g}"]
    if model_kind == "xgb":
        return [f"bb_c_{emax:g}", f"bb_c_{cmax:g}", f"bb_n_{emax:g}", f"bb_u_{emax:g}"]
    return [f"pgd_c_{e:g}" for e in cfg.epsilons] + [f"pgd_c_{cmax:g}", f"pgd_n_{emax:g}", f"pgd_u_{emax:g}"]


# ---------------------------------------------------------------- explanations
def run_explanations(dets, advs, z, prep, cfg, base, seed, device, log, arrays):
    groups = prep.groups
    cat_cols = [j for j, k in enumerate(prep.col_kind) if k == "cat"]
    Xtr = z["X_train"]
    bg = Xtr[z["shap_bg_idx"]]; lime_bg = Xtr[z["lime_bg_idx"]]; pool = Xtr[z["knn_pool_idx"]]
    ben, mal = z["explain_benign_idx"], z["explain_mal_idx"]
    Xc = np.concatenate([z["X_test"][ben], z["X_test"][mal]])
    yc = np.concatenate([z["y_test"][ben], z["y_test"][mal]])
    fc = np.concatenate([z["fam_test"][ben], z["fam_test"][mal]])
    n_b = len(ben)
    refs = ["median", "marginal", "knn"]
    rec_rows, rt_rows = [], []
    dev_knn = str(device) if device.type == "cuda" else "cpu"
    for mname, det in dets.items():
        views = {"clean": Xc}
        for tag in view_plan(cfg, det.kind):
            views[tag] = advs[(mname, tag)]
        clean_groups = {}
        for vname, Xv in views.items():
            masker = E.Masker(groups, Xv.shape[1], bg, pool, cfg.marginal_draws, cfg.knn_neighbors,
                              seed=seed * 7 + len(vname), device=dev_knn)
            p = det.predict(Xv); cls = (p >= det.threshold).astype(int)
            yv = yc if vname == "clean" else np.ones(len(Xv), int)
            fv = fc if vname == "clean" else fc[n_b:]
            # random-ranking control (shared by both explainers)
            rng = np.random.default_rng(seed + 991)
            rand = {r: [np.zeros(len(Xv)), np.zeros(len(Xv))] for r in refs}
            for _ in range(cfg.random_order_draws):
                ev = rng.random((len(Xv), len(groups)))
                fr = E.faithfulness(det, Xv, ev, masker, cfg.k_fracs, refs, cls)
                for r in refs:
                    rand[r][0] += fr[r][0] / cfg.random_order_draws
                    rand[r][1] += fr[r][1] / cfg.random_order_draws
            for ex in ["SHAP", "LIME"]:
                if ex == "SHAP":
                    Araw, secs = E.timed(E.shap_values, det, Xv, bg)
                else:
                    Araw, secs = E.timed(E.lime_values, det, Xv, lime_bg, cat_cols, cfg.lime_num_samples,
                                         seed=seed * 1000 + zlib.crc32(vname.encode()) % 997)
                Ag = E.to_groups(Araw, groups)
                arrays[f"attr__{mname}__{ex}__{vname}"] = Ag.astype(np.float16)
                rt_rows.append(dict(base, model=mname, explainer=ex, view=vname, n=len(Xv), seconds=secs,
                                    ms_per_record=1000 * secs / len(Xv), device=str(device)))
                evid = np.where(cls[:, None] == 1, Ag, -Ag)
                fa = E.faithfulness(det, Xv, evid, masker, cfg.k_fracs, refs, cls)
                # v1-style variant: malicious score, |attribution| ranking, median reference
                v1 = E.faithfulness(det, Xv, np.abs(Ag), masker, cfg.k_fracs, ["median"], np.ones(len(Xv), int))
                rec = pd.DataFrame(dict(rec=np.arange(len(Xv)), y=yv, family=fv, p_mal=p, cls=cls))
                for r in refs:
                    rec[f"comp_{r}"], rec[f"suff_{r}"] = fa[r][0], fa[r][1]
                    rec[f"comp_rand_{r}"], rec[f"suff_rand_{r}"] = rand[r][0], rand[r][1]
                rec["comp_v1style"] = v1["median"][0]
                if vname == "clean":
                    clean_groups[ex] = Ag
                else:
                    Ac = clean_groups[ex][n_b:]
                    st = E.stability(Ac, Ag, cfg.stability_ks + [max(1, int(round(0.2 * len(groups))))])
                    for k_, v_ in st.items():
                        rec[f"stab_{k_}"] = v_
                    p0 = det.predict(Xc[n_b:])
                    rec["evaded"] = ((p0 >= det.threshold) & (p < det.threshold)).astype(int)
                for k_, v_ in base.items():
                    rec[k_] = v_
                rec["model"], rec["explainer"], rec["view"] = mname, ex, vname
                rec_rows.append(rec)
        log(f"  explanations done: {mname}")
    return pd.concat(rec_rows, ignore_index=True), pd.DataFrame(rt_rows)


def run_repeats(dets, z, prep, cfg, base, seed, log):
    """Repeated-call LIME (different sampling seeds) and SHAP background resampling."""
    cat_cols = [j for j, k in enumerate(prep.col_kind) if k == "cat"]
    X = z["X_test"][z["explain_mal_idx"][:cfg.repeat_rows]]
    Xtr = z["X_train"]
    rows = []
    for mname, det in dets.items():
        for ex in ["LIME", "SHAP"]:
            reps = []
            R = cfg.lime_repeats if ex == "LIME" else cfg.shap_bg_repeats
            for r in range(R):
                if ex == "LIME":
                    A_ = E.lime_values(det, X, Xtr[z["lime_bg_idx"]], cat_cols, cfg.lime_num_samples,
                                       seed=seed * 7919 + r)
                else:
                    rng = np.random.default_rng(seed * 104729 + r)
                    bg = Xtr[rng.choice(len(Xtr), cfg.shap_background_rows, replace=False)]
                    A_ = E.shap_values(det, X, bg)
                reps.append(E.to_groups(A_, prep.groups))
            for a, b in combinations(range(R), 2):
                st = E.stability(reps[a], reps[b], cfg.stability_ks)
                rows.append(dict(base, model=mname, explainer=ex, source="lime_sampling" if ex == "LIME" else "shap_background",
                                 rep_a=a, rep_b=b, **{k: float(np.nanmean(v)) for k, v in st.items()}))
        log(f"  repeats done: {mname}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one run
def run_one(cfg: Config, dataset, split_id, seed, log, explain=True, repeats=False):
    out = Path(cfg.out_dir); work = Path(cfg.work_dir)
    rdir = out / "runs" / f"{dataset}_split{split_id}_seed{seed}"
    if (rdir / "done.json").exists():
        log(f"skip {rdir.name} (done)"); return
    rdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    t_run = time.time()
    z, prep = prepare(cfg, dataset, split_id, work, out, log)
    base = {"dataset": dataset, "split": split_id, "seed": seed}
    log(f"RUN {rdir.name} on {device}: d={z['X_train'].shape[1]} groups={len(prep.groups)}")
    dets, hist, trainrows = build_models(z, cfg, seed, device, cfg.models, log)
    for k, v in base.items():
        hist[k] = v; trainrows[k] = v
    hist.to_csv(rdir / "training_history.csv", index=False)
    trainrows.to_csv(rdir / "training.csv", index=False)
    pm, fm, probs = predictive(dets, z, base)
    pm.to_csv(rdir / "predictive.csv", index=False); fm.to_csv(rdir / "predictive_family.csv", index=False)
    arrays = dict(probs)
    wanted = set()
    for name, det in dets.items():
        for v in view_plan(cfg, det.kind):
            wanted.add((name, v))
    am, af, advs = run_attacks(dets, z, prep, cfg, base, seed, log,
                               store_views=lambda n, t: explain and (n, t) in wanted)
    am.to_csv(rdir / "attacks.csv", index=False); af.to_csv(rdir / "attacks_family.csv", index=False)
    for (n, t), Xa in advs.items():
        arrays[f"adv__{n}__{t}"] = Xa.astype(np.float16)
    if explain:
        rec, rt = run_explanations(dets, advs, z, prep, cfg, base, seed, device, log, arrays)
        rec.to_csv(rdir / "explanations_records.csv.gz", index=False)
        from .analysis import seed_level_explanations
        seed_level_explanations(rec).to_csv(rdir / "explanations_seed.csv", index=False)
        rt.to_csv(rdir / "explanations_runtime.csv", index=False)
    if repeats:
        run_repeats(dets, z, prep, cfg, base, seed, log).to_csv(rdir / "repeats.csv", index=False)
    np.savez_compressed(rdir / "arrays.npz", **arrays)
    mdir = rdir / "models"; mdir.mkdir(exist_ok=True)
    for name, det in dets.items():
        if det.kind == "mlp":
            torch.save(det.obj.state_dict(), mdir / f"{name}.pt")
        else:
            det.obj.save_model(str(mdir / f"{name}.json"))
    (rdir / "done.json").write_text(json.dumps({"seconds": time.time() - t_run, **base}))
    log(f"DONE {rdir.name} in {time.time() - t_run:.0f}s")
    if cfg.print_tables:
        from .report import print_run
        print_run(rdir)


def run_all(cfg: Config, log=None):
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    log = log or Logger(out / "log.txt")
    cfg.to_json(out / f"config_{int(time.time())}.json")
    (out / "environment.json").write_text(json.dumps(environment(), indent=2))
    for ds in cfg.datasets:
        for sp in cfg.split_ids:
            for i, seed in enumerate(cfg.seeds):
                try:
                    run_one(cfg, ds, sp, seed, log, explain=cfg.explain, repeats=(i < cfg.repeat_seeds and sp == 0))
                except Exception:
                    log("ERROR in run " + f"{ds}/{sp}/{seed}\n" + traceback.format_exc())
                    raise
