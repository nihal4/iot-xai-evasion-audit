"""Supplementary experiments requested in review:
(1) adversarial-training sensitivity sweep (robustness-accuracy trade-off),
(2) multiclass attack-family detection with targeted/untargeted evasion and
    attribution stability, (3) explanation/inference resource profiling."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from . import attacks as A
from . import explain as E
from .data import prepare
from .models import Detector, MLP, binary_metrics, resolve_device, set_seed, train_mlp, train_xgb
from .runner import Logger, build_models
from .report import print_block


# ------------------------------------------------------------ (1) AT sweep
def at_sweep(cfg, log):
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rows = []
    names = ["mlp_clean"] + [f"sweep_{e:g}_{l:g}" for e in cfg.at_sweep_eps for l in cfg.at_sweep_lambda]
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        idx = z["attack_idx"]; X, y = z["X_test"][idx], z["y_test"][idx]
        for seed in cfg.seeds[:cfg.at_sweep_seeds]:
            dets, _, tr = build_models(z, cfg, seed, device, names, log)
            for name, det in dets.items():
                pt = det.predict(z["X_test"])
                m = binary_metrics(z["y_test"], pt, det.threshold)
                m05 = binary_metrics(z["y_test"], pt, 0.5)
                for threat in cfg.threat_models:
                    for eps in cfg.epsilons:
                        Xa = A.pgd(det, X, eps, prep, threat, steps=cfg.pgd_steps, seed=seed)
                        s, _, _ = A.attack_summary(det, X, y, Xa, det.threshold, prep)
                        rows.append({"dataset": ds, "seed": seed, "model": name, "threat": threat, "eps": eps,
                                     "clean_macro_f1": m["macro_f1"], "clean_macro_f1_at_0p5": m05["macro_f1"],
                                     "clean_fpr": m["fpr"], "clean_fnr": m["fnr"], "threshold": det.threshold,
                                     "asr": s["asr"], "mal_recall_adv": s["mal_recall_adv"]})
            log(f"sweep {ds} seed {seed} done")
            pd.DataFrame(rows).to_csv(out / "at_sweep.csv", index=False)
    print_block("extras", "at_sweep", pd.DataFrame(rows))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ (2) multiclass
class _SoftWrap(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base
    def forward(self, x):
        return torch.softmax(self.base(x), dim=-1)


def _mc_pgd(model, X, eps, prep, threat, steps, target=None, true=None, device="cpu", seed=0):
    mut_np, up_np = A.masks(prep, threat, X.shape[1])
    mut = torch.as_tensor(mut_np, device=device, dtype=torch.float32)
    up = torch.as_tensor(up_np, device=device)
    x0 = torch.as_tensor(X, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    xa = A._project(x0 + (torch.rand(x0.shape, generator=g).to(device) * 2 - 1) * eps, x0, eps, mut, up)
    alpha = 2.5 * eps / steps
    ce = nn.CrossEntropyLoss(reduction="sum")
    for _ in range(steps):
        xa.requires_grad_(True)
        logits = model(xa)
        if target is not None:   # targeted: decrease loss toward target
            loss = -ce(logits, torch.full((len(X),), target, device=device, dtype=torch.long))
        else:                    # untargeted: increase loss of true class
            loss = ce(logits, torch.as_tensor(true, device=device, dtype=torch.long))
        grad = torch.autograd.grad(loss, xa)[0]
        xa = A._project(xa.detach() + alpha * grad.sign(), x0, eps, mut, up)
    Xa = xa.detach().cpu().numpy()
    if threat == "constrained":
        Xa = A.repair(prep, Xa, X)
    return Xa


def multiclass(cfg, log):
    import shap
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    met_rows, att_rows, stab_rows, cls_rows = [], [], [], []
    emax = max(cfg.epsilons)
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        classes = sorted(np.unique(z["fam_train"]))
        cmap = {c: i for i, c in enumerate(classes)}
        benign_cls = cmap.get("normal", None)
        yt = lambda f: np.array([cmap.get(v, -1) for v in f])
        ytr, yva, yte = yt(z["fam_train"]), yt(z["fam_val"]), yt(z["fam_test"])
        mal = z["explain_mal_idx"]
        X, ytrue = z["X_test"][mal], yte[mal]
        for seed in cfg.seeds[:cfg.multiclass_seeds]:
            net, hist, secs = train_mlp(z["X_train"], ytr, z["X_val"], yva, seed, cfg, device,
                                        n_classes=len(classes), log=log)
            det = Detector("mlp", net, device)
            pred = det.logits(z["X_test"]).argmax(1)
            met_rows.append({"dataset": ds, "seed": seed, "model": "mlp_multiclass",
                             "macro_f1": f1_score(yte, pred, average="macro"),
                             "weighted_f1": f1_score(yte, pred, average="weighted"),
                             "binary_collapse_macro_f1": f1_score(yte != benign_cls, pred != benign_cls, average="macro")})
            per = f1_score(yte, pred, average=None, labels=list(range(len(classes))))
            for c, f in zip(classes, per):
                cls_rows.append({"dataset": ds, "seed": seed, "model": "mlp_multiclass", "family": c, "f1": f})
            clf, _ = train_xgb(z["X_train"], ytr, z["X_val"], yva, seed, cfg, device, n_classes=len(classes))
            px = clf.predict_proba(z["X_test"]).argmax(1)
            met_rows.append({"dataset": ds, "seed": seed, "model": "xgb_multiclass",
                             "macro_f1": f1_score(yte, px, average="macro"),
                             "weighted_f1": f1_score(yte, px, average="weighted"),
                             "binary_collapse_macro_f1": f1_score(yte != benign_cls, px != benign_cls, average="macro")})
            # attacks on the 300 explained malicious records
            p0 = det.logits(X).argmax(1)
            correct = p0 == ytrue
            wrapped = _SoftWrap(net).to(device).eval()
            bg = torch.as_tensor(z["X_train"][z["shap_bg_idx"]], device=device)
            expl = shap.DeepExplainer(wrapped, bg)
            def attr(Xin):
                v = np.asarray(expl.shap_values(torch.as_tensor(Xin, device=device), check_additivity=False))
                if v.ndim == 3 and v.shape[0] == len(classes) and v.shape[1] == len(Xin):
                    v = np.transpose(v, (1, 2, 0))
                return v  # (n, d, C)
            Ac = attr(X)
            Ac_true = E.to_groups(Ac[np.arange(len(X)), :, ytrue], prep.groups)
            for threat in cfg.threat_models:
                for mode in ["untargeted", "to_benign"]:
                    if mode == "to_benign" and benign_cls is None:
                        continue
                    Xa = _mc_pgd(net, X, emax, prep, threat, cfg.pgd_steps,
                                 target=benign_cls if mode == "to_benign" else None, true=ytrue,
                                 device=device, seed=seed)
                    p1 = det.logits(Xa).argmax(1)
                    succ = correct & ((p1 != ytrue) if mode == "untargeted" else (p1 == benign_cls))
                    fams = np.array(classes)[ytrue]
                    for f in np.unique(fams):
                        m = fams == f
                        att_rows.append({"dataset": ds, "seed": seed, "threat": threat, "mode": mode, "eps": emax,
                                         "family": f, "eligible": int(correct[m].sum()), "success": int(succ[m].sum())})
                    Aa = attr(Xa)
                    Aa_true = E.to_groups(Aa[np.arange(len(X)), :, ytrue], prep.groups)
                    st = E.stability(Ac_true, Aa_true, cfg.stability_ks)
                    for k, v in st.items():
                        stab_rows.append({"dataset": ds, "seed": seed, "threat": threat, "mode": mode,
                                          "metric": k, "mean": float(np.nanmean(v)),
                                          "mean_evaded": float(np.nanmean(v[succ])) if succ.any() else np.nan})
            log(f"multiclass {ds} seed {seed} done")
            pd.DataFrame(met_rows).to_csv(out / "multiclass_metrics.csv", index=False)
            pd.DataFrame(cls_rows).to_csv(out / "multiclass_per_class_f1.csv", index=False)
            pd.DataFrame(att_rows).to_csv(out / "multiclass_attacks.csv", index=False)
            pd.DataFrame(stab_rows).to_csv(out / "multiclass_stability.csv", index=False)
    for nm, rr in (("multiclass_metrics", met_rows), ("multiclass_per_class_f1", cls_rows),
                   ("multiclass_attacks", att_rows), ("multiclass_stability", stab_rows)):
        print_block("extras", nm, pd.DataFrame(rr))


# ------------------------------------------------------------ (3) profiling
class PeakRSS:
    def __init__(self, interval=0.005):
        import psutil
        self.p = psutil.Process(os.getpid()); self.interval = interval
    def __enter__(self):
        self.start = self.p.memory_info().rss; self.peak = self.start; self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self
    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.p.memory_info().rss); time.sleep(self.interval)
    def __exit__(self, *a):
        self._run = False; self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)
        self.delta_mb = (self.peak - self.start) / 2 ** 20
        self.peak_mb = self.peak / 2 ** 20
        self.start_mb = self.start / 2 ** 20


def profiling(cfg, log):
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    gpu = torch.cuda.is_available()
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        seed = cfg.seeds[0]
        dev0 = resolve_device(cfg.device)
        dets, _, _ = build_models(z, cfg, seed, dev0, ["mlp_clean", "xgb"], log)
        X = z["X_test"][z["explain_mal_idx"][:cfg.profile_rows]]
        bg = z["X_train"][z["shap_bg_idx"]]; lbg = z["X_train"][z["lime_bg_idx"]]
        cat_cols = [j for j, k in enumerate(prep.col_kind) if k == "cat"]
        settings = [("cpu", 1), ("cpu", os.cpu_count() or 1)] + ([("cuda", None)] if gpu else [])
        for devname, threads in settings:
            if threads:
                torch.set_num_threads(threads)
            dev = torch.device(devname)
            for mname, det in dets.items():
                if det.kind == "mlp":
                    det.obj.to(dev); det.device = dev
                elif devname == "cuda":
                    continue  # tree models profiled on CPU only
                else:
                    det.obj.set_params(n_jobs=threads)
                # inference latency
                x1 = X[:1]
                for _ in range(20):
                    det.predict(x1)
                t0 = time.perf_counter()
                for _ in range(500):
                    det.predict(x1)
                lat1 = (time.perf_counter() - t0) / 500
                Xb = np.repeat(X, 11, axis=0)[:1024]
                t0 = time.perf_counter(); det.predict(Xb); latb = time.perf_counter() - t0
                rows.append({"dataset": ds, "device": devname, "threads": threads, "model": mname, "task": "inference",
                             "ms_per_record": 1000 * lat1, "batch1024_ms_per_record": 1000 * latb / 1024})
                tasks = [("SHAP", lambda: E.shap_values(det, X, bg)),
                         ("LIME", lambda: E.lime_values(det, X, lbg, cat_cols, cfg.lime_num_samples, seed)),
                         ("BB_attack_c", lambda: A.blackbox(det, X, max(cfg.epsilons), prep, "constrained",
                                                           cfg.bb_queries, seed, det.threshold))]
                if det.kind == "mlp":
                    tasks.append(("PGD_attack_c", lambda: A.pgd(det, X, max(cfg.epsilons), prep, "constrained",
                                                               cfg.pgd_steps, seed)))
                if det.kind == "mlp":
                    model_kb = sum(p.numel() * p.element_size() for p in det.obj.parameters()) / 1024
                else:
                    model_kb = len(det.obj.get_booster().save_raw("json")) / 1024
                for tname, fn in tasks:
                    import gc, tracemalloc
                    gc.collect()
                    if gpu:
                        torch.cuda.reset_peak_memory_stats()
                    tracemalloc.start()
                    with PeakRSS() as pr:
                        t0 = time.perf_counter(); fn(); secs = time.perf_counter() - t0
                    tm_peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
                    tracemalloc.stop()
                    rows.append({"dataset": ds, "device": devname, "threads": threads, "model": mname, "task": tname,
                                 "n": len(X), "seconds": secs, "ms_per_record": 1000 * secs / len(X),
                                 "peak_rss_delta_mb": pr.delta_mb, "process_rss_before_mb": pr.start_mb,
                                 "process_rss_peak_mb": pr.peak_mb, "py_alloc_peak_mb": tm_peak,
                                 "model_size_kb": model_kb,
                                 "gpu_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if (gpu and devname == "cuda") else np.nan})
                    log(f"profile {ds} {devname}/{threads} {mname} {tname}: {1000*secs/len(X):.1f} ms/rec")
                    pd.DataFrame(rows).to_csv(out / "profiling.csv", index=False)
        torch.set_num_threads(os.cpu_count() or 1)
    print_block("extras", "profiling", pd.DataFrame(rows))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ (4) baseline ablation
def baseline_ablation(cfg, log, combos=None, out_name="baseline_ablation"):
    """Isolates the contributors to the v1 Edge-IIoTset baseline deficit: feature
    policy (numeric-only vs. numeric+categorical), de-duplication, and threshold rule."""
    import dataclasses
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rows = []
    combos = combos or [(p, d, "logminmax") for p in ["numeric_only", "v2"] for d in [False, True]]
    for policy, dedup, scaling in combos:
        if True:
            c2 = dataclasses.replace(cfg, feature_policy=policy, dedup=dedup, scaling=scaling)
            for ds in cfg.datasets:
                z, prep = prepare(c2, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
                for seed in cfg.seeds[:cfg.at_sweep_seeds]:
                    dets, _, _ = build_models(z, c2, seed, device, ["mlp_clean", "xgb"], log)
                    for name, det in dets.items():
                        pt = det.predict(z["X_test"])
                        for rule, thr in (("fixed_0.5", 0.5), ("val_tuned", det.threshold)):
                            m = binary_metrics(z["y_test"], pt, thr)
                            rows.append({"dataset": ds, "feature_policy": policy, "dedup": dedup, "scaling": scaling, "seed": seed,
                                         "model": name, "threshold_rule": rule, "n_columns": z["X_train"].shape[1],
                                         "n_train": len(z["y_train"]), **m})
                    log(f"ablation {ds} {policy} dedup={dedup} {scaling} seed {seed} done")
                    pd.DataFrame(rows).to_csv(out / f"{out_name}.csv", index=False)
    print_block("extras", out_name, pd.DataFrame(rows))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ (5) operating point
def operating_point(cfg, log):
    """Attack success as a function of the decision threshold. Thresholds are matched
    on the validation benign records to a target false-positive rate, so detectors are
    compared at equal false-alarm cost.  PGD minimises the logit independently of the
    threshold, so one adversarial set is scored at every threshold; the black-box attack
    early-stops at the threshold and is re-run per threshold."""
    out = Path(cfg.out_dir) / "extras"; out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rows = []
    targets = [0.001, 0.005, 0.01, 0.02, 0.05]
    settings = [("unconstrained", max(cfg.epsilons)), ("numeric", max(cfg.epsilons)),
                ("constrained", max(cfg.epsilons)), ("constrained", 1.0)]
    for ds in cfg.datasets:
        z, prep = prepare(cfg, ds, 0, Path(cfg.work_dir), Path(cfg.out_dir), log)
        idx = z["attack_idx"]; X, y = z["X_test"][idx], z["y_test"][idx]
        ben_val = z["X_val"][z["y_val"] == 0]
        for seed in cfg.seeds[:cfg.at_sweep_seeds]:
            dets, _, _ = build_models(z, cfg, seed, device, ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"], log)
            for name, det in dets.items():
                pv_ben = det.predict(ben_val)
                pt = det.predict(z["X_test"])
                thr = {"val_tuned": det.threshold, "fixed_0.5": 0.5}
                for t in targets:
                    thr[f"fpr_{t:g}"] = float(np.quantile(pv_ben, 1 - t))
                for threat, eps in settings:
                    if det.kind == "mlp":
                        Xa = A.pgd(det, X, eps, prep, threat, steps=cfg.pgd_steps, seed=seed)
                    for tname, tv in thr.items():
                        if det.kind != "mlp":
                            Xa, _ = A.blackbox(det, X, eps, prep, threat, cfg.bb_queries, seed=seed, thr=tv)
                        s, _, _ = A.attack_summary(det, X, y, Xa, tv, prep)
                        m = binary_metrics(z["y_test"], pt, tv)
                        rows.append({"dataset": ds, "seed": seed, "model": name, "threat": threat, "eps": eps,
                                     "threshold_rule": tname, "threshold": tv, "test_fpr": m["fpr"],
                                     "test_mal_recall": m["mal_recall"], "test_macro_f1": m["macro_f1"],
                                     "eligible": s["eligible"], "asr": s["asr"], "mal_recall_adv": s["mal_recall_adv"]})
            log(f"operating point {ds} seed {seed} done")
            pd.DataFrame(rows).to_csv(out / "operating_point.csv", index=False)
    print_block("extras", "operating_point", pd.DataFrame(rows))
    return pd.DataFrame(rows)


def v1_path(cfg, log):
    """Reproduces the original (v1) pipeline and changes one factor at a time:
    P0 v1 features + v1 quantile scaling, no de-duplication (= original submission)
    P1 v1 features + log/min-max scaling
    (P2-P4, i.e. v2 numeric-only / +categorical / +de-duplication, are in baseline_ablation)."""
    return baseline_ablation(cfg, log, combos=[("v1", False, "quantile_v1"), ("v1", False, "logminmax")],
                             out_name="baseline_v1_path")
