"""Evasion attacks under two threat models and a domain-validity checker.

unconstrained : every model input may move by <= eps (l_inf) inside [0,1]
                (the v1 threat model, kept for continuity).
constrained   : only attacker-controllable flow features move (mutability mask),
                'up' features may only increase (padding / delay / extra packets),
                one-hot protocol blocks are frozen, and after the gradient search
                the record is repaired in raw units: integer rounding (ceil for
                increase-only features) and data-supported protocol relations
                (e.g. IP bytes >= payload bytes + 20 x packets).  Success is
                evaluated on the repaired record.
"""
from __future__ import annotations

import numpy as np
import torch


def masks(prep, threat, d):
    if threat == "unconstrained":
        return np.ones(d, bool), np.zeros(d, bool)
    if threat == "numeric":
        return np.array([k == "num" for k in prep.col_kind]), np.zeros(d, bool)
    mut = prep.mutable_dir > 0
    up = prep.mutable_dir == 1
    return mut, up


def _project(xa, x0, eps, mut, up):
    delta = torch.clamp(xa - x0, -eps, eps)
    delta = torch.where(up, delta.clamp(min=0), delta)
    delta = delta * mut
    return (x0 + delta).clamp(0, 1)


def pgd(det, X, eps, prep, threat, steps=10, seed=0, fgsm=False, target_logit_sign=1.0):
    """Minimise the malicious logit (binary) inside the feasible set."""
    model, device = det.obj, det.device
    model.eval()
    d = X.shape[1]
    mut_np, up_np = masks(prep, threat, d)
    mut = torch.as_tensor(mut_np, device=device, dtype=torch.float32)
    up = torch.as_tensor(up_np, device=device)
    out = []
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for i in range(0, len(X), 8192):
        x0 = torch.as_tensor(np.asarray(X[i:i + 8192], np.float32), device=device)
        if fgsm:
            xa = x0.clone().requires_grad_(True)
            grad = torch.autograd.grad(model(xa).sum(), xa)[0]
            xa = _project(x0 - eps * grad.sign(), x0, eps, mut, up)
        else:
            alpha = eps / max(1, steps) * 2.5
            noise = (torch.rand(x0.shape, generator=g) * 2 - 1).to(device) * eps
            xa = _project(x0 + noise, x0, eps, mut, up)
            for _ in range(steps):
                xa.requires_grad_(True)
                grad = torch.autograd.grad(model(xa).sum(), xa)[0]
                xa = _project(xa.detach() - alpha * grad.sign(), x0, eps, mut, up)
        out.append(xa.detach().cpu().numpy())
    Xa = np.concatenate(out).astype(np.float32)
    if threat == "constrained":
        Xa = repair(prep, Xa, np.asarray(X, np.float32))
    return Xa


def repair(prep, Xa, X0):
    """Raw-unit repair: integer rounding + data-supported relations. Only mutable
    columns are ever modified; if a relation cannot be restored by raising a
    mutable left-hand side, the offending mutable features are reverted."""
    Xa = Xa.copy()
    mut_cols = np.where(prep.mutable_dir > 0)[0]
    raw = {j: prep.to_raw(Xa, j) for j in mut_cols}
    raw0 = {j: prep.to_raw(X0, j) for j in mut_cols}
    for j in mut_cols:
        p = prep.params[prep.columns[j]]
        if p["integer"]:
            # floor keeps an increase-only change inside the eps budget
            raw[j] = np.floor(raw[j] + 1e-6) if prep.mutable_dir[j] == 1 else np.round(raw[j])
            raw[j] = np.minimum(raw[j], p["raw_max"])
        if prep.mutable_dir[j] == 1:
            raw[j] = np.maximum(raw[j], raw0[j])
    cols = prep.columns
    def val(name, Xsrc):
        j = cols.index(name)
        return raw[j] if j in raw else prep.to_raw(Xsrc, j)
    for rel in prep.relations:
        if not rel["kept"]:
            continue
        lhs = rel["lhs"]; jl = cols.index(lhs)
        rhs_val = sum(c * val(f, Xa) for f, c in rel["rhs"].items())
        rhs_val0 = sum(c * prep.to_raw(X0, cols.index(f)) for f, c in rel["rhs"].items())
        lhs0 = prep.to_raw(X0, jl)
        held0 = lhs0 >= rhs_val0 - 1e-6
        cur_l = val(lhs, Xa)
        viol = held0 & (cur_l < rhs_val - 1e-6)
        if not viol.any():
            continue
        if jl in raw:
            cap = prep.params[lhs]["raw_max"]
            new = np.where(viol, np.minimum(np.ceil(rhs_val), cap), raw[jl])
            raw[jl] = new
            still = viol & (raw[jl] < rhs_val - 1e-6)
        else:
            still = viol
        if still.any():   # revert the mutable rhs features for these rows
            for f in rel["rhs"]:
                jf = cols.index(f)
                if jf in raw:
                    raw[jf] = np.where(still, raw0[jf], raw[jf])
    for j in mut_cols:
        Xa[:, j] = prep.from_raw(raw[j], j)
    return Xa


def validity(prep, Xa, X0, tol=1e-4):
    """Per-record validity flags of an adversarial record w.r.t. the domain."""
    n, d = Xa.shape
    res = {}
    # one-hot integrity
    ok_cat = np.ones(n, bool)
    for g in prep.groups:
        if g["kind"] != "cat":
            continue
        B = Xa[:, g["cols"]]
        binary = np.all((np.abs(B) < tol) | (np.abs(B - 1) < tol), axis=1)
        ok_cat &= binary & (B.round().sum(1) <= 1)
    res["onehot_valid"] = ok_cat
    # integer features in raw units
    ok_int = np.ones(n, bool)
    for j, c in enumerate(prep.columns):
        if prep.col_kind[j] == "num" and prep.params[c]["integer"]:
            changed = np.abs(Xa[:, j] - X0[:, j]) > 1e-7
            v = prep.to_raw(Xa, j)
            ok_int &= ~changed | (np.abs(v - np.round(v)) <= 1e-3 * np.maximum(1, np.abs(v)))
    res["integer_valid"] = ok_int
    ok_rel = np.ones(n, bool)
    cols = prep.columns
    for rel in prep.relations:
        if not rel["kept"]:
            continue
        r = lambda Xs: (prep.to_raw(Xs, cols.index(rel["lhs"])),
                        sum(c * prep.to_raw(Xs, cols.index(f)) for f, c in rel["rhs"].items()))
        l0, r0 = r(X0); l1, r1 = r(Xa)
        tolr = 1e-3 * np.maximum(1, np.abs(r1))
        ok_rel &= ~(l0 >= r0 - 1e-6) | (l1 >= r1 - tolr)
    res["relation_valid"] = ok_rel
    changed = np.abs(Xa - X0) > 1e-6
    res["immutable_untouched"] = ~np.any(changed[:, prep.mutable_dir == 0], axis=1)
    up = prep.mutable_dir == 1
    res["direction_valid"] = ~np.any((Xa[:, up] < X0[:, up] - 1e-6), axis=1)
    res["syntactic_valid"] = res["onehot_valid"] & res["integer_valid"] & res["relation_valid"]
    res["threat_model_valid"] = res["syntactic_valid"] & res["immutable_untouched"] & res["direction_valid"]
    res["n_changed"] = changed.sum(1)
    return res


def blackbox(det, X, eps, prep, threat, queries=300, seed=0, thr=0.5):
    """Score-based l_inf random-search attack (Square-attack style) on vertices of
    the feasible box; works for any model exposing P(malicious)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    n, d = X.shape
    mut, up = masks(prep, threat, d)
    idx = np.where(mut)[0]
    if len(idx) == 0:
        return X.copy(), np.zeros(n, int)
    def vertex(shape_rows, cols):
        v = rng.choice([-1.0, 1.0], size=(shape_rows, len(cols))) * eps
        upc = up[cols]
        v[:, upc] = rng.choice([0.0, 1.0], size=(shape_rows, upc.sum())) * eps
        return v.astype(np.float32)
    def feasible(D):
        Xa = np.clip(X + D, 0, 1).astype(np.float32)
        return repair(prep, Xa, X) if threat == "constrained" else Xa
    D = np.zeros_like(X); D[:, idx] = vertex(n, idx)
    best_x = feasible(D); best = det.predict(best_x)
    used = np.ones(n, int)
    active = best >= thr
    for q in range(1, queries):
        if not active.any():
            break
        frac = max(1.0 / len(idx), 0.3 * (1 - q / queries))
        k = max(1, int(round(frac * len(idx))))
        rows = np.where(active)[0]
        Dn = D[rows].copy()
        m = len(rows)
        sel = np.argpartition(rng.random((m, len(idx))), k - 1, axis=1)[:, :k]
        cols = idx[sel]
        sign = rng.choice([-1.0, 1.0], size=(m, k))
        sign = np.where(up[cols], (sign > 0).astype(float), sign)
        Dn[np.arange(m)[:, None], cols] = (sign * eps).astype(np.float32)
        Xn = np.clip(X[rows] + Dn, 0, 1).astype(np.float32)
        if threat == "constrained":
            Xn = repair(prep, Xn, X[rows])
        s = det.predict(Xn)
        used[rows] += 1
        imp = s < best[rows]
        r_imp = rows[imp]
        D[r_imp] = Dn[imp]; best[r_imp] = s[imp]; best_x[r_imp] = Xn[imp]
        active = best >= thr
    return best_x, used


def attack_summary(det, X, y, Xa, thr, prep):
    p0, p1 = det.predict(X), det.predict(Xa)
    pred0, pred1 = (p0 >= thr).astype(int), (p1 >= thr).astype(int)
    elig = (y == 1) & (pred0 == 1)
    succ = elig & (pred1 == 0)
    v = validity(prep, Xa, X)
    linf = np.abs(Xa - X).max(1)
    return {"n_attack": int(len(y)), "eligible": int(elig.sum()), "success": int(succ.sum()),
            "asr": float(succ.sum() / max(elig.sum(), 1)),
            "mal_recall_clean": float(pred0[y == 1].mean()), "mal_recall_adv": float(pred1[y == 1].mean()),
            "mean_linf": float(linf[elig].mean()) if elig.any() else np.nan,
            "mean_features_changed": float(v["n_changed"][elig].mean()) if elig.any() else np.nan,
            "syntactic_valid_rate": float(v["syntactic_valid"][elig].mean()) if elig.any() else np.nan,
            "threat_model_valid_rate": float(v["threat_model_valid"][elig].mean()) if elig.any() else np.nan,
            "onehot_valid_rate": float(v["onehot_valid"][elig].mean()) if elig.any() else np.nan,
            "integer_valid_rate": float(v["integer_valid"][elig].mean()) if elig.any() else np.nan,
            "relation_valid_rate": float(v["relation_valid"][elig].mean()) if elig.any() else np.nan,
            "mean_p_drop": float((p0 - p1)[elig].mean()) if elig.any() else np.nan,
            "asr_syntactic_valid": float((succ & v["syntactic_valid"]).sum() / max(elig.sum(), 1)),
            "asr_threat_model_valid": float((succ & v["threat_model_valid"]).sum() / max(elig.sum(), 1))}, elig, succ
