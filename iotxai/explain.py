"""SHAP / LIME attributions, group-level aggregation, faithfulness under three
masking references with a random-ranking control, and stability metrics."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata


# ------------------------------------------------------------------ attributions
class _ProbWrap(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base
    def forward(self, x):
        return torch.sigmoid(self.base(x)).reshape(-1, 1)


def shap_values(det, X, background):
    """Signed attributions toward P(malicious), shape (n, d)."""
    import shap
    X = np.asarray(X, np.float32)
    if det.kind == "mlp":
        wrapped = _ProbWrap(det.obj).to(det.device).eval()
        bg = torch.as_tensor(np.asarray(background, np.float32), device=det.device)
        xx = torch.as_tensor(X, device=det.device)
        try:
            vals = shap.DeepExplainer(wrapped, bg).shap_values(xx, check_additivity=False)
        except Exception as exc:  # pragma: no cover
            print("DeepExplainer fallback to GradientExplainer:", exc)
            vals = shap.GradientExplainer(wrapped, bg).shap_values(xx)
    else:
        ex = shap.TreeExplainer(det.obj, data=np.asarray(background, np.float32),
                                model_output="probability", feature_perturbation="interventional")
        vals = ex.shap_values(X, check_additivity=False)
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[..., -1]
    return vals.astype(np.float32)


def lime_values(det, X, background, cat_cols, num_samples, seed):
    from lime.lime_tabular import LimeTabularExplainer
    X = np.asarray(X, np.float32)
    d = X.shape[1]
    ex = LimeTabularExplainer(np.asarray(background, np.float32), mode="classification",
                              categorical_features=list(cat_cols), discretize_continuous=False,
                              sample_around_instance=True, random_state=int(seed))
    def fn(a):
        p = det.predict(np.asarray(a, np.float32))
        return np.column_stack([1 - p, p])
    out = np.zeros((len(X), d), np.float32)
    for i in range(len(X)):
        e = ex.explain_instance(X[i], fn, labels=(1,), num_features=d, num_samples=num_samples)
        for j, w in e.local_exp[1]:
            out[i, j] = w
    return out


def to_groups(A, groups):
    return np.stack([A[:, g["cols"]].sum(1) for g in groups], axis=1).astype(np.float32)


# ------------------------------------------------------------------ faithfulness
class Masker:
    """Builds masked inputs for a set of group masks under three references."""

    def __init__(self, groups, d, bg, pool, n_marginal, n_knn, seed, device="cpu"):
        self.groups, self.d = groups, d
        self.G = len(groups)
        self.colmap = np.zeros((self.G, d), np.float32)
        for gi, g in enumerate(groups):
            self.colmap[gi, g["cols"]] = 1
        ref = np.median(bg, axis=0)
        for g in groups:      # categorical blocks: modal level pattern, not column medians
            if g["kind"] == "cat":
                B = bg[:, g["cols"]].round().astype(int)
                pats, cnt = np.unique(B, axis=0, return_counts=True)
                ref[g["cols"]] = pats[cnt.argmax()]
        self.median_ref = ref.astype(np.float32)
        self.bg = np.asarray(bg, np.float32)
        self.pool = np.asarray(pool, np.float32)
        self.n_marginal, self.n_knn = n_marginal, n_knn
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.P = torch.as_tensor(self.pool, device=device)
        self.P2 = (self.P ** 2)

    def build(self, X, gmask, ref):
        """X (m,d), gmask (m,G) bool masked-groups -> (m*r, d) inputs, r replicates."""
        cm = (gmask.astype(np.float32) @ self.colmap) > 0  # (m,d)
        if ref == "median":
            return np.where(cm, self.median_ref[None, :], X).astype(np.float32), 1
        if ref == "marginal":
            r = self.n_marginal
            rows = self.rng.integers(0, len(self.bg), size=(len(X), r))
            R = self.bg[rows]                                   # (m,r,d)
            out = np.where(cm[:, None, :], R, X[:, None, :])
            return out.reshape(-1, self.d).astype(np.float32), r
        if ref == "knn":
            r = self.n_knn
            keep = torch.as_tensor((~cm).astype(np.float32), device=self.device)
            Xt = torch.as_tensor(X, device=self.device)
            res = []
            for i in range(0, len(X), 1024):
                k_, x_ = keep[i:i + 1024], Xt[i:i + 1024]
                dist = (k_ * x_ ** 2).sum(1, keepdim=True) - 2 * (k_ * x_) @ self.P.T + k_ @ self.P2.T
                nn_idx = torch.topk(dist, r, largest=False).indices.cpu().numpy()
                res.append(nn_idx)
            nn_idx = np.concatenate(res)
            R = self.pool[nn_idx]
            out = np.where(cm[:, None, :], R, X[:, None, :])
            return out.reshape(-1, self.d).astype(np.float32), r
        raise ValueError(ref)


def _topk_masks(order, k, G):
    m = np.zeros((len(order), G), bool)
    np.put_along_axis(m, order[:, :k], True, axis=1)
    return m


def faithfulness(det, X, evid, masker, k_fracs, refs, cls):
    """evid: (n,G) attribution in favour of the evaluated class ``cls`` (n,)
    (larger = more supporting).  Class score f_c = P(mal) if c==1 else 1-P(mal).
    Returns dict ref -> (comp_aopc[n], suff_aopc[n], comp[n,K], suff[n,K]);
    AOPC = mean over the k fractions.
    comprehensiveness_k = f_c(x) - E f_c(x with top-k groups replaced)   (higher = better)
    sufficiency_k      = f_c(x) - E f_c(x with all but top-k replaced) (lower = better)"""
    G = evid.shape[1]
    cls = np.asarray(cls).astype(int)
    def fc(p, c):
        return np.where(c == 1, p, 1.0 - p)
    base = fc(det.predict(X), cls)
    order = np.argsort(-evid, axis=1, kind="stable")
    out = {}
    for ref in refs:
        comp, suff = np.zeros((len(X), len(k_fracs))), np.zeros((len(X), len(k_fracs)))
        for ki, fr in enumerate(k_fracs):
            k = int(min(G - 1, max(1, round(fr * G))))
            top = _topk_masks(order, k, G)
            for which, M in (("comp", top), ("suff", ~top)):
                Xm, r = masker.build(X, M, ref)
                p = det.predict(Xm).reshape(len(X), r)
                p = fc(p, cls[:, None]).mean(1)
                (comp if which == "comp" else suff)[:, ki] = base - p
        out[ref] = (comp.mean(1), suff.mean(1), comp, suff)
    return out


# ------------------------------------------------------------------ stability
def _rank_rows(A):
    return np.apply_along_axis(rankdata, 1, A)


def stability(A, B, ks):
    """Row-wise comparisons of group attributions A,B (n,G), signed toward malicious."""
    absA, absB = np.abs(A), np.abs(B)
    res = {}
    oa, ob = np.argsort(-absA, 1, kind="stable"), np.argsort(-absB, 1, kind="stable")
    G = A.shape[1]
    for k in ks:
        k = min(k, G)
        ja = np.zeros(len(A))
        for i in range(len(A)):
            sa, sb = set(oa[i, :k]), set(ob[i, :k])
            ja[i] = len(sa & sb) / len(sa | sb)
        res[f"jaccard@{k}"] = ja
    ra, rb = _rank_rows(absA), _rank_rows(absB)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    res["spearman"] = np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    res["signed_cosine"] = np.where((na > 0) & (nb > 0), (A * B).sum(1) / np.maximum(na * nb, 1e-12), np.nan)
    res["top1_agree"] = (oa[:, 0] == ob[:, 0]).astype(float)
    return res


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
