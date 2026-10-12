"""Detectors (MLP family + gradient-boosted trees), training conditions,
validation-based threshold selection, calibration and predictive metrics."""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)


def resolve_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout, n_out=1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        self.n_out = n_out

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.n_out == 1 else out


class Detector:
    """Uniform wrapper. ``predict(X)`` returns P(malicious) as float64 numpy."""

    def __init__(self, kind, obj, device, temperature=1.0):
        self.kind, self.obj, self.device = kind, obj, device
        self.temperature = temperature
        self.threshold = 0.5

    def logits(self, X, batch=65536):
        assert self.kind == "mlp"
        self.obj.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(np.asarray(X[i:i + batch], np.float32), device=self.device)
                out.append(self.obj(xb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, X):
        X = np.asarray(X, np.float32)
        if len(X) == 0:
            return np.zeros(0)
        if self.kind == "mlp":
            return 1.0 / (1.0 + np.exp(-self.logits(X).astype(np.float64)))
        return self.obj.predict_proba(X)[:, 1].astype(np.float64)

    def predict_calibrated(self, X):
        if self.kind != "mlp":
            return self.predict(X)
        return 1.0 / (1.0 + np.exp(-self.logits(X).astype(np.float64) / self.temperature))


# ----------------------------------------------------------------- MLP training
def _perturb_unconstrained(model, xb, yb, wfn, eps, steps):
    """Loss-ascent perturbation for adversarial training (all features, l_inf, [0,1])."""
    model.eval()
    if steps <= 1:
        x = xb.clone().requires_grad_(True)
        loss = wfn(model(x), yb)
        g = torch.autograd.grad(loss, x)[0]
        adv = (xb + eps * g.sign()).clamp(0, 1)
    else:
        alpha = 2.5 * eps / steps
        adv = (xb + torch.empty_like(xb).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = wfn(model(adv), yb)
            g = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * g.sign()
            adv = torch.max(torch.min(adv, xb + eps), xb - eps).clamp(0, 1)
    model.train()
    return adv.detach()


def train_mlp(X_tr, y_tr, X_va, y_va, seed, cfg, device, mode="clean", eps=0.0, lam=1.0,
              steps=1, n_classes=1, log=print):
    """mode: clean | at.  For AT the minibatch loss is lam*L(adv) + (1-lam)*L(clean)."""
    set_seed(seed)
    d = X_tr.shape[1]
    model = MLP(d, cfg.hidden_dims, cfg.dropout, n_out=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if n_classes == 1:
        pos = float(np.mean(y_tr)); w = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=device)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        def wfn(logit, yb):
            return (bce(logit, yb.float()) * w[yb.long()]).mean()
    else:
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        cw = torch.tensor(len(y_tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
        ce = nn.CrossEntropyLoss(weight=cw)
        def wfn(logit, yb):
            return ce(logit, yb.long())
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            if mode == "at":
                adv = _perturb_unconstrained(model, xb, yb, wfn, eps, steps)
                loss = lam * wfn(model(adv), yb)
                if lam < 1.0:
                    loss = loss + (1 - lam) * wfn(model(xb), yb)
            else:
                loss = wfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        det = Detector("mlp", model, device)
        if n_classes == 1:
            pv = det.predict(X_va)
            score = f1_score(y_va, (pv >= 0.5).astype(int), average="macro", zero_division=0)
        else:
            pv = det.logits(X_va).argmax(1)
            score = f1_score(y_va, pv, average="macro", zero_division=0)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(perm), "val_macro_f1": score})
        if score > best + 1e-5:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, hist, time.time() - t0


def fit_temperature(det, X_va, y_va):
    """Temperature scaling on validation logits (does not change ranking)."""
    z = torch.as_tensor(det.logits(X_va), dtype=torch.float64)
    y = torch.as_tensor(y_va, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = bce(z / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.detach().exp())


def train_xgb(X_tr, y_tr, X_va, y_va, seed, cfg, device, n_classes=1):
    import xgboost as xgb
    params = dict(n_estimators=cfg.xgb_n_estimators, max_depth=cfg.xgb_max_depth,
                  learning_rate=cfg.xgb_lr, subsample=cfg.xgb_subsample,
                  colsample_bytree=cfg.xgb_colsample, tree_method="hist", random_state=seed,
                  early_stopping_rounds=30, n_jobs=-1,
                  device="cuda" if device.type == "cuda" else "cpu")
    if n_classes == 1:
        pos = float(np.mean(y_tr))
        params.update(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=(1 - pos) / pos)
    else:
        params.update(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)
    t0 = time.time()
    try:
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except Exception as exc:  # older xgboost without the `device` argument / no GPU build
        print("xgboost GPU/device fallback:", exc)
        params.pop("device", None)
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    try:  # predictions/attacks/TreeSHAP on CPU for portability
        clf.set_params(device="cpu")
    except Exception:
        pass
    return clf, time.time() - t0


# ----------------------------------------------------------------- thresholds & metrics
def tune_threshold(y, p):
    """Validation threshold maximising macro-F1 (searched over <=2000 quantiles)."""
    qs = np.unique(np.quantile(p, np.linspace(0, 1, 2001)))
    best_t, best_f = 0.5, -1
    for t in qs:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def ece_score(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": thr, "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "mal_precision": precision_score(y, pred, zero_division=0),
            "mal_recall": recall_score(y, pred, zero_division=0),
            "mal_f1": f1_score(y, pred, zero_division=0),
            "ben_precision": tn / max(tn + fn, 1), "ben_recall": tn / max(tn + fp, 1),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece_score(y, p)}


def family_recall(y, fam, p, thr):
    pred = (p >= thr).astype(int)
    rows = []
    for f in np.unique(fam):
        m = fam == f
        target = 1 if y[m].mean() > 0.5 else 0
        rows.append({"family": f, "n": int(m.sum()), "correct_rate": float(np.mean(pred[m] == target))})
    return rows
