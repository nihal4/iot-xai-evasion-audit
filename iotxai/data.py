"""Data loading, auditing, de-duplication, partitioning and preprocessing.

Design rules
------------
* Raw-file SHA-256, row counts, duplicate counts and label conflicts are logged.
* De-duplication is performed on the model-visible feature vector + label BEFORE
  partitioning, so no identical record can appear in two partitions; feature
  vectors that occur with both labels are removed as irreducibly ambiguous.
* Everything learned from data (categorical levels, constant-column removal,
  scaling ranges, relation support) is fitted on the training partition only.
* Partition membership is saved as raw-file row numbers so the exact splits can
  be rebuilt from the public raw CSV plus this code.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_SPECS, V1_FEATURES

MISSING_TOKENS = {"-", "", "nan", "none", "null", "?"}
MAX_LEVELS = 20


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover_file(cfg, name):
    explicit = cfg.ton_csv if name == "ton" else cfg.edge_csv
    if explicit:
        return Path(explicit)
    pats = DATASET_SPECS[name]["file_patterns"]
    cands = []
    for p in Path(cfg.input_root).rglob("*.csv"):
        low = p.name.lower()
        for rank, pat in enumerate(pats):
            if pat in low:
                cands.append((rank, p.stat().st_size, p))
    if not cands:
        raise FileNotFoundError(f"No CSV for {name} under {cfg.input_root}; set cfg.{name}_csv")
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands[0][2]


def _norm_str(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.where(~out.str.lower().isin(MISSING_TOKENS), "__missing__")
    return out


def _canon_cat(s: pd.Series) -> pd.Series:
    """Categorical strings; numeric-looking tokens are canonicalised so that
    e.g. '0' and '0.0' (a known Edge-IIoTset export quirk) are one level."""
    st = _norm_str(s)
    num = pd.to_numeric(st, errors="coerce")
    canon = num.map(lambda v: f"{v:g}" if pd.notna(v) else None)
    return canon.where(num.notna(), st)


def load_and_audit(cfg, name, log):
    spec = DATASET_SPECS[name]
    path = discover_file(cfg, name)
    log(f"[{name}] reading {path}")
    drop = set(spec["drop"])
    keep_only = None
    if cfg.feature_policy == "v1":   # exact v1 feature list (ablation only)
        keep_only = set(V1_FEATURES[name]) | {spec["label"], spec["family"]}
        drop = set()
    if keep_only is not None:
        raw = pd.read_csv(path, usecols=lambda c: c.strip() in keep_only, low_memory=False)
    else:
        raw = None
    try:
        if raw is not None:
            raise StopIteration
        raw = pd.read_csv(path, usecols=lambda c: c.strip() not in drop, low_memory=False)
    except StopIteration:
        pass
    except Exception as exc:  # malformed quoting in payload-like columns
        log(f"[{name}] C parser failed ({exc}); retrying with python engine")
        raw = pd.read_csv(path, usecols=lambda c: c.strip() not in drop, engine="python",
                          on_bad_lines="skip")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.reset_index(drop=True)
    raw["__row__"] = np.arange(len(raw))
    audit = {"dataset": name, "source_path": str(path), "sha256": sha256_file(path),
             "file_bytes": Path(path).stat().st_size, "rows_raw": int(len(raw)),
             "columns_loaded": [c for c in raw.columns if c != "__row__"],
             "columns_dropped_by_policy": sorted(drop)}

    y_raw = pd.to_numeric(raw[spec["label"]], errors="coerce")
    fam = raw[spec["family"]].astype(str).str.strip().str.lower()
    bad = y_raw.isna() | ~y_raw.isin([0, 1])
    audit["rows_missing_or_invalid_label"] = int(bad.sum())
    raw, y_raw, fam = raw[~bad], y_raw[~bad].astype(int), fam[~bad]

    # ---- classify columns
    feats_num, feats_cat, dropped_auto = [], [], {}
    for c in raw.columns:
        if c in (spec["label"], spec["family"], "__row__"):
            continue
        if c in spec["categorical"]:
            if cfg.feature_policy == "numeric_only":
                dropped_auto[c] = "categorical_excluded_by_numeric_only_ablation"
            else:
                feats_cat.append(c)
            continue
        s = raw[c]
        if pd.api.types.is_numeric_dtype(s):
            feats_num.append(c)
            continue
        st = _norm_str(s)
        nonmiss = st[st != "__missing__"]
        parsed = pd.to_numeric(nonmiss, errors="coerce")
        if len(nonmiss) == 0 or parsed.notna().mean() >= 0.95:
            feats_num.append(c)
        else:
            dropped_auto[c] = "non_numeric_not_in_categorical_policy"
    audit["dropped_auto"] = dropped_auto

    num = pd.DataFrame(index=raw.index)
    for c in feats_num:
        s = raw[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(_norm_str(s).replace("__missing__", np.nan), errors="coerce")
        num[c] = s.astype(float)
    audit["numeric_missing_counts"] = {c: int(num[c].isna().sum()) for c in feats_num if num[c].isna().any()}
    num = num.fillna(0.0)   # '-' / empty denote an absent protocol field
    cat = pd.DataFrame({c: _canon_cat(raw[c]) for c in feats_cat}, index=raw.index)

    # ---- duplicate audit and de-duplication on model-visible features + label
    key = pd.concat([num.round(9), cat], axis=1)
    audit["rows_exact_duplicate_loaded_columns"] = int(raw.drop(columns="__row__").duplicated().sum())
    hashes = pd.util.hash_pandas_object(key, index=False)
    frame = pd.DataFrame({"h": hashes.values, "y": y_raw.values}, index=raw.index)
    n_labels = frame.groupby("h")["y"].nunique()
    conflict_h = set(n_labels[n_labels > 1].index)
    is_conflict = frame["h"].isin(conflict_h)
    audit["rows_in_label_conflicting_groups"] = int(is_conflict.sum())
    keep = (~is_conflict) & (~frame.duplicated(subset=["h", "y"], keep="first"))
    if not cfg.dedup:   # ablation only: keep duplicates and conflicting records
        keep = pd.Series(True, index=frame.index)
    audit["rows_duplicate_feature_label"] = int((~is_conflict & frame.duplicated(subset=["h", "y"], keep="first")).sum())
    num, cat, y, fam = num[keep], cat[keep], y_raw[keep].values, fam[keep].values
    rows = raw.loc[keep, "__row__"].values
    audit["rows_after_dedup"] = int(len(rows))

    if cfg.max_rows_per_dataset and len(rows) > cfg.max_rows_per_dataset:
        rng = np.random.default_rng(cfg.split_seed_base)
        sel = []
        frac = cfg.max_rows_per_dataset / len(rows)
        for f in np.unique(fam):
            idx = np.where(fam == f)[0]
            k = max(1, int(round(len(idx) * frac)))
            sel.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        sel = np.sort(np.concatenate(sel))
        num, cat, y, fam, rows = num.iloc[sel], cat.iloc[sel], y[sel], fam[sel], rows[sel]
        audit["rows_after_cap"] = int(len(rows))

    fam_counts = pd.Series(fam).value_counts()
    audit["family_counts_after_dedup"] = {k: int(v) for k, v in fam_counts.items()}
    audit["positive_rate_after_dedup"] = float(np.mean(y))
    data = {"num": num.reset_index(drop=True), "cat": cat.reset_index(drop=True),
            "y": np.asarray(y, int), "family": np.asarray(fam, object), "row": np.asarray(rows, int)}
    return data, audit


def make_split(data, split_id, cfg):
    seed = cfg.split_seed_base + split_id
    fam = data["family"]
    strat = fam.copy()
    vc = pd.Series(fam).value_counts()
    rare = set(vc[vc < 10].index)
    strat = np.array([("rare_" + str(yy)) if f in rare else f for f, yy in zip(fam, data["y"])], dtype=object)
    idx = np.arange(len(fam))
    hold = cfg.val_frac + cfg.test_frac
    tr, rest = train_test_split(idx, test_size=hold, random_state=seed, stratify=strat)
    va, te = train_test_split(rest, test_size=cfg.test_frac / hold, random_state=seed,
                              stratify=strat[rest])
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te), "seed": seed}


class Preprocessor:
    """Train-fitted transform: numeric -> log1p (if non-negative) -> min-max to [0,1];
    categorical -> one-hot over the <=20 most frequent training levels (+__other__)."""

    def fit(self, data, train_idx, spec, cfg, log):
        num = data["num"].iloc[train_idx]
        self.num_cols, self.params, self.const_dropped = [], {}, []
        for c in num.columns:
            v = num[c].values
            if np.nanmax(v) == np.nanmin(v):
                self.const_dropped.append(c)
                continue
            if getattr(cfg, "scaling", "logminmax") == "quantile_v1":
                # v1 scaling: train 1st/99th percentiles, linear, clipped; zero span -> removed
                uselog = False
                lo, hi = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
                if hi <= lo:
                    self.const_dropped.append(c)
                    continue
            else:
                uselog = bool(np.nanmin(v) >= 0)
                t = np.log1p(v) if uselog else v
                lo, hi = float(np.min(t)), float(np.max(t))
            self.params[c] = {"log": uselog, "lo": lo, "hi": hi,
                              "raw_min": float(np.min(v)), "raw_max": float(np.max(v)),
                              "integer": bool(c in spec["integer"] or np.all(np.mod(v, 1) == 0))}
            self.num_cols.append(c)
        self.cat_levels = {}
        cat = data["cat"].iloc[train_idx]
        for c in cat.columns:
            vc = cat[c].value_counts()
            if len(vc) <= 1:
                self.const_dropped.append(c)
                continue
            levels = list(vc.index[:MAX_LEVELS])
            if len(vc) > MAX_LEVELS:
                levels.append("__other__")
            self.cat_levels[c] = levels
        # column layout and groups
        self.columns, self.groups, self.col_kind = [], [], []
        for c in self.num_cols:
            self.groups.append({"name": c, "cols": [len(self.columns)], "kind": "num"})
            self.columns.append(c); self.col_kind.append("num")
        for c, levels in self.cat_levels.items():
            start = len(self.columns)
            for lv in levels:
                self.columns.append(f"{c}={lv}"); self.col_kind.append("cat")
            self.groups.append({"name": c, "cols": list(range(start, len(self.columns))), "kind": "cat"})
        # mutability
        self.mutable_dir = np.zeros(len(self.columns), dtype=np.int8)  # 0 immutable, 1 up, 2 both
        for c, d in spec["mutable"].items():
            if c in self.num_cols:
                self.mutable_dir[self.columns.index(c)] = 1 if d == "up" else 2
        # relations kept only when supported by the clean training data
        self.relations = []
        for rel in spec["relations"]:
            need = [rel["lhs"]] + list(rel["rhs"])
            if not all(n in data["num"].columns for n in need):
                continue
            lhs = num[rel["lhs"]].values
            rhs = sum(coef * num[f].values for f, coef in rel["rhs"].items())
            support = float(np.mean(lhs >= rhs - 1e-6))
            r = dict(rel, support=support, kept=support >= cfg.relation_min_support)
            self.relations.append(r)
            log(f"relation {rel['lhs']} >= {rel['rhs']}: support={support:.4f} kept={r['kept']}")
        return self

    def transform(self, data, idx):
        n = len(idx)
        Z = np.zeros((n, len(self.columns)), dtype=np.float32)
        num = data["num"].iloc[idx]
        for j, c in enumerate(self.num_cols):
            p = self.params[c]
            v = num[c].values.astype(float)
            t = np.log1p(np.maximum(v, 0)) if p["log"] else v
            Z[:, j] = np.clip((t - p["lo"]) / (p["hi"] - p["lo"]), 0, 1)
        cat = data["cat"].iloc[idx]
        for g in self.groups:
            if g["kind"] != "cat":
                continue
            c = g["name"]; levels = self.cat_levels[c]
            v = cat[c].values
            other = "__other__" in levels
            for k, lv in enumerate(levels):
                if lv == "__other__":
                    Z[:, g["cols"][k]] = ~np.isin(v, levels)
                else:
                    Z[:, g["cols"][k]] = (v == lv)
            if not other:  # unseen level -> all-zero block (documented)
                pass
        return Z

    # ---- raw-unit helpers for constraint repair/validity
    def to_raw(self, Z, j):
        c = self.columns[j]; p = self.params[c]
        t = Z[:, j] * (p["hi"] - p["lo"]) + p["lo"]
        return np.expm1(t) if p["log"] else t

    def from_raw(self, v, j):
        c = self.columns[j]; p = self.params[c]
        t = np.log1p(np.maximum(v, 0)) if p["log"] else v
        return np.clip((t - p["lo"]) / (p["hi"] - p["lo"]), 0, 1)

    def meta(self):
        return {"columns": self.columns, "groups": self.groups, "col_kind": self.col_kind,
                "num_params": self.params, "cat_levels": self.cat_levels,
                "constant_dropped": self.const_dropped,
                "mutable": {self.columns[j]: int(d) for j, d in enumerate(self.mutable_dir) if d},
                "relations": self.relations}


def stratified_pick(idx, fam, n, rng):
    """Pick n indices from idx proportionally to family frequency (>=1 per family)."""
    idx = np.asarray(idx)
    if n >= len(idx):
        return rng.permutation(idx)
    fams = pd.Series(fam[idx])
    vc = fams.value_counts()
    alloc = np.maximum(1, np.floor(vc / vc.sum() * n)).astype(int)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n:
        spare = (vc - alloc)[(vc - alloc) > 0]
        alloc[spare.idxmax()] += 1
    out = []
    for f, k in alloc.items():
        pool = idx[fams.values == f]
        out.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return rng.permutation(np.concatenate(out))


def prepare(cfg, name, split_id, work_dir: Path, out_dir: Path, log, data_cache={}):
    """Return a dict with arrays + preprocessor for (dataset, split). Cached on disk."""
    work_dir.mkdir(parents=True, exist_ok=True)
    sc = getattr(cfg, "scaling", "logminmax")
    tag = "" if (cfg.dedup and cfg.feature_policy == "v2" and sc == "logminmax") else \
        f"_{cfg.feature_policy}_dedup{int(cfg.dedup)}" + ("" if sc == "logminmax" else f"_{sc}")
    cache = work_dir / f"{name}_split{split_id}{tag}.npz"
    meta_path = work_dir / f"{name}_split{split_id}{tag}_prep.pkl"
    import pickle
    if cache.exists() and meta_path.exists():
        z = dict(np.load(cache, allow_pickle=True))
        prep = pickle.loads(meta_path.read_bytes())
        return z, prep
    ckey = (name, cfg.feature_policy, cfg.dedup)   # scaling is applied after loading
    if ckey not in data_cache:
        data_cache[ckey] = load_and_audit(cfg, name, log)
    data, audit = data_cache[ckey]
    spec = DATASET_SPECS[name]
    sp = make_split(data, split_id, cfg)
    prep = Preprocessor().fit(data, sp["train"], spec, cfg, log)
    out = {}
    for part in ["train", "val", "test"]:
        out[f"X_{part}"] = prep.transform(data, sp[part])
        out[f"y_{part}"] = data["y"][sp[part]]
        out[f"fam_{part}"] = data["family"][sp[part]].astype(str)
        out[f"row_{part}"] = data["row"][sp[part]]
    # cross-partition duplicate verification on the transformed feature vectors
    def hv(X):
        return set(pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values)
    htr, hva, hte = hv(out["X_train"]), hv(out["X_val"]), hv(out["X_test"])
    audit_split = {"split_id": split_id, "split_seed": sp["seed"],
                   "n_train": len(sp["train"]), "n_val": len(sp["val"]), "n_test": len(sp["test"]),
                   "train_test_shared_vectors": len(htr & hte), "train_val_shared_vectors": len(htr & hva),
                   "val_test_shared_vectors": len(hva & hte),
                   "n_model_columns": len(prep.columns), "n_feature_groups": len(prep.groups),
                   "train_positive_rate": float(out["y_train"].mean()),
                   "test_positive_rate": float(out["y_test"].mean())}
    # NB: shared *transformed* vectors can be >0 because clipping / top-20 level
    # bucketing maps distinct raw records to the same vector; raw duplicates are 0.
    rng = np.random.default_rng(sp["seed"])
    yt, ft = out["y_test"], out["fam_test"]
    mal = np.where(yt == 1)[0]; ben = np.where(yt == 0)[0]
    exp_mal = stratified_pick(mal, ft, cfg.explain_malicious_rows, rng)
    rest = np.setdiff1d(mal, exp_mal)
    extra = stratified_pick(rest, ft, max(0, cfg.attack_rows - len(exp_mal)), rng) if len(rest) else np.array([], int)
    out["attack_idx"] = np.concatenate([exp_mal, extra]).astype(int)        # first 300 = explained
    out["explain_benign_idx"] = rng.choice(ben, size=min(cfg.explain_benign_rows, len(ben)), replace=False)
    out["explain_mal_idx"] = exp_mal.astype(int)
    out["shap_bg_idx"] = rng.choice(len(out["y_train"]), size=cfg.shap_background_rows, replace=False)
    out["lime_bg_idx"] = rng.choice(len(out["y_train"]), size=min(cfg.lime_background_rows, len(out["y_train"])), replace=False)
    out["knn_pool_idx"] = rng.choice(len(out["y_train"]), size=min(cfg.knn_pool_rows, len(out["y_train"])), replace=False)
    np.savez_compressed(cache, **out)
    meta_path.write_bytes(pickle.dumps(prep))
    # public artefacts
    d = out_dir / ("data_audit" if not tag else f"data_audit_ablation{tag}"); d.mkdir(parents=True, exist_ok=True)
    (d / f"audit_{name}.json").write_text(json.dumps(audit, indent=2, default=str))
    (d / f"split_{name}_split{split_id}.json").write_text(json.dumps(audit_split, indent=2))
    (d / f"preprocessor_{name}_split{split_id}.json").write_text(json.dumps(prep.meta(), indent=2, default=str))
    parts = []
    for part in ["train", "val", "test"]:
        parts.append(pd.DataFrame({"raw_row": out[f"row_{part}"], "partition": part}))
    pd.concat(parts).sort_values("raw_row").to_csv(d / f"partition_{name}_split{split_id}.csv.gz", index=False)
    pd.DataFrame({"test_pos": out["attack_idx"], "raw_row": out["row_test"][out["attack_idx"]],
                  "explained": np.arange(len(out["attack_idx"])) < len(exp_mal)}).to_csv(
        d / f"attack_subset_{name}_split{split_id}.csv", index=False)
    pd.DataFrame({"test_pos": out["explain_benign_idx"], "raw_row": out["row_test"][out["explain_benign_idx"]]}).to_csv(
        d / f"explain_benign_{name}_split{split_id}.csv", index=False)
    log(f"[{name}] split {split_id}: {audit_split}")
    return out, prep
