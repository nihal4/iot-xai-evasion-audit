"""Command-line entry point: python -m iotxai.cli --session A --input-root ./data --out ./results"""
import argparse
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config, session_config
from .runner import Logger, run_all
from . import extras
from . import round2


def run_session(session, out, work, input_root="/kaggle/input", ton_csv="", edge_csv="", device="auto"):
    cfg = session_config(session, out_dir=str(out), work_dir=str(work), input_root=str(input_root),
                         ton_csv=ton_csv, edge_csv=edge_csv, device=device)
    log = Logger(Path(out) / f"log_session_{session}.txt")
    t0 = time.time()
    run_all(cfg, log)
    if cfg.run_at_sweep:
        extras.at_sweep(Config(**{**asdict(cfg), "split_ids": [0]}), log)
    if cfg.run_multiclass:
        extras.multiclass(cfg, log)
    if cfg.run_baseline_ablation:
        extras.baseline_ablation(cfg, log)
    if cfg.run_v1_path:
        extras.v1_path(cfg, log)
    if cfg.run_operating_point:
        extras.operating_point(cfg, log)
    if cfg.run_profiling:
        extras.profiling(cfg, log)
    if cfg.run_r2_robustness:
        round2.robustness_r2(cfg, log)
    if cfg.run_r2_fairness:
        round2.explainer_fairness(cfg, log)
    if cfg.data_variants:
        from .runner import run_one
        from .report import print_block
        import json as _json
        for ds in cfg.datasets:
            for sid in cfg.data_variants:
                for seed in cfg.seeds:
                    try:
                        run_one(cfg, ds, sid, seed, log, explain=cfg.explain, repeats=False)
                    except Exception as exc:
                        import traceback
                        log(f"VARIANT FAILED {ds} split{sid} seed{seed}: {exc}\n" + traceback.format_exc())
                        break
                for f in sorted((Path(out) / "data_audit").glob(f"*{ds}*")):
                    if f.suffix == ".json":
                        print(f"### BEGIN data_audit {f.name}\n{f.read_text()}\n### END data_audit {f.name}", flush=True)
    log(f"SESSION {session} finished in {(time.time() - t0) / 3600:.2f} h")
    from .report import cross_seed_summary
    try:
        cross_seed_summary(out)
    except Exception as exc:
        log(f"cross-seed summary failed: {exc}")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=["TEST", "A", "B", "C1", "C2", "C3", "C4", "R0", "R1", "R2", "R3", "R4"])
    ap.add_argument("--input-root", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--work", default="./cache")
    ap.add_argument("--ton-csv", default="")
    ap.add_argument("--edge-csv", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    run_session(a.session, a.out, a.work, a.input_root, a.ton_csv, a.edge_csv, a.device)
