import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iotxai.config import Config
from iotxai.runner import run_all, Logger
from iotxai import extras
SYN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth"; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smoke_out"
cfg = Config(out_dir=OUT + "/results_v2", work_dir=OUT + "/cache", input_root=SYN,
             seeds=[17, 29], max_epochs=3, xgb_n_estimators=40, explain_benign_rows=40, explain_malicious_rows=40,
             attack_rows=120, lime_num_samples=200, bb_queries=40, repeat_seeds=1, repeat_rows=10, lime_repeats=2,
             shap_bg_repeats=2, knn_pool_rows=500, lime_background_rows=500, at_sweep_seeds=1, at_sweep_eps=[0.03],
             at_sweep_lambda=[0.5], multiclass_seeds=1, profile_rows=10)
t=time.time()
run_all(cfg)
print("main", time.time()-t)
log = Logger(OUT + "/results_v2/log.txt")
extras.at_sweep(cfg, log); extras.multiclass(cfg, log); extras.profiling(cfg, log)
print("total", time.time()-t)
