"""Experiment configuration for the revised (v2) multi-seed XAI/adversarial IDS study.

Every value used by the pipeline is defined here and written to ``config.json``
at the start of every run so that a result bundle is self-describing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Original five seeds are kept as the first five so the v1 run is a subset.
SEEDS_20 = [17, 29, 41, 53, 67, 83, 97, 101, 113, 127,
            131, 149, 163, 179, 191, 211, 223, 239, 251, 269]


@dataclass
class Config:
    # ------------------------------------------------------------------ I/O
    out_dir: str = "/kaggle/working/results_v2"
    work_dir: str = "/kaggle/working/cache_v2"
    input_root: str = "/kaggle/input"
    # Explicit file paths (leave "" for automatic discovery under input_root).
    ton_csv: str = ""
    edge_csv: str = ""

    # ---------------------------------------------------------- design
    datasets: list = field(default_factory=lambda: ["ton", "edge"])
    seeds: list = field(default_factory=lambda: list(SEEDS_20))
    split_ids: list = field(default_factory=lambda: [0])     # 0 = primary partition
    split_seed_base: int = 2026                                # split s uses 2026 + s
    max_rows_per_dataset: int | None = None                    # None = all rows after dedup
    dedup: bool = True                                         # de-duplicate before partitioning
    feature_policy: str = "v2"                                 # "v2" | "numeric_only" | "v1" (ablation)
    scaling: str = "logminmax"                                 # "logminmax" | "quantile_v1" (ablation)
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ---------------------------------------------------------- models
    models: list = field(default_factory=lambda: ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"])
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 2048
    max_epochs: int = 30
    patience: int = 5
    # FGSM-AT reproduces the v1 control (eps=0.01, adversarial batch only).
    fgsmat_eps: float = 0.01
    fgsmat_lambda: float = 1.0
    # PGD-AT is the stronger defence requested by reviewers (mixed clean/adv loss).
    pgdat_eps: float = 0.03
    pgdat_steps: int = 5
    pgdat_lambda: float = 0.5
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8

    # ---------------------------------------------------------- attacks
    epsilons: list = field(default_factory=lambda: [0.01, 0.03, 0.05])
    pgd_steps: int = 10
    attack_rows: int = 2000            # malicious test records, stratified by family
    bb_queries: int = 300              # score-based black-box attack budget
    # unconstrained = every input column (v1-style, incl. one-hot columns);
    # numeric = all numeric fields, both directions, categorical blocks frozen (closest to the v1 feature space);
    # constrained = mutability mask + increase-only + integer/relation repair.
    threat_models: list = field(default_factory=lambda: ["unconstrained", "numeric", "constrained"])
    # padding budgets beyond the l_inf grid, constrained threat model only (1.0 = up to the training maximum)
    constrained_extra_eps: list = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    relation_min_support: float = 0.995  # data-derived invariants must hold in >=99.5% of train rows

    # ---------------------------------------------------------- explanations
    explain: bool = True
    explain_benign_rows: int = 300
    explain_malicious_rows: int = 300
    shap_background_rows: int = 100
    lime_num_samples: int = 1000
    lime_background_rows: int = 2000
    k_fracs: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    stability_ks: list = field(default_factory=lambda: [3, 5])
    marginal_draws: int = 16
    knn_neighbors: int = 8
    knn_pool_rows: int = 5000
    random_order_draws: int = 3
    # MLP views: PGD constrained at all eps + PGD unconstrained at max eps.
    # XGB views: black-box constrained/unconstrained at max eps.
    adv_views_reduced: bool = False    # True = only constrained@max-eps (split-sensitivity runs)
    repeat_seeds: int = 5              # repeated-call LIME/SHAP-background on the first N seeds
    repeat_rows: int = 100
    lime_repeats: int = 5
    shap_bg_repeats: int = 3

    # ---------------------------------------------------------- extras
    run_at_sweep: bool = False
    at_sweep_eps: list = field(default_factory=lambda: [0.01, 0.03, 0.05])
    at_sweep_lambda: list = field(default_factory=lambda: [0.5, 1.0])
    at_sweep_seeds: int = 5
    run_multiclass: bool = False
    multiclass_seeds: int = 5
    run_profiling: bool = False
    run_baseline_ablation: bool = False
    run_operating_point: bool = False
    run_v1_path: bool = False
    profile_rows: int = 100
    # ---------------------------------------------------------- second-round analyses
    run_r2_robustness: bool = False     # all seeds, threshold-free + matched-FPR, projection, strong tree attacks
    bb_queries_strong: int = 3000
    coord_passes: int = 3
    run_r2_fairness: bool = False       # LIME grid + KernelSHAP on a subset
    fair_rows: int = 120                # 60 benign + 60 malicious explained records
    fair_lime_samples: list = field(default_factory=lambda: [1000, 5000])
    fair_lime_kernel: list = field(default_factory=lambda: [0.25, 0.75, 2.0])   # x sqrt(d); 0.75 = LIME default
    kernel_bg_rows: int = 50
    kernel_nsamples: int = 2048
    data_variants: list = field(default_factory=list)   # split ids 10/20/30 (see data.split_variant)

    device: str = "auto"
    print_tables: bool = True          # print each run's result tables into the log (copy-paste fallback)

    def to_json(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------
# Dataset feature policies.  Columns not listed are auto-classified: numeric if
# >=95% of values parse as numbers, otherwise dropped (and logged).
# Roles of numeric features for the constrained threat model:
#   "up"   = mutable, attacker may only increase (padding, delay, extra packets)
#   "both" = mutable in both directions
#   anything not listed in MUTABLE = immutable (protocol semantics, server
#            responses, checksums, codes, one-hot categorical blocks).
# --------------------------------------------------------------------------
DATASET_SPECS = {
    "ton": {
        "label": "label",
        "family": "type",
        "benign_family": "normal",
        "file_patterns": ["train_test_network", "ton_iot", "network_dataset"],
        "drop": ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "dns_query",
                 "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
                 "http_referrer", "weird_name", "weird_addl"],
        "categorical": ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA",
                        "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
                        "ssl_established", "http_method", "http_version",
                        "http_orig_mime_types", "http_resp_mime_types", "weird_notice"],
        "integer": ["src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts",
                    "src_ip_bytes", "dst_ip_bytes", "http_request_body_len",
                    "http_response_body_len", "http_trans_depth", "dns_qclass",
                    "dns_qtype", "dns_rcode", "http_status_code"],
        "mutable": {"duration": "up", "src_bytes": "up", "src_pkts": "up",
                    "src_ip_bytes": "up", "http_request_body_len": "up"},
        # Explicit protocol relations (raw units): lhs >= sum(coef * feature) holds
        # for valid flows. Each is kept only if it holds in the clean training data.
        "relations": [
            {"lhs": "src_ip_bytes", "rhs": {"src_bytes": 1.0, "src_pkts": 20.0},
             "why": "IP bytes include payload plus >=20-byte IPv4 header per packet"},
            {"lhs": "dst_ip_bytes", "rhs": {"dst_bytes": 1.0, "dst_pkts": 20.0},
             "why": "same invariant for responder direction"},
        ],
    },
    "edge": {
        "label": "Attack_label",
        "family": "Attack_type",
        "benign_family": "normal",
        "file_patterns": ["ml-edgeiiot-dataset", "ml_edgeiiot", "dnn-edgeiiot-dataset"],
        # Ferrag et al. (2022) recommended drops + capture-artefact identifiers.
        "drop": ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                 "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                 "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
                 "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
                 "udp.stream", "tcp.ack_raw", "mbtcp.trans_id", "mbtcp.unit_id",
                 "http.tls_port"],
        "categorical": ["http.request.method", "http.referer", "http.request.version",
                        "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname",
                        "mqtt.topic"],
        "integer": ["tcp.len", "http.content_length", "mqtt.len", "tcp.ack", "tcp.seq",
                    "icmp.seq_le", "mqtt.msg_decoded_as", "mqtt.topic_len",
                    "mbtcp.len", "tcp.flags", "mqtt.hdrflags", "mqtt.msgtype"],
        "mutable": {"tcp.len": "up", "http.content_length": "up", "mqtt.len": "up"},
        "relations": [],
    },
}

ALL_MODELS = ["mlp_clean", "mlp_fgsmat", "mlp_pgdat", "xgb"]

# Exact feature lists of the original (v1) submission, from its archived metadata.
V1_FEATURES = {
    "ton": ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts", "dns_qtype", "dns_rcode"],
    "edge": ["arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le", "http.content_length", "http.response",
             "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst",
             "tcp.connection.syn", "tcp.connection.synack", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.seq",
             "udp.stream", "dns.qry.name", "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype"],
}


def session_config(session: str, **base) -> Config:
    """Presets for the reported run (one Kaggle notebook version per session)."""
    s = session.upper()
    if s == "TEST":
        return Config(**base, datasets=["ton", "edge"], seeds=[17], max_epochs=5, xgb_n_estimators=100,
                      explain_benign_rows=50, explain_malicious_rows=50, attack_rows=300, lime_num_samples=300,
                      bb_queries=100, repeat_seeds=1, repeat_rows=20, lime_repeats=2, shap_bg_repeats=2)
    if s == "A":
        return Config(**base, datasets=["ton"])
    if s == "B":
        return Config(**base, datasets=["edge"])
    if s == "C1":   # partition sensitivity: 2 extra stratified partitions x 5 seeds
        return Config(**base, datasets=["ton", "edge"], split_ids=[1, 2], seeds=SEEDS_20[:5],
                      adv_views_reduced=True, repeat_seeds=0)
    if s == "C2":   # AT trade-off sweep, multiclass, resource profiling
        return Config(**base, datasets=["ton", "edge"], seeds=SEEDS_20[:5], split_ids=[],
                      run_at_sweep=True, run_multiclass=True, run_profiling=True,
                      run_baseline_ablation=True)
    if s == "C4":   # v1 -> v2 baseline path (feature list, scaling) + absolute-memory profiling
        return Config(**base, datasets=["ton", "edge"], seeds=SEEDS_20[:5], split_ids=[],
                      run_v1_path=True, run_profiling=True)
    if s == "C3":   # attack success vs. FPR-matched operating points (short)
        return Config(**base, datasets=["ton", "edge"], seeds=SEEDS_20[:5], split_ids=[],
                      run_operating_point=True)
    # ---------------- second review round
    if s == "R0":   # quick test of all second-round code paths
        return Config(**base, datasets=["ton", "edge"], seeds=[17], split_ids=[], max_epochs=3,
                      xgb_n_estimators=60, attack_rows=200, bb_queries=50, bb_queries_strong=150, coord_passes=1,
                      run_r2_robustness=True, run_r2_fairness=True, fair_rows=20, fair_lime_samples=[300, 600],
                      fair_lime_kernel=[0.75, 2.0], kernel_nsamples=256, kernel_bg_rows=20,
                      data_variants=[10, 20, 30], explain_benign_rows=40, explain_malicious_rows=40,
                      lime_num_samples=300, repeat_seeds=0, adv_views_reduced=True)
    if s == "R1":   # ToN-IoT: 20 seeds, matched-FPR + threshold-free robustness, projection, strong tree attacks
        return Config(**base, datasets=["ton"], split_ids=[], run_r2_robustness=True)
    if s == "R2":   # Edge-IIoTset: same as R1
        return Config(**base, datasets=["edge"], split_ids=[], run_r2_robustness=True)
    if s == "R3":   # explainer fairness: LIME grid + KernelSHAP, 5 seeds, both datasets
        return Config(**base, datasets=["ton", "edge"], seeds=SEEDS_20[:5], split_ids=[], run_r2_fairness=True)
    if s == "R4":   # data variants: label conflicts retained, temporal (global), temporal (within family)
        return Config(**base, datasets=["ton", "edge"], seeds=SEEDS_20[:5], split_ids=[],
                      data_variants=[10, 20, 30], adv_views_reduced=True, repeat_seeds=0)
    raise ValueError(session)
