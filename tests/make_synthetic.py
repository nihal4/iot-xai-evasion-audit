"""Synthetic CSVs with the real ToN-IoT / Edge-IIoTset column schemas (for smoke tests only)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path

def ton(n, rng):
    fams = ["normal","backdoor","ddos","dos","injection","mitm","password","ransomware","scanning","xss"]
    fam = rng.choice(fams, n, p=[.3]+[.7/9]*9)
    k = np.array([fams.index(f) for f in fam])
    pk = rng.poisson(2 + k, n); src_bytes = rng.poisson(50 * (1 + k), n) * (rng.random(n) < .8)
    df = pd.DataFrame({
        "src_ip": [f"192.168.1.{i%255}" for i in range(n)], "src_port": rng.integers(1, 65535, n),
        "dst_ip": "10.0.0.1", "dst_port": rng.choice([80, 53, 443, 22], n),
        "proto": rng.choice(["tcp","udp","icmp"], n), "service": rng.choice(["-","http","dns","ssl"], n),
        "duration": rng.exponential(1 + k / 3, n), "src_bytes": src_bytes, "dst_bytes": rng.poisson(30, n),
        "conn_state": np.where(k % 2 == 0, "SF", rng.choice(["S0","REJ","OTH"], n)), "missed_bytes": 0,
        "src_pkts": pk, "src_ip_bytes": src_bytes + 20 * pk + rng.integers(0, 20, n), "dst_pkts": rng.poisson(2, n),
        "dns_query": rng.choice(["-","a.com","b.org"], n), "dns_qclass": rng.choice([0, 1], n),
        "dns_qtype": rng.choice([0, 1, 28], n), "dns_rcode": rng.choice([0, 3], n),
        "dns_AA": rng.choice(["-","T","F"], n), "dns_RD": rng.choice(["-","T","F"], n),
        "dns_RA": rng.choice(["-","T","F"], n), "dns_rejected": rng.choice(["-","T","F"], n),
        "ssl_version": "-", "ssl_cipher": "-", "ssl_resumed": "-", "ssl_established": "-",
        "ssl_subject": "-", "ssl_issuer": "-", "http_trans_depth": rng.choice(["-","1"], n),
        "http_method": rng.choice(["-","GET","POST"], n), "http_uri": "-", "http_version": "-",
        "http_request_body_len": rng.poisson(5 * (k > 4), n), "http_response_body_len": 0, "http_status_code": rng.choice([0, 200], n),
        "http_user_agent": "-", "http_orig_mime_types": "-", "http_resp_mime_types": "-",
        "weird_name": "-", "weird_addl": "-", "weird_notice": rng.choice(["-","F"], n)})
    df["dst_ip_bytes"] = df["dst_bytes"] + 20 * df["dst_pkts"]
    df["label"] = (fam != "normal").astype(int); df["type"] = fam
    return df

def edge(n, rng):
    fams = ["Normal","DDoS_UDP","DDoS_ICMP","SQL_injection","Password","Vulnerability_scanner","DDoS_TCP",
            "DDoS_HTTP","Uploading","Backdoor","Port_Scanning","XSS","Ransomware","MITM","Fingerprinting"]
    fam = rng.choice(fams, n, p=[.25]+[.75/14]*14); k = np.array([fams.index(f) for f in fam])
    cols = ["frame.time","ip.src_host","ip.dst_host","arp.dst.proto_ipv4","arp.opcode","arp.hw.size","arp.src.proto_ipv4",
            "icmp.checksum","icmp.seq_le","icmp.transmit_timestamp","icmp.unused","http.file_data","http.content_length",
            "http.request.uri.query","http.request.method","http.referer","http.request.full_uri","http.request.version",
            "http.response","http.tls_port","tcp.ack","tcp.ack_raw","tcp.checksum","tcp.connection.fin","tcp.connection.rst",
            "tcp.connection.syn","tcp.connection.synack","tcp.dstport","tcp.flags","tcp.flags.ack","tcp.len","tcp.options",
            "tcp.payload","tcp.seq","tcp.srcport","udp.port","udp.stream","udp.time_delta","dns.qry.name","dns.qry.name.len",
            "dns.qry.qu","dns.qry.type","dns.retransmission","dns.retransmit_request","dns.retransmit_request_in",
            "mqtt.conack.flags","mqtt.conflag.cleansess","mqtt.conflags","mqtt.hdrflags","mqtt.len","mqtt.msg_decoded_as",
            "mqtt.msg","mqtt.msgtype","mqtt.proto_len","mqtt.protoname","mqtt.topic","mqtt.topic_len","mqtt.ver",
            "mbtcp.len","mbtcp.trans_id","mbtcp.unit_id"]
    df = pd.DataFrame({c: 0.0 for c in cols}, index=range(n))
    df["frame.time"] = "2021 x"; df["ip.src_host"] = "192.168.0.1"; df["ip.dst_host"] = "0"
    df["tcp.len"] = rng.poisson(10 * (k % 5), n).astype(float); df["tcp.flags"] = rng.choice([16, 24, 2], n)
    df["tcp.seq"] = rng.integers(0, 1000, n); df["tcp.ack"] = rng.integers(0, 1000, n)
    df["tcp.ack_raw"] = rng.integers(0, 4e9, n); df["tcp.checksum"] = rng.integers(0, 65535, n)
    df["tcp.connection.syn"] = (k == 6).astype(float); df["mqtt.len"] = np.where(k == 0, rng.poisson(20, n), 0)
    df["http.content_length"] = rng.poisson(3 * (k == 3), n); df["udp.stream"] = rng.integers(0, 1e5, n)
    df["http.request.method"] = rng.choice(["0", "0.0", "GET", "POST"], n)
    df["http.referer"] = rng.choice(["0", "0.0", "127.0.0.1"], n); df["http.request.version"] = rng.choice(["0", "HTTP/1.1"], n)
    df["dns.qry.name.len"] = rng.choice(["0", "0.0", "1.0", "example.com"], n)
    df["mqtt.conack.flags"] = rng.choice(["0", "0.0", "0x00000000"], n); df["mqtt.protoname"] = rng.choice(["0", "0.0", "MQTT"], n)
    df["mqtt.topic"] = rng.choice(["0", "0.0", "Temperature_and_Humidity"], n)
    df["tcp.payload"] = "0"; df["mqtt.msg"] = "0"; df["tcp.options"] = "0"; df["http.file_data"] = "0"
    df["icmp.checksum"] = np.where(k == 2, rng.integers(0, 65535, n), 0)
    df["udp.time_delta"] = rng.exponential(0.1, n) * (k == 1)
    df["Attack_label"] = (fam != "Normal").astype(int); df["Attack_type"] = fam
    return df

if __name__ == "__main__":
    out = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    rng = np.random.default_rng(0)
    (out / "ton").mkdir(parents=True, exist_ok=True); (out / "edge").mkdir(parents=True, exist_ok=True)
    t = ton(n, rng); t = pd.concat([t, t.iloc[:50]])  # inject duplicates
    t.to_csv(out / "ton" / "train_test_network.csv", index=False)
    edge(n, rng).to_csv(out / "edge" / "ML-EdgeIIoT-dataset.csv", index=False)
    print("ok")
