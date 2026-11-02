"""HIKARI BRUTEFORCE_HTTPS 캡처(04-11, 04-16)에서 플로우를 뽑고 잠정 라벨을 붙인다.

라벨 규칙(docs/plan.md, 잠정): 공격자 → 피해자:443 = bruteforce_https, 그 밖의 443 = benign.
04-16은 피해자가 둘이다. CSV(ALLFLOWMETER)만으로는 라벨을 정할 수 없어(공격자 → 128.199.242.104에
Benign 행이 6,794개) 캡처 단위 규칙을 쓰고, CSV는 확인용 감사 열(csv_category)로만 붙인다.

Run: uv run python -m src.track_b.https_flows --run-id <ID>
"""

import argparse
import csv
import json
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.results import sha256_file
from src.track_b.pcap_flows import K_MAX, assemble_flows, flow_rows

ATTACKER = "103.255.15.23"
CAPTURES = {
    "Sunday_2021-04-11_2154.pcap": dict(day="2021-04-11", attacker=ATTACKER, victims=("128.199.242.104",)),
    "Friday_2021-04-16_2304.pcap": dict(day="2021-04-16", attacker=ATTACKER,
                                        victims=("128.199.242.104", "128.199.88.81")),
}
ATTACK, BENIGN = "bruteforce_https", "benign"
CSV_NAME = "ALLFLOWMETER_HIKARI2021.csv.zip"


def label_flow(client_ip, server_ip, capture):
    return ATTACK if client_ip == capture["attacker"] and server_ip in capture["victims"] else BENIGN


def csv_audit_columns(rows, csv_443):
    """(client_ip, client_port, server_ip)와 페이로드 패킷 수가 같은 CSV 행을 찾아 감사 열을 붙인다.

    csv_match: 0 = CSV에 없음, 1 = 정확히 1행, 2 이상 = 같은 키가 여러 행(csv_category는 값이 하나일 때만 채움).
    """
    by_key = defaultdict(list)
    for key, category, packets in zip(zip(csv_443.originh, csv_443.originp, csv_443.responh),
                                      csv_443.traffic_category, csv_443.data_pkts):
        by_key[key].append((category, packets))
    for row in rows:
        hits = [h for h in by_key.get((row["client_ip"], row["client_port"], row["server_ip"]), [])
                if h[1] == row["payload_packets"]]
        categories = {h[0] for h in hits}
        row["csv_match"] = len(hits)
        row["csv_category"] = categories.pop() if len(categories) == 1 else ("" if not hits else "mixed")
    return rows


def read_csv_443(path):
    with zipfile.ZipFile(path) as zf:
        frame = pd.read_csv(zf.open("ALLFLOWMETER_HIKARI2021.csv"),
                            usecols=["originh", "originp", "responh", "responp", "traffic_category", "flow_duration",
                                     "fwd_data_pkts_tot", "bwd_data_pkts_tot"])
    frame = frame[frame.responp == 443].copy()
    frame["data_pkts"] = frame.fwd_data_pkts_tot + frame.bwd_data_pkts_tot
    return frame


AGREE = {ATTACK: {"Bruteforce"}, BENIGN: {"Benign", "Background"}}


def match_status(label, categories):
    """CSV 후보 범주 목록(행마다 하나, 중복 포함) -> 라벨과의 관계. 양쪽 라벨에 같은 기준을 쓴다.

    unmatched / single_agree / single_disagree / multi_agree(여러 행이 모두 라벨과 같은 범주) /
    conflict(후보 범주가 섞이거나 라벨과 다른 범주가 하나라도 있음)
    """
    if not categories:
        return "unmatched"
    inside = [c in AGREE[label] for c in categories]
    if len(categories) == 1:
        return "single_agree" if inside[0] else "single_disagree"
    return "multi_agree" if all(inside) else "conflict"


def reaudit(flows, csv_443):
    """flows.csv(DataFrame)를 기준별로 CSV와 대조한다. 반환: (flow별 상태, 기준별 분모 표).

    기준: key = (client_ip, client_port, server_ip) 일치, +payload = 거기에 페이로드 패킷 수(CSV 정/역방향 합)까지 일치.
    """
    csv_443 = csv_443.reset_index(drop=True)
    index = defaultdict(list)
    for i, key in enumerate(zip(csv_443.originh, csv_443.originp, csv_443.responh)):
        index[key].append(i)
    category, packets = csv_443.traffic_category.to_numpy(), csv_443.data_pkts.to_numpy()
    status_rows, used = [], defaultdict(set)
    for f in flows.itertuples():
        key_hits = index.get((f.client_ip, f.client_port, f.server_ip), [])
        pay_hits = [i for i in key_hits if packets[i] == f.payload_packets]
        group = (f.pcap, f.label)
        used[group].update(pay_hits)
        status_rows.append(dict(pcap=f.pcap, flow_id=f.flow_id, label=f.label, server_ip=f.server_ip,
                                payload_packets=f.payload_packets, key_rows=len(key_hits), key_payload_rows=len(pay_hits),
                                status_key=match_status(f.label, [category[i] for i in key_hits]),
                                status_key_payload=match_status(f.label, [category[i] for i in pay_hits])))
    status = pd.DataFrame(status_rows)
    out = []
    for (pcap, label), g in status.groupby(["pcap", "label"]):
        row = dict(pcap=pcap, label=label, flows=len(g), payload0=int((g.payload_packets == 0).sum()),
                   key_flows=int((g.key_rows > 0).sum()), key_payload_flows=int((g.key_payload_rows > 0).sum()),
                   key_payload_nonzero_flows=int(((g.key_payload_rows > 0) & (g.payload_packets > 0)).sum()),
                   key_payload_one_row=int((g.key_payload_rows == 1).sum()),
                   csv_rows_hit_key_payload=len(used[(pcap, label)]))
        row.update({f"kp_{s}": int(n) for s, n in g.status_key_payload.value_counts().items()})
        row.update({f"k_{s}": int(n) for s, n in g.status_key.value_counts().items()})
        out.append(row)
    return status, pd.DataFrame(out).fillna(0)


def audit_table(rows):
    """(캡처, 라벨, 피해자 여부)별 플로우 수와 CSV 매칭 상태. 성능을 보기 전에 범위를 고정하는 근거."""
    table = defaultdict(lambda: defaultdict(int))
    for r in rows:
        group = table[(r["pcap"], r["label"], r["server_ip"] if r["label"] == ATTACK else "other")]
        group["flows"] += 1
        group[f"csv_match_{min(r['csv_match'], 2)}"] += 1
        group[f"csv_{r['csv_category'] or 'none'}"] += 1
        group["payload0"] += r["payload_packets"] == 0
        group["midstream"] += r["midstream"]
    return [dict(pcap=p, label=l, group=g, **dict(v)) for (p, l, g), v in sorted(table.items())]


def duration_tolerance_table(flows, csv_443, tolerances=(0.01, 0.05, 0.1, 1.0, float("inf"))):
    """정상 플로우의 (키+페이로드 수) 일치 수가 지속시간 허용오차에 따라 어떻게 변하는지. 이전 보고 수치와 대조용."""
    merged = flows[flows.label == BENIGN].merge(csv_443, left_on=["client_ip", "client_port", "server_ip"],
                                                right_on=["originh", "originp", "responh"])
    merged = merged[merged.payload_packets == merged.data_pkts]
    gap = (merged.duration - merged.flow_duration).abs()
    return pd.DataFrame([dict(pcap=p, tolerance_s=t, matched_flows=int(g.loc[gap[g.index] <= t, "flow_id"].nunique()))
                         for p, g in merged.groupby("pcap") for t in tolerances])


def audit_main(args):
    from src.common.config import DATA_DIR, MACHINE

    root = Path(__file__).resolve().parents[2]
    flows_path = DATA_DIR / "processed" / f"track_b_https_{args.audit_of}" / "flows.csv"
    csv_path = DATA_DIR / "raw" / "hikari" / CSV_NAME
    out_data = DATA_DIR / "processed" / f"track_b_https_audit_{args.run_id}"
    out_report = root / "outputs/results" / f"track_b_https_audit_{args.run_id}"
    if out_data.exists() or out_report.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows = pd.read_csv(flows_path, keep_default_na=False,
                        usecols=["pcap", "flow_id", "label", "client_ip", "client_port", "server_ip",
                                 "payload_packets", "duration"])
    csv_443 = read_csv_443(csv_path)
    status, table = reaudit(flows, csv_443)
    tolerance = duration_tolerance_table(flows, csv_443)
    out_data.mkdir(parents=True)
    out_report.mkdir(parents=True)
    status.to_csv(out_data / "flow_status.csv", index=False)
    table.to_csv(out_report / "label_audit_v2.csv", index=False)
    tolerance.to_csv(out_report / "duration_tolerance.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out_report / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, audit_of=args.audit_of, machine=MACHINE, flows_sha256=sha256_file(flows_path),
        csv_sha256=sha256_file(csv_path), flow_status_sha256=sha256_file(out_data / "flow_status.csv"),
        code_sha256={"src/track_b/https_flows.py": sha256_file(Path(__file__))}, git_commit=git("rev-parse", "HEAD"),
        git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        note="성능 확인 전 라벨 감사 v2. 상태는 양쪽 라벨에 같은 기준(match_status)으로 매김. 모델 입력이 아니다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.T.to_string())
    print("DONE", out_data, out_report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--audit-of", help="이미 추출한 flows run-id를 CSV와 다시 대조(감사 v2). 추출은 하지 않는다")
    args = parser.parse_args()
    if args.audit_of:
        return audit_main(args)
    from src.common.config import DATA_DIR, MACHINE

    root = Path(__file__).resolve().parents[2]
    source = DATA_DIR / "raw" / "hikari"
    out_data = DATA_DIR / "processed" / f"track_b_https_{args.run_id}"
    out_report = root / "outputs/results" / f"track_b_https_extract_{args.run_id}"
    if out_data.exists() or out_report.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    csv_443 = read_csv_443(source / CSV_NAME)
    all_rows, summary = [], []
    for name, capture in CAPTURES.items():
        flows, skipped, tcp_packets = assemble_flows(source / name)
        rows = flow_rows(flows, name, dict(day=capture["day"], attacker="", victim="", attack=""))
        for r in rows:
            r["label"] = label_flow(r["client_ip"], r["server_ip"], capture)
        csv_audit_columns(rows, csv_443)
        all_rows.extend(rows)
        summary.append(dict(pcap=name, tcp_packets=tcp_packets, flows=len(flows), skipped=skipped,
                            pcap_sha256=sha256_file(source / name)))
        print(summary[-1], flush=True)
    out_data.mkdir(parents=True)
    out_report.mkdir(parents=True)
    with (out_data / "flows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    audit = audit_table(all_rows)
    pd.DataFrame(audit).fillna(0).to_csv(out_report / "label_audit.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out_report / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, machine=MACHINE, captures=CAPTURES, flows_sha256=sha256_file(out_data / "flows.csv"),
        csv_sha256=sha256_file(source / CSV_NAME), code_sha256={"src/track_b/https_flows.py": sha256_file(Path(__file__)),
                                                                "src/track_b/pcap_flows.py": sha256_file(root / "src/track_b/pcap_flows.py")},
        git_commit=git("rev-parse", "HEAD"), k_max=K_MAX, summary=summary,
        note="라벨은 캡처 단위 잠정 규칙. csv_match/csv_category는 확인용 감사 열이며 모델 입력이 아니다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DONE", out_data, out_report, flush=True)


if __name__ == "__main__":
    main()
