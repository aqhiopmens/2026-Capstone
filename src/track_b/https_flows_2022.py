"""HIKARI 2022(v1.4.0) HTTPS 브루트포스 pcap에서 플로우를 뽑고 라벨을 감사한다 (성능을 보기 전 단계).

2021과 규칙이 다르다. 2021은 공격자·정상이 같은 클라이언트 IP라 '서버 IP가 피해자인가'로 라벨을 정했지만, 2022는 공격자
(203.178.143.112)가 정상과 다른 IP이고 정상도 피해자 서버로 간다. 그래서 라벨 = **공격자 클라이언트 IP → 피해자 서버:443 = 공격,
그 밖의 443 = 정상**이다(잠정 라벨). CSV(ALLFLOWMETER_HIKARI2022)는 확인용 감사 열이며 모델 입력이 아니다.

Run: uv run python -m src.track_b.https_flows_2022 --run-id <ID>
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.results import sha256_file
from src.track_b.https_flows import ATTACK, BENIGN, label_flow, reaudit
from src.track_b.pcap_flows import K_MAX, assemble_flows, flow_rows

CAPTURES_2022 = {
    "Sunday_2022-04-10_2335_BRUTEFORCE_HTTPS_150s.pcap": dict(day="2022-04-10", attacker="203.178.143.112",
                                                             victims=("184.0.48.150", "184.0.48.169")),
}
CSV_2022 = "ALLFLOWMETER_HIKARI2022.csv"


def read_csv_2022(path):
    """2022 CSV의 443 행을 2021 감사 함수가 받는 열(traffic_category, data_pkts)로 맞춘다."""
    frame = pd.read_csv(path, usecols=["originh", "originp", "responh", "responp", "attack_category", "fwd_data_pkts_tot",
                                       "bwd_data_pkts_tot"])
    frame = frame[frame.responp == 443].copy()
    frame["traffic_category"] = frame.attack_category
    frame["data_pkts"] = frame.fwd_data_pkts_tot + frame.bwd_data_pkts_tot
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR, MACHINE

    root = Path(__file__).resolve().parents[2]
    source = DATA_DIR / "raw" / "hikari"
    out_data = DATA_DIR / "processed" / f"track_b_https2022_{args.run_id}"
    out_report = root / "outputs/results" / f"track_b_https2022_audit_{args.run_id}"
    if out_data.exists() or out_report.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    csv_443 = read_csv_2022(source / CSV_2022)
    rows, summary = [], []
    for name, capture in CAPTURES_2022.items():
        flows, skipped, tcp_packets = assemble_flows(source / name)
        cap_rows = flow_rows(flows, name, dict(day=capture["day"], attacker="", victim="", attack=""))
        for r in cap_rows:
            r["label"] = label_flow(r["client_ip"], r["server_ip"], capture)
        rows += cap_rows
        summary.append(dict(pcap=name, tcp_packets=tcp_packets, flows=len(flows), skipped=skipped,
                            pcap_sha256=sha256_file(source / name)))
    flows_df = pd.DataFrame(rows)
    status, table = reaudit(flows_df, csv_443)
    out_data.mkdir(parents=True)
    out_report.mkdir(parents=True)
    flows_df.to_csv(out_data / "flows.csv", index=False)
    status.to_csv(out_data / "flow_status.csv", index=False)
    table.to_csv(out_report / "label_audit.csv", index=False)
    csv_attack = csv_443[csv_443.Label.eq(1)] if "Label" in csv_443 else csv_443[csv_443.attack_category != "Benign"]
    csv_summary = csv_443.groupby(["attack_category", "originh", "responh"]).size().reset_index(name="rows")
    csv_summary.to_csv(out_report / "csv_443_summary.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out_report / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, machine=MACHINE, captures=CAPTURES_2022, summary=summary, k_max=K_MAX,
        flows_sha256=sha256_file(out_data / "flows.csv"), flow_status_sha256=sha256_file(out_data / "flow_status.csv"),
        csv_sha256=sha256_file(source / CSV_2022),
        code_sha256={"src/track_b/https_flows_2022.py": sha256_file(Path(__file__)),
                     "src/track_b/https_flows.py": sha256_file(root / "src/track_b/https_flows.py"),
                     "src/track_b/pcap_flows.py": sha256_file(root / "src/track_b/pcap_flows.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        label_rule="공격자 클라이언트 IP → 피해자 서버:443 = 공격, 그 밖의 443 = 정상(잠정). CSV는 확인용",
        note="성능 확인 전 라벨 감사. flows.csv·flow_status.csv는 DATA_DIR에만 있다"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table.T.to_string())
    print(status.groupby(["label", "status_key_payload"]).size().to_string())
    print("DONE", out_data, out_report, flush=True)


if __name__ == "__main__":
    main()
