"""HIKARI SCANCMS 캡처(2021-05-02 12:06)에서 443 플로우를 뽑고 Probing 잠정 라벨을 붙인 뒤 CSV와 대칭 감사한다(안건 9 확정안).

라벨 규칙(성능 확인 전 고정, 캡처 단위 잠정): 공격자 103.255.15.150 → 피해자 128.199.242.104:443 = probing, 그 밖의 443 = benign.
CSV(ALLFLOWMETER)만으로는 라벨을 정할 수 없다(같은 공격자→피해자 쌍에 Benign 행이 많다). 감사는 #34와 같은 방식(키 + 페이로드 수 일치)이다.
k=5·10 도달 비율(payload_packets >= k)을 클래스별로 같이 낸다.

Run: uv run python -m src.track_b.https_probing --run-id <ID>
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.results import sha256_file
from src.track_b import https_flows
from src.track_b.https_flows import CSV_NAME, read_csv_443, reaudit
from src.track_b.pcap_flows import assemble_flows, flow_rows

PROBING = "probing"
PCAP = "Sunday_2021-05-02_1206.pcap"
CAPTURE = dict(day="2021-05-02", attacker="103.255.15.150", victim="128.199.242.104", attack=PROBING)
https_flows.AGREE.setdefault(PROBING, {"Probing"})  # 감사에서 probing 라벨과 같은 CSV 범주


def reach_table(flows):
    out = []
    for label, g in flows.groupby("label"):
        row = dict(label=label, flows=len(g), payload0=int((g.payload_packets == 0).sum()), midstream=int(g.midstream.sum()))
        for k in (5, 10, 20, 30):
            row[f"reached_k{k}"] = int((g.payload_packets >= k).sum())
        row["payload_median"] = float(g.payload_packets.median())
        row["start_min"], row["start_max"] = float(g.start_ts.min()), float(g.start_ts.max())
        out.append(row)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR, MACHINE

    root = Path(__file__).resolve().parents[2]
    pcap, csv_path = DATA_DIR / "raw" / "hikari" / PCAP, DATA_DIR / "raw" / "hikari" / CSV_NAME
    out_data = DATA_DIR / "processed" / f"track_b_https_probing_{args.run_id}"
    out = root / "outputs/results" / f"track_b_https_probing_{args.run_id}"
    if out_data.exists() or out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows, skipped, tcp_packets = assemble_flows(pcap)
    frame = pd.DataFrame(flow_rows(flows, PCAP, CAPTURE))
    status, table = reaudit(frame[["pcap", "flow_id", "label", "client_ip", "client_port", "server_ip", "payload_packets", "duration"]], read_csv_443(csv_path))
    reach = reach_table(frame)
    out_data.mkdir(parents=True)
    out.mkdir(parents=True)
    frame.to_csv(out_data / "flows.csv", index=False)
    status.to_csv(out_data / "flow_status.csv", index=False)
    table.to_csv(out / "label_audit.csv", index=False)
    reach.to_csv(out / "reach_by_class.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, machine=MACHINE, pcap=PCAP, pcap_bytes=pcap.stat().st_size, pcap_md5_zenodo="21a44865c0cdd6fa417e28e69170aa84",
        tcp_packets=tcp_packets, skipped=skipped, flows=len(frame), capture_rule=CAPTURE, flows_sha256=sha256_file(out_data / "flows.csv"),
        csv_sha256=sha256_file(csv_path), flow_status_sha256=sha256_file(out_data / "flow_status.csv"),
        code_sha256={"src/track_b/https_probing.py": sha256_file(Path(__file__)), "src/track_b/pcap_flows.py": sha256_file(root / "src/track_b/pcap_flows.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        note="성능 확인 전 라벨 감사·범위 고정. 캡처 단위 잠정 라벨이며 CSV는 확인용이다. 모델 입력이 아니다"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(table.T.to_string())
    print(reach.T.to_string())
    print("DONE", out_data, out, flush=True)


if __name__ == "__main__":
    main()
