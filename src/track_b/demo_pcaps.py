"""시연 pcap 만들기: 원본 시험 pcap의 앞쪽을 잘라 저장하고, 원본 시험 행과의 대응표(`*.match.csv`)를 그 파일의 SHA에 묶어 함께 저장한다.

자른 파일에서 다시 부여되는 flow_id는 원본 ID로 가정하지 않는다. 대응은 (클라이언트 ip·port, 서버 ip, 시작 시각)로만 하며 분석 때 `match_to_original`이 매칭을 판정한다.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.track_b.demo_infer import file_sha256
from src.track_b.demo_model import INPUTS
from src.track_b.pcap_flows import assemble_flows
from src.track_b.perf_measure import cut_pcap

ORIG_COLS = ["pcap", "day", "flow_id", "label", "client_ip", "client_port", "server_ip", "start_ts", "payload_packets"]


def make_demo_pcap(src, dst, min_bytes, flows_csv, orig_pcap_name):
    """src 앞부분을 min_bytes 이상이 되는 첫 패킷 경계에서 잘라 dst에 저장하고 대응표를 dst + '.match.csv'로 쓴다."""
    size, packets = cut_pcap(src, dst, min_bytes)
    sha = file_sha256(dst)
    flows, _, _ = assemble_flows(dst)
    lo, hi = min(f.start_ts for f in flows), max(f.last_ts for f in flows)
    orig = pd.read_csv(flows_csv, usecols=ORIG_COLS, keep_default_na=False).query("payload_packets > 0")
    orig = orig[(orig.pcap == orig_pcap_name) & (orig.start_ts >= lo - 1e-6) & (orig.start_ts <= hi + 1e-6)]
    table = orig.rename(columns={"flow_id": "orig_flow_id", "payload_packets": "orig_payload_packets", "pcap": "orig_pcap"}).assign(pcap_sha256=sha)
    table.to_csv(str(dst) + ".match.csv", index=False)
    return dict(path=str(dst), sha256=sha, bytes=size, packets=packets, cut_connections=len(flows), table_rows=len(table))


def main():
    from src.common.config import DATA_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="출력 이름. 예: https_0416")
    ap.add_argument("--src", required=True, help="원본 pcap 파일 이름(DATA_DIR/raw/hikari 아래)")
    ap.add_argument("--mib", type=float, default=10)
    ap.add_argument("--flows", default=INPUTS["https"], help="원본 시험 행이 있는 flows.csv(DATA_DIR 기준 상대 경로)")
    args = ap.parse_args()
    out = DATA_DIR / "demo_pcaps"
    out.mkdir(exist_ok=True)
    dst = out / f"{args.name}.pcap"
    if dst.exists():
        raise FileExistsError("기존 시연 pcap을 덮어쓰지 않습니다")
    print(make_demo_pcap(DATA_DIR / "raw/hikari" / args.src, dst, int(args.mib * 1024 * 1024), DATA_DIR / args.flows, args.src))


if __name__ == "__main__":
    main()
