"""시연 pcap 만들기: 시험 캡처의 시험 구간 시작점부터 N MiB를 패킷 경계에서 잘라 저장하고, **시험 행으로만 만든** 원본 대응표(`*.match.csv`)를 그 파일의 SHA에 묶어 함께 저장한다.

자른 파일에서 다시 부여되는 flow_id는 원본 ID로 가정하지 않는다. 대응은 (클라이언트 ip·port, 서버 ip, 시작 시각)로만 하며 분석 때 `match_to_original`이 매칭을 판정한다.
대응표에는 원본 앞 10개 값(`orig_seq_0..9`)과 `source`(HTTPS는 피해자 구분)를 같이 저장해, 잘린 파일의 입력이 원본과 다른 연결을 채점에서 뺄 수 있게 한다.
"""

import argparse
import struct
from pathlib import Path

import pandas as pd

from src.track_b.demo_infer import file_sha256
from src.track_b.demo_model import K, load_data
from src.track_b.four_class import GAP_S
from src.track_b.pcap_flows import assemble_flows

TABLE_COLS = ["pcap", "flow_id", "label", "source", "client_ip", "client_port", "server_ip", "start_ts", "payload_packets"] + [f"seq_{i}" for i in range(K)]


def cut_from(src, dst, min_bytes, start_ts=None):
    """classic libpcap을 시작 시각(start_ts 이상)부터 읽어 파일 크기가 min_bytes 이상이 되는 첫 패킷까지 dst에 쓴다. (바이트, 패킷 수) 반환."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        header = fin.read(24)
        fout.write(header)
        size, packets = 24, 0
        while size < min_bytes:
            rec = fin.read(16)
            if len(rec) < 16:
                break
            ts_s, ts_u, incl, _ = struct.unpack("<IIII", rec)
            if start_ts is not None and ts_s + ts_u / 1e6 < start_ts:
                fin.seek(incl, 1)
                continue
            data = fin.read(incl)
            if len(data) < incl:
                break
            fout.write(rec + data)
            size += 16 + incl
            packets += 1
    return size, packets


def build_match_table(data, pcap_name, lo, hi, sha):
    """평가 코드의 시험 행(`part == "test"`)만 골라, 자른 파일의 시간 창 [lo, hi]에 시작한 연결로 대응표를 만든다."""
    rows = data[(data.pcap == pcap_name) & (data.part == "test") & (data.start_ts >= lo - 1e-6) & (data.start_ts <= hi + 1e-6)]
    return rows[TABLE_COLS].rename(columns={"flow_id": "orig_flow_id", "payload_packets": "orig_payload_packets", "pcap": "orig_pcap",
                                            **{f"seq_{i}": f"orig_seq_{i}" for i in range(K)}}).assign(pcap_sha256=sha)


def make_demo_pcap(data_dir, src_name, dst, min_bytes, start_ts=None):
    data, boundary = load_data(data_dir, extra=("client_port",))
    if start_ts == "test_start":
        start_ts = boundary + GAP_S  # Probing: 4클래스 분할의 시험 시작 시각 T+600초
    size, packets = cut_from(Path(data_dir) / "raw/hikari" / src_name, dst, min_bytes, start_ts)
    sha = file_sha256(dst)
    flows, _, _ = assemble_flows(dst)
    table = build_match_table(data, src_name, min(f.start_ts for f in flows), max(f.last_ts for f in flows), sha)
    table.to_csv(str(dst) + ".match.csv", index=False)
    return dict(path=str(dst), sha256=sha, bytes=size, packets=packets, start_ts=start_ts, cut_connections=len(flows), table_rows=len(table),
                table_by_source=table.groupby("source").size().to_dict())


def main():
    from src.common.config import DATA_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="출력 이름. 예: https_0416_64")
    ap.add_argument("--src", required=True, help="원본 pcap 파일 이름(DATA_DIR/raw/hikari 아래)")
    ap.add_argument("--mib", type=float, default=64)
    ap.add_argument("--from-test-start", action="store_true", help="Probing: 시험 시작 시각(T+600초)부터 자른다")
    args = ap.parse_args()
    out = DATA_DIR / "demo_pcaps"
    out.mkdir(exist_ok=True)
    dst = out / f"{args.name}.pcap"
    if dst.exists():
        raise FileExistsError("기존 시연 pcap을 덮어쓰지 않습니다")
    print(make_demo_pcap(DATA_DIR, args.src, dst, int(args.mib * 1024 * 1024), "test_start" if args.from_test_start else None))


if __name__ == "__main__":
    main()
