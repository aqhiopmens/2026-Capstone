"""HIKARI pcap에서 TCP 443 플로우를 조립하고 앞 k개 페이로드 패킷 시퀀스를 만든다 (표준 라이브러리만 사용).

A 트랙(Malicious_TLS)의 시퀀스와 같은 정의: 페이로드가 있는 패킷만 세고, 클라이언트→서버는 +길이,
서버→클라이언트는 -길이, 30칸 뒤쪽을 0으로 채운다. 식별자(IP·포트)는 모델 입력이 아니다.

Run: uv run python -m src.track_b.pcap_flows --run-id <ID>
"""

import argparse
import csv
import json
import mmap
import socket
import struct
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.common.results import sha256_file

SERVER_PORT = 443
K_MAX = 30
KS = (5, 10, 20, 30)
IDLE_TIMEOUT = 120.0  # 초. 마지막 패킷 후 이만큼 지나면 같은 4-튜플도 새 플로우
PADDING_POLICY = "trailing_zero_30"
FIN, SYN, RST, ACK = 0x01, 0x02, 0x04, 0x10

# 캡처 파일이 곧 공격 유형 (docs/plan.md 라벨 규칙): 공격자→피해자:443 = 공격, 그 밖의 443 = 정상
CAPTURES = {
    "Monday_2021-04-12_0611.pcap": dict(day="2021-04-12", attacker="103.255.15.23",
                                        victim="128.199.242.104", attack="bruteforce_xml"),
    "Saturday_2021-04-17_0357.pcap": dict(day="2021-04-17", attacker="103.255.15.23",
                                          victim="128.199.242.104", attack="bruteforce_xml"),
}


@dataclass
class Flow:
    client_ip: str
    client_port: int
    server_ip: str
    start_ts: float
    last_ts: float
    midstream: bool = False
    packets: int = 0
    payload_packets: int = 0
    retransmissions: int = 0
    closed: bool = False
    seq: list = field(default_factory=list)    # 최대 30개, 부호 있는 길이
    times: list = field(default_factory=list)  # 같은 위치 패킷의 start_ts 대비 경과 초
    seq_ref: dict = field(default_factory=dict)  # 방향별 기준 순서 번호(첫 페이로드)
    ranges: dict = field(default_factory=dict)   # 방향별 이미 받은 바이트 구간(병합된 [시작, 끝) 목록)


def iter_tcp_packets(path):
    """(ts, src, sport, dst, dport, flags, seq, payload_len) 를 캡처 순서대로 낸다.

    클래식 libpcap(마이크로초)·이더넷(+VLAN 한 겹)·IPv4·TCP만 읽는다. 페이로드 길이는 이더넷
    패딩이 섞이지 않도록 IP 전체 길이에서 계산한다. 조각난 IP 패킷은 건너뛴다(skipped에 집계).
    """
    skipped = defaultdict(int)
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        magic = mm[:4]
        if magic != b"\xd4\xc3\xb2\xa1":
            raise ValueError(f"지원하지 않는 pcap 형식: {magic.hex()} (리틀엔디안 마이크로초 classic만 지원)")
        if struct.unpack_from("<I", mm, 20)[0] != 1:
            raise ValueError("이더넷(linktype 1) 캡처만 지원합니다")
        pos, size = 24, len(mm)
        while pos + 16 <= size:
            ts_s, ts_u, incl, _ = struct.unpack_from("<IIII", mm, pos)
            start = pos + 16
            pos = start + incl
            if pos > size:
                skipped["truncated_file"] += 1
                break
            if incl < 14:
                skipped["short_frame"] += 1
                continue
            ethertype, off = struct.unpack_from(">H", mm, start + 12)[0], 14
            if ethertype == 0x8100 and incl >= 18:
                ethertype, off = struct.unpack_from(">H", mm, start + 16)[0], 18
            if ethertype != 0x0800:
                skipped["non_ipv4"] += 1
                continue
            if incl < off + 20 or mm[start + off + 9] != 6:
                skipped["non_tcp"] += 1
                continue
            if mm[start + off] >> 4 != 4:
                skipped["bad_ip_version"] += 1
                continue
            ihl = (mm[start + off] & 0x0F) * 4
            total = struct.unpack_from(">H", mm, start + off + 2)[0]
            if struct.unpack_from(">H", mm, start + off + 6)[0] & 0x3FFF:
                skipped["fragment"] += 1
                continue
            tcp = start + off + ihl
            if ihl < 20 or incl < off + ihl + 20:
                skipped["short_tcp"] += 1
                continue
            sport, dport, seq = struct.unpack_from(">HHI", mm, tcp)
            data_off = (mm[tcp + 12] >> 4) * 4
            payload = total - ihl - data_off
            if data_off < 20:
                skipped["bad_tcp_offset"] += 1
                continue
            if payload < 0:
                skipped["bad_length"] += 1
                continue
            yield (ts_s + ts_u / 1e6, socket.inet_ntoa(mm[start + off + 12:start + off + 16]), sport,
                   socket.inet_ntoa(mm[start + off + 16:start + off + 20]), dport, mm[tcp + 13], seq,
                   payload), skipped


def record_segment(flow, direction, seq, length):
    """페이로드 구간 [seq, seq+length)를 기록한다. 새 바이트가 하나라도 있으면 True, 이미 받은 구간만
    다시 온 것(완전한 재전송)이면 False.

    순서 뒤바뀜(앞 구간이 늦게 도착)과 재전송은 다르다. 이미 받은 구간들의 합집합으로 판단하므로,
    겹치지 않는 늦은 구간은 새 패킷으로 센다. 일부만 겹치는 구간도 새 바이트가 있어 센다.
    순서 번호는 방향별 첫 값을 기준으로 부호 있는 상대 위치로 바꿔 32비트 순환을 처리한다.
    """
    reference = flow.seq_ref.setdefault(direction, seq)
    start = ((seq - reference + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    end = start + length
    ranges = flow.ranges.setdefault(direction, [])
    for first, last in ranges:
        if first <= start and end <= last:
            return False
    merged, kept = [start, end], []
    for first, last in ranges:
        if first <= merged[1] and last >= merged[0]:
            merged = [min(first, merged[0]), max(last, merged[1])]
        else:
            kept.append([first, last])
    kept.append(merged)
    kept.sort()
    flow.ranges[direction] = kept
    return True


def assemble_flows(path):
    """443 포트 TCP 패킷을 플로우로 묶는다. (flows, skipped, total_tcp_packets) 반환.

    플로우 경계: 마지막 패킷 후 IDLE_TIMEOUT 초 경과, 또는 이미 페이로드가 오갔거나 FIN/RST로 끝난
    4-튜플에 SYN(ACK 없음)이 다시 도착하면 새 플로우. 같은 방향에서 이미 받은 바이트 구간만 다시 온
    패킷(완전한 재전송)은 세지 않고, 순서가 뒤바뀌어 늦게 온 새 구간은 센다(record_segment).
    SYN 없이 시작한 플로우는 midstream 으로 표시한다.
    """
    flows, active, tcp_packets, skipped = [], {}, 0, {}
    for packet, skipped in iter_tcp_packets(path):
        ts, src, sport, dst, dport, flags, seq, plen = packet
        tcp_packets += 1
        if dport == SERVER_PORT:
            key, direction = (src, sport, dst), 1
        elif sport == SERVER_PORT:
            key, direction = (dst, dport, src), -1
        else:
            continue
        flow = active.get(key)
        is_syn = bool(flags & SYN) and not flags & ACK
        if flow and (ts - flow.last_ts > IDLE_TIMEOUT
                     or (is_syn and (flow.payload_packets or flow.closed))):
            flows.append(flow)
            flow = None
        if flow is None:
            flow = active[key] = Flow(key[0], key[1], key[2], ts, ts,
                                      midstream=not (is_syn and direction == 1))
        flow.packets += 1
        flow.last_ts = ts
        if flags & (FIN | RST):
            flow.closed = True
        if plen > 0:
            if not record_segment(flow, direction, seq, plen):
                flow.retransmissions += 1
                continue
            flow.payload_packets += 1
            if len(flow.seq) < K_MAX:
                flow.seq.append(direction * plen)
                flow.times.append(round(ts - flow.start_ts, 6))
    flows.extend(active.values())
    flows.sort(key=lambda flow: (flow.start_ts, flow.client_port))
    return flows, dict(skipped), tcp_packets


def label_for(flow, capture):
    attack = flow.client_ip == capture["attacker"] and flow.server_ip == capture["victim"]
    return capture["attack"] if attack else "benign"


def prefix_input(flow, k):
    """앞 k개 값만 남기고 나머지는 0. 30칸 구조를 유지한다 (A의 접두 평가와 같은 방식)."""
    values = list(flow.seq) + [0] * (K_MAX - len(flow.seq))
    return values[:k] + [0] * (K_MAX - k)


def prefix_report(rows, ks=KS):
    """(캡처, 라벨)·k별 rows / reached_k_rows / short_rows / short_fraction / padding_policy.

    분모는 해당 (캡처, 라벨)의 전체 플로우 수다. 비영 칸 수가 아니라 실제 페이로드 패킷 수
    (payload_packets)로 도달 여부를 판단하므로 30개를 넘는 흐름도 정확히 센다.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pcap"], row["label"])].append(row["payload_packets"])
    report = []
    for (pcap, label), counts in sorted(groups.items()):
        for k in ks:
            reached = sum(count >= k for count in counts)
            short = len(counts) - reached
            report.append(dict(pcap=pcap, label=label, k=k, rows=len(counts), reached_k_rows=reached,
                               short_rows=short, short_fraction=round(short / len(counts), 6),
                               padding_policy=PADDING_POLICY))
    return report


def flow_rows(flows, pcap_name, capture):
    rows = []
    for index, flow in enumerate(flows):
        row = dict(pcap=pcap_name, day=capture["day"], flow_id=index, label=label_for(flow, capture),
                   client_ip=flow.client_ip, client_port=flow.client_port, server_ip=flow.server_ip,
                   start_ts=round(flow.start_ts, 6), duration=round(flow.last_ts - flow.start_ts, 6),
                   packets=flow.packets, payload_packets=flow.payload_packets,
                   retransmissions=flow.retransmissions, midstream=int(flow.midstream),
                   closed=int(flow.closed))
        padded = list(flow.seq) + [0] * (K_MAX - len(flow.seq))
        times = list(flow.times) + [""] * (K_MAX - len(flow.times))
        row.update({f"seq_{i}": padded[i] for i in range(K_MAX)})
        row.update({f"t_{i}": times[i] for i in range(K_MAX)})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR, MACHINE

    root = Path(__file__).resolve().parents[2]
    source = DATA_DIR / "raw" / "hikari"
    out_data = DATA_DIR / "processed" / f"track_b_{args.run_id}"
    out_report = root / "outputs/results" / f"track_b_extract_{args.run_id}"
    if out_data.exists() or out_report.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    all_rows, summary = [], []
    for name, capture in CAPTURES.items():
        flows, skipped, tcp_packets = assemble_flows(source / name)
        rows = flow_rows(flows, name, capture)
        all_rows.extend(rows)
        labels = defaultdict(int)
        for row in rows:
            labels[row["label"]] += 1
        # 라벨 규칙 점검: 피해자로 가는 다른 클라이언트 플로우는 benign 으로 들어간다(정책은 문서에 명시)
        other_victim = sum(1 for r in rows if r["server_ip"] == capture["victim"]
                           and r["client_ip"] != capture["attacker"])
        summary.append(dict(pcap=name, tcp_packets=tcp_packets, flows=len(flows), labels=dict(labels),
                            skipped=skipped, midstream=sum(r["midstream"] for r in rows),
                            retransmissions=sum(r["retransmissions"] for r in rows),
                            victim_flows_from_other_clients=other_victim,
                            pcap_sha256=sha256_file(source / name)))
        print(summary[-1], flush=True)
    out_data.mkdir(parents=True)
    out_report.mkdir(parents=True)
    with (out_data / "flows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = prefix_report(all_rows)
    with (out_report / "prefix_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8").strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tests"], cwd=root,
                                             text=True, encoding="utf-8").strip())
    metadata = dict(run_id=args.run_id, machine=MACHINE, captures=CAPTURES, idle_timeout=IDLE_TIMEOUT,
                    date=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                    flows_sha256=sha256_file(out_data / "flows.csv"),
                    code_sha256={"src/track_b/pcap_flows.py": sha256_file(Path(__file__))},
                    git_commit=git_commit, git_dirty_src_tests=git_dirty,
                    retransmission_policy="완전히 이미 받은 바이트 구간만 재전송으로 제외(record_segment). "
                                          "순서가 뒤바뀐 새 구간·일부 겹침은 센다",
                    k_max=K_MAX, ks=list(KS), padding_policy=PADDING_POLICY, summary=summary,
                    model_input_columns=[f"seq_{i}" for i in range(K_MAX)],
                    identifier_columns=["client_ip", "client_port", "server_ip", "pcap", "day", "flow_id"],
                    note="flows.csv는 DATA_DIR에만 저장. 식별자·라벨·시간 컬럼은 모델 입력이 아니다")
    (out_report / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print("DONE", out_data, out_report, flush=True)


if __name__ == "__main__":
    main()
