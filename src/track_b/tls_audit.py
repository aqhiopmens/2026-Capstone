"""앞 칸(seq_0~)이 실제 TLS 레코드와 맞는지 표본 플로우로 확인하는 감사(안건 8 부가, 마스킹 결과와 분리).

표본: 그룹(2021 04-11 .242.104 공격·정상, 2021 04-16 .88.81 공격·.242.104 공격, 2022 공격)마다 첫 공격 플로우 시작 후 20분 창 안의 payload>=10 플로우에서
고정 시드로 5개씩 뽑는다. 로컬 pcap을 창 구간만 읽어 TCP 페이로드 바이트를 방향별로 재조립하고 TLS 레코드(5바이트 헤더)와 핸드셰이크 메시지를 해석한다.
검증 1: 재조립한 패킷별 부호 있는 길이가 flows.csv의 seq_0~와 일치해야 같은 플로우로 본다(불일치는 '식별 불일치'로 따로 센다).
검증 2: 각 패킷이 레코드 경계에서 시작하는지, 여러 레코드를 담는지, 레코드 중간(분할)인지를 적는다. 길이 일치만으로 메시지 종류를 확정하지 않고, 경계가 맞지 않는 사례는 '미확인'으로 센다.
TLS 1.3처럼 ServerHello 뒤가 암호화돼 type 23으로 보이는 레코드는 안쪽 메시지 종류를 알 수 없으므로 'opaque'로 둔다.

Run: uv run python -m src.track_b.tls_audit --flows-run-id 2026-11-02_v1 --flows-2022-run-id 2026-11-05_v2 --run-id <ID>
"""

import argparse
import json
import mmap
import socket
import struct
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.results import sha256_file
from src.track_b.https_flows import ATTACK, BENIGN

SEED, PER_GROUP, WINDOW_S, MIN_PAYLOAD, FIRST_N = 0, 5, 1200.0, 10, 8
HS_TYPES = {1: "ClientHello", 2: "ServerHello", 4: "NewSessionTicket", 8: "EncryptedExtensions", 11: "Certificate", 12: "ServerKeyExchange",
            13: "CertificateRequest", 14: "ServerHelloDone", 15: "CertificateVerify", 16: "ClientKeyExchange", 20: "Finished"}
REC_TYPES = {20: "ChangeCipherSpec", 21: "Alert", 22: "Handshake", 23: "AppData"}
GROUPS = [("2021-04-11", "Sunday_2021-04-11_2154.pcap", "attack_242104"), ("2021-04-11", "Sunday_2021-04-11_2154.pcap", "benign"),
          ("2021-04-16", "Friday_2021-04-16_2304.pcap", "attack_88_81"), ("2021-04-16", "Friday_2021-04-16_2304.pcap", "attack_242104"),
          ("2022-04-10", "Sunday_2022-04-10_2335_BRUTEFORCE_HTTPS_150s.pcap", "attack_2022")]


def parse_records(stream):
    """방향별 바이트 스트림을 TLS 레코드 목록으로. 각 항목: (시작 오프셋, 레코드 종류, 레코드 길이, 핸드셰이크 메시지 이름 목록, 본문이 스트림 안에 다 있는지). 헤더가 비정상이면 중단한다."""
    out, pos = [], 0
    while pos + 5 <= len(stream):
        rtype, ver, length = stream[pos], stream[pos + 1], struct.unpack_from(">H", stream, pos + 3)[0]
        if rtype not in REC_TYPES or ver != 3 or length > 18432:
            break
        msgs = []
        if rtype == 22:  # 평문 핸드셰이크 레코드 안의 메시지(여러 개 가능, 잘리면 앞부분만)
            p, end = pos + 5, min(pos + 5 + length, len(stream))
            while p + 4 <= end:
                msgs.append(HS_TYPES.get(stream[p], f"hs{stream[p]}"))
                p += 4 + int.from_bytes(stream[p + 1:p + 4], "big")
        out.append((pos, REC_TYPES[rtype], length, msgs, pos + 5 + length <= len(stream)))
        pos += 5 + length
    return out


def reassemble(packets, stats=None):
    """패킷 목록[(방향, tcp seq, 바이트)]에서 방향별 스트림과 패킷별 (방향, 스트림 오프셋, 길이)를 만든다. 이미 받은 구간만 다시 온 패킷은 버린다.
    첫 패킷보다 seq가 앞선(재정렬) 패킷은 가장 앞선 seq를 스트림 0으로 잡아 제자리에 놓고 stats['pre_base']로 센다. 받지 못한 구간(gap)은 0으로 채워지므로 gap_bytes로 따로 센다."""
    streams, placed, first, low = {1: bytearray(), -1: bytearray()}, [], {}, {}
    seen = {1: [], -1: []}
    stats = stats if stats is not None else {}
    rel = lambda d, seq: ((seq - first[d] + 2**31) & 0xFFFFFFFF) - 2**31  # 첫 패킷 기준 부호 있는 오프셋(seq 되감김 처리)
    for d, seq, data in packets:
        first.setdefault(d, seq)
        low[d] = min(low.get(d, 0), rel(d, seq))
    stats["pre_base"] = sum(rel(d, seq) < 0 for d, seq, _ in packets)  # 도착 순서상 첫 패킷보다 앞 seq인 재정렬 패킷 수
    for d, seq, data in packets:
        off = rel(d, seq) - low[d]  # 가장 앞선 seq를 스트림 0으로
        if any(off >= a and off + len(data) <= b for a, b in seen[d]):
            continue
        seen[d].append((off, off + len(data)))
        buf = streams[d]
        if len(buf) < off + len(data):
            buf.extend(bytes(off + len(data) - len(buf)))
        buf[off:off + len(data)] = data
        placed.append((d, off, len(data)))
    stats["gap_bytes"] = sum(len(streams[d]) - _covered(seen[d]) for d in seen)
    return {d: bytes(b) for d, b in streams.items()}, placed


def _covered(spans):
    """겹치는 구간을 합쳐 덮인 바이트 수를 센다."""
    total, end = 0, 0
    for a, b in sorted(spans):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def describe_packet(placed_one, records):
    """한 패킷이 담은 레코드 설명. 반환: (설명 문자열, 경계 상태). 겹치는 레코드가 없거나 첫 겹침 레코드가 패킷 시작과 어긋나면 '미확인'이다."""
    d, off, length = placed_one
    start, end = off, off + length
    parts, starts_here, spans_earlier = [], False, False
    for pos, kind, rlen, msgs, complete in records.get(d, []):
        rec_end = pos + 5 + rlen
        if rec_end <= start or pos >= end:
            continue
        starts_here |= pos == start
        spans_earlier |= pos < start
        label = kind + (f"[{'+'.join(msgs)}]" if msgs else "")
        parts.append(f"{label} len{rlen}" + ("(이어짐)" if rec_end > end else "") + ("" if complete else "(스트림 끝에서 잘림)"))
    boundary = "레코드 중간에서 시작(분할)" if spans_earlier else "경계 시작" if starts_here else "미확인"
    return ("; ".join(parts) if parts else "해석 불가(레코드 밖)"), boundary


def read_window(path, t0, t1, keys):
    """pcap에서 [t0, t1] 구간의 443 TCP 페이로드 패킷을 플로우 키별로 모은다. 키 = (client_ip, client_port, server_ip)."""
    got = {k: [] for k in keys}
    with open(path, "rb") as h, mmap.mmap(h.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        pos, size = 24, len(mm)
        while pos + 16 <= size:
            ts_s, ts_u, incl, _ = struct.unpack_from("<IIII", mm, pos)
            start, ts = pos + 16, ts_s + ts_u / 1e6
            pos = start + incl
            if ts < t0:
                continue
            if ts > t1:
                break
            et, off = struct.unpack_from(">H", mm, start + 12)[0], 14
            if et == 0x8100:
                et, off = struct.unpack_from(">H", mm, start + 16)[0], 18
            if et != 0x0800 or mm[start + off + 9] != 6:
                continue
            ihl = (mm[start + off] & 0x0F) * 4
            total = struct.unpack_from(">H", mm, start + off + 2)[0]
            tcp = start + off + ihl
            sport, dport, seq = struct.unpack_from(">HHI", mm, tcp)
            dlen = (mm[tcp + 12] >> 4) * 4
            payload = total - ihl - dlen
            if payload <= 0:
                continue
            src, dst = socket.inet_ntoa(mm[start + off + 12:start + off + 16]), socket.inet_ntoa(mm[start + off + 16:start + off + 20])
            key, d = ((src, sport, dst), 1) if dport == 443 else (((dst, dport, src), -1) if sport == 443 else (None, 0))
            if key in got:  # (시각, 방향, tcp seq, 페이로드, 원본 pcap 레코드 헤더+프레임)
                got[key].append((ts, d, seq, bytes(mm[tcp + dlen:tcp + dlen + payload]), bytes(mm[start - 16:start + incl])))
        header = bytes(mm[:24])
    return got, header


def pick(flows, group):
    day, pcap, kind = group
    f = flows[(flows.pcap == pcap) & (flows.payload_packets >= MIN_PAYLOAD)]
    atk = f[f.label == ATTACK]
    if kind == "attack_242104":
        f = f[(f.server_ip == "128.199.242.104")]
        atk = f[f.label == ATTACK]
    elif kind == "attack_88_81":
        atk = atk[atk.server_ip == "128.199.88.81"]
    t0 = float(atk.start_ts.min())
    window = f[(f.start_ts >= t0) & (f.start_ts <= t0 + WINDOW_S)]
    pool = window[window.label == BENIGN] if kind == "benign" else window[(window.label == ATTACK) & window.index.isin(atk.index)]
    return t0, pool.sample(n=min(PER_GROUP, len(pool)), random_state=SEED).sort_values("start_ts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--flows-2022-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_tls_audit_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    p21 = DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"
    p22 = DATA_DIR / "processed" / f"track_b_https2022_{args.flows_2022_run_id}" / "flows.csv"
    cols = ["pcap", "day", "flow_id", "label", "client_ip", "client_port", "server_ip", "start_ts", "duration", "payload_packets"] + [f"seq_{i}" for i in range(30)]
    flows = pd.concat([pd.read_csv(p21, usecols=cols, keep_default_na=False), pd.read_csv(p22, usecols=cols, keep_default_na=False)], ignore_index=True)
    sel_rows, pkt_rows, excerpts = [], [], {}
    arows = DATA_DIR / "audit" / f"track_b_tls_audit_{args.run_id}"  # 행별 자료·pcap 발췌는 Git 밖(DATA_DIR/Drive)
    arows.mkdir(parents=True)
    for group in GROUPS:
        t0, sample = pick(flows, group)
        keys = [(r.client_ip, r.client_port, r.server_ip) for r in sample.itertuples()]
        packets, header = read_window(DATA_DIR / "raw" / "hikari" / group[1], t0, t0 + WINDOW_S + 600, keys)
        raw_frames = []
        for r in sample.itertuples():
            # 연결 범위 고정: 이 플로우의 [시작-1초, 시작+지속시간+2초] 안의 패킷만 쓴다(같은 3-tuple의 다른 연결 제외, 4-tuple에 서버 포트 443 고정)
            lo, hi = r.start_ts - 1.0, r.start_ts + float(r.duration) + 2.0
            pk = sorted((x for x in packets[(r.client_ip, r.client_port, r.server_ip)] if lo <= x[0] <= hi), key=lambda x: x[0])
            raw_frames += [x[4] for x in pk]
            stats = {}
            streams, placed = reassemble([x[1:4] for x in pk], stats)
            records = {d: parse_records(s) for d, s in streams.items()}
            mine = [d * n for d, _, n in placed][:FIRST_N]
            theirs = [int(getattr(r, f"seq_{i}")) for i in range(FIRST_N)]
            same = mine == theirs
            clean = same and stats["gap_bytes"] == 0 and stats["pre_base"] == 0
            sel_rows.append(dict(group=f"{group[0]}_{group[2]}", pcap=group[1], flow_id=r.flow_id, label=r.label, server_ip=r.server_ip,
                                 start_ts=r.start_ts, payload_packets=r.payload_packets, lengths_match_flows_csv=same,
                                 gap_bytes=stats["gap_bytes"], pre_base_packets=stats["pre_base"], interpretable=clean))
            for i, pl in enumerate(placed[:FIRST_N]):
                desc, boundary = describe_packet(pl, records)
                pkt_rows.append(dict(group=f"{group[0]}_{group[2]}", flow_id=r.flow_id, idx=i, direction="C->S" if pl[0] == 1 else "S->C",
                                     signed_length=pl[0] * pl[2], flows_csv_value=theirs[i] if i < len(theirs) else None,
                                     records=desc, boundary=boundary, flow_identified=same, flow_interpretable=clean))
        excerpts[group[2] + "_" + group[0]] = arows / f"excerpt_{group[0]}_{group[2]}.pcap"
        excerpts[group[2] + "_" + group[0]].write_bytes(header + b"".join(raw_frames))
    out.mkdir(parents=True)
    sel, pk = pd.DataFrame(sel_rows), pd.DataFrame(pkt_rows)
    sel.to_csv(arows / "selected_flows.csv", index=False)
    pk.to_csv(arows / "packets.csv", index=False)
    ident = pk[pk.flow_interpretable]
    summ = (ident[ident.idx < 3].groupby(["group", "idx"]).apply(lambda g: pd.Series(dict(
        packets=len(g), boundary_start=int((g.boundary == "경계 시작").sum()), split=int(g.boundary.str.startswith("레코드 중간").sum()),
        unconfirmed=int((g.boundary == "미확인").sum()),
        records_top=g.records.str.replace(r" len\d+", "", regex=True).value_counts().head(2).to_dict())),
        include_groups=False).reset_index())
    summ.to_csv(out / "first3_summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, seed=SEED, per_group=PER_GROUP, window_s=WINDOW_S, min_payload=MIN_PAYLOAD, first_n=FIRST_N,
        flows_sha256=sha256_file(p21), flows_2022_sha256=sha256_file(p22), code_sha256={"src/track_b/tls_audit.py": sha256_file(Path(__file__))},
        flows_selected=len(sel), identified=int(sel.lengths_match_flows_csv.sum()), interpretable=int(sel.interpretable.sum()),
        row_level_files={n: dict(path=str((arows / n).relative_to(DATA_DIR)), sha256=sha256_file(arows / n)) for n in ("selected_flows.csv", "packets.csv")},
        pcap_excerpts={k: dict(path=str(v.relative_to(DATA_DIR)), sha256=sha256_file(v)) for k, v in excerpts.items()},
        connection_scope="플로우별 [시작-1초, 시작+지속시간+2초] 패킷만, 페이로드>0 패킷만 발췌(SYN·ACK 없음)",
        note="표본 20~25개(첫 공격 후 20분 창, 고정 시드). 레코드 해석은 방향별 재조립 바이트의 TLS 헤더 기준이며 길이 일치만으로 메시지 종류를 확정하지 않는다. "
             "경계 집계는 길이 일치·gap 0·재정렬 0인 플로우(interpretable)만 쓴다. '미확인'은 경계로 세지 않는다. 표본이 작고 시간 창이 한정돼 대표성이 제한적이다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(sel.groupby("group").lengths_match_flows_csv.agg(["sum", "count"]))
    print(pk[pk.idx < 3].groupby(["group", "idx", "direction"]).records.agg(lambda s: s.str.replace(r" len\d+", "", regex=True).value_counts().head(2).to_dict()).to_string())
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
