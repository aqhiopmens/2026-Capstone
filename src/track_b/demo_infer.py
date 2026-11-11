"""시연 추론: 저장된 pcap을 읽어 연결마다 앞 10개 payload 패킷만으로 4클래스를 판정하고, 판정 가능 여부·정답 매칭·채점 분모를 기록한다.

상태는 midstream을 먼저 가른다(SYN 없이 시작한 연결은 나중에 RST·FIN을 봐도 채점에서 제외). 10개 미만 연결은 RST 또는 양방향 FIN이 관측된 경우에만
종료 확인 뒤 0패딩 판정을 하고, 그 관측 대기시간은 연결 시작부터 종료 증거를 확인한 시점까지다. 프로그램 처리시간과 구분한다.
"""

import hashlib
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.track_b.demo_model import K, class_probs
from src.track_b.four_class import CLASSES
from src.track_b.pcap_flows import FIN, RST, SERVER_PORT, assemble_flows, iter_tcp_packets, prefix_input

SCORABLE = ("reached_k", "short_closed_padded")


def classify_status(midstream, payload_packets, rst, fin_dirs):
    if midstream:
        return "midstream_excluded"
    if payload_packets >= K:
        return "reached_k"
    if rst or {1, -1} <= set(fin_dirs):
        return "short_closed_padded"
    return "short_unconfirmed"


def events_from_packets(packets):
    """패킷 튜플(iter_tcp_packets와 같은 형식)에서 연결 키별 FIN/RST 사건 (ts, 방향, flags)를 모은다. 방향 +1 = 클라이언트→서버."""
    events = defaultdict(list)
    for ts, src, sport, dst, dport, flags, _seq, _plen in packets:
        if dport == SERVER_PORT:
            key, direction = (src, sport, dst), 1
        elif sport == SERVER_PORT:
            key, direction = (dst, dport, src), -1
        else:
            continue
        if flags & (FIN | RST):
            events[key].append((ts, direction, flags))
    return events


def close_evidence(flow, events):
    """그 연결의 시간 창 안에서 (RST를 봤는가, FIN을 본 방향 집합, 종료 증거를 확인한 시각 또는 None). 같은 4-튜플의 다른 연결 사건은 섞지 않는다."""
    rst, fin_first, t_rst = False, {}, None
    for ts, direction, flags in sorted(events.get((flow.client_ip, flow.client_port, flow.server_ip), [])):
        if ts < flow.start_ts or ts > flow.last_ts:
            continue
        if flags & RST:
            rst, t_rst = True, ts if t_rst is None else t_rst
        if flags & FIN:
            fin_first.setdefault(direction, ts)
    t_fin = max(fin_first.values()) if len(fin_first) == 2 else None
    candidates = [t for t in (t_rst, t_fin) if t is not None]
    return rst, set(fin_first), (min(candidates) if candidates else None)


def observation_wait(flow, status, t_close):
    if status == "reached_k":
        return flow.times[K - 1]
    if status == "short_closed_padded" and t_close is not None:
        return t_close - flow.start_ts
    return float("nan")


def model_inputs(flows):
    """모델 입력: 각 연결의 앞 10개 payload 패킷 길이(부호=방향), 나머지 20칸은 0. 패킷 수·종료 여부 등 뒤 정보는 쓰지 않는다."""
    return np.array([prefix_input(f, K) for f in flows])


def match_to_original(conn, table, pcap_sha256):
    """대응표(축소 pcap의 SHA에 묶임)로 연결을 원본 시험 행에 매칭한다. 키 = 클라이언트 ip·port, 서버 ip, 시작 시각. 후보가 여러 개면 채점 제외(ambiguous)."""
    if not (table.pcap_sha256 == pcap_sha256).all():
        raise ValueError("대응표가 이 pcap(SHA)용이 아닙니다")
    key = ["client_ip", "client_port", "server_ip", "start_ts"]
    left = conn.assign(start_ts=conn.start_ts.round(6))
    extra = [c for c in ("orig_payload_packets",) if c in table]
    right = table.assign(start_ts=table.start_ts.round(6))[key + ["orig_flow_id", "label"] + extra]
    counts = right.groupby(key).size().rename("n_candidates").reset_index()
    merged = left.merge(counts, on=key, how="left").fillna({"n_candidates": 0})
    one = right.drop_duplicates(key, keep=False)
    merged = merged.merge(one, on=key, how="left")
    merged["match"] = np.select([merged.n_candidates == 1, merged.n_candidates > 1], ["matched", "ambiguous"], "unmatched")
    return merged.rename(columns={"label": "actual"}).drop(columns="n_candidates").assign(start_ts=conn.start_ts.to_numpy())


def score_connections(conn):
    """채점 분모: 전체 연결(payload>0) / 정답 매칭 / 실제 채점(판정 가능 상태 ∧ 매칭) / 제외·미도달 건수. 혼동행렬은 표시 순서(CLASSES)."""
    scored = conn[conn.status.isin(SCORABLE) & (conn.match == "matched")]
    cm = pd.crosstab(scored.actual, scored.predicted).reindex(index=list(CLASSES), columns=list(CLASSES), fill_value=0)
    benign = scored[scored.actual == "benign"]
    short = conn.status.isin(("short_closed_padded", "short_unconfirmed"))
    return dict(total=len(conn), matched=int((conn.match == "matched").sum()), scored=len(scored), unmatched=int((conn.match == "unmatched").sum()),
                ambiguous=int((conn.match == "ambiguous").sum()), excluded_midstream=int((conn.status == "midstream_excluded").sum()),
                excluded_short_unconfirmed=int((conn.status == "short_unconfirmed").sum()), short_closed_padded=int((conn.status == "short_closed_padded").sum()),
                short_total=int(short.sum()), short_ratio=float(short.mean()) if len(conn) else float("nan"), benign_scored=len(benign),
                benign_false_alarms=int((benign.predicted != "benign").sum()), confusion=cm)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_pcap(path, model):
    """pcap → 연결 표(payload>0)와 처리시간. 판정 가능한 연결(reached_k·short_closed_padded)에만 예측·확률을 붙이고 정답은 붙이지 않는다."""
    t0 = time.perf_counter()
    flows, skipped, tcp_packets = assemble_flows(path)
    flows = [f for f in flows if f.payload_packets > 0]
    events = events_from_packets(p for p, _ in iter_tcp_packets(path))
    t1 = time.perf_counter()
    rows = []
    for f in flows:
        rst, fin_dirs, t_close = close_evidence(f, events)
        status = classify_status(f.midstream, f.payload_packets, rst, fin_dirs)
        rows.append(dict(client_ip=f.client_ip, client_port=f.client_port, server_ip=f.server_ip, start_ts=f.start_ts, payload_packets=f.payload_packets,
                         midstream=f.midstream, status=status, observation_wait_s=observation_wait(f, status, t_close), rst=rst, fin_directions=len(fin_dirs)))
    conn = pd.DataFrame(rows)
    x = pd.DataFrame(model_inputs(flows), columns=[f"seq_{i}" for i in range(30)])
    for i in range(K):
        conn[f"len_{i}"] = x[f"seq_{i}"].to_numpy()  # 화면 표시용: 모델이 본 앞 10개 값
    t2 = time.perf_counter()
    probs = class_probs(model, x)
    ok = conn.status.isin(SCORABLE).to_numpy()
    conn["predicted"] = np.where(ok, probs.idxmax(axis=1).to_numpy(), None)
    for c in probs.columns:
        conn[f"p_{c}"] = np.where(ok, probs[c].to_numpy(), np.nan)
    t3 = time.perf_counter()
    timing = dict(parse_and_assemble_s=t1 - t0, features_s=t2 - t1, predict_s=t3 - t2, total_s=t3 - t0, connections=len(conn), tcp_packets=tcp_packets, skipped=skipped)
    return conn, timing
