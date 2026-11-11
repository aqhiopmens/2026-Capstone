import numpy as np
import pandas as pd
import pytest

from src.track_b.demo_infer import (classify_status, close_evidence, events_from_packets, match_to_original, model_inputs, observation_wait, score_connections)
from src.track_b.pcap_flows import ACK, FIN, RST, SYN, Flow

C, S = ("10.0.0.2", 50000, "10.0.0.1"), "10.0.0.1"


def flow(payload=12, midstream=False, start=100.0, seq=None, times=None, port=50000):
    seq = list(seq if seq is not None else [(-1) ** i * (100 + i) for i in range(min(payload, 30))])
    times = list(times if times is not None else [round(0.01 * (i + 1), 6) for i in range(len(seq))])
    return Flow("10.0.0.2", port, S, start, start + 1.0, packets=payload + 3, payload_packets=payload, midstream=midstream, seq=seq, times=times)


def pkt(ts, direction, flags, sport=50000):
    return (ts, "10.0.0.2", sport, S, 443, flags, 0, 0) if direction == 1 else (ts, S, 443, "10.0.0.2", sport, flags, 0, 0)


def test_status_checks_midstream_first_then_payload_count_then_closure_evidence():
    assert classify_status(midstream=True, payload_packets=15, rst=True, fin_dirs={1, -1}) == "midstream_excluded"   # 나중에 RST·FIN을 봐도 제외
    assert classify_status(False, 10, False, set()) == "reached_k"
    assert classify_status(False, 9, True, set()) == "short_closed_padded"                                         # RST
    assert classify_status(False, 9, False, {1, -1}) == "short_closed_padded"                                      # 양방향 FIN
    assert classify_status(False, 9, False, {1}) == "short_unconfirmed"                                            # FIN 하나뿐
    assert classify_status(False, 9, False, set()) == "short_unconfirmed"                                          # 종료 증거 없음


def test_close_evidence_from_packets_per_flow_window_and_time():
    f = flow(payload=5, start=100.0)
    f.last_ts = 105.0
    other = flow(payload=5, start=300.0)                                                                           # 같은 4-튜플의 다른 연결(포트 재사용)
    other.last_ts = 305.0
    ev = events_from_packets([pkt(100.0, 1, SYN), pkt(101.0, 1, FIN | ACK), pkt(103.0, -1, FIN | ACK), pkt(301.0, -1, RST)])
    rst, fin_dirs, t = close_evidence(f, ev)
    assert (rst, fin_dirs, t) == (False, {1, -1}, 103.0)                                                           # 두 번째 방향 FIN을 본 시각
    rst2, fin2, t2 = close_evidence(other, ev)
    assert (rst2, fin2, t2) == (True, set(), 301.0)                                                                # 첫 연결의 FIN은 두 번째 연결에 섞이지 않는다
    assert close_evidence(flow(payload=5, port=50001), ev) == (False, set(), None)


def test_observation_wait_is_kth_packet_time_or_time_to_closure_evidence_for_padded():
    assert observation_wait(flow(payload=12), "reached_k", None) == pytest.approx(0.10)                           # 10번째 payload 패킷 시각(times[9])
    short = flow(payload=4)
    assert observation_wait(short, "short_closed_padded", 103.0) == pytest.approx(3.0)                            # 시작 ~ 종료 증거 확인 시점
    assert np.isnan(observation_wait(short, "short_unconfirmed", None))


def test_model_input_uses_only_first_ten_payload_packets_leakage_invariance():
    a = flow(payload=12, seq=[(-1) ** i * (100 + i) for i in range(30)])
    b = flow(payload=29, seq=[(-1) ** i * (100 + i) for i in range(10)] + [7] * 20)                                # 뒷부분·패킷 수가 달라도
    xa, xb = model_inputs([a, b])
    assert (xa == xb).all() and (xa[10:] == 0).all() and list(xa[:10]) == [(-1) ** i * (100 + i) for i in range(10)]


def test_match_to_original_requires_same_pcap_sha_and_exact_unique_key():
    conn = pd.DataFrame(dict(client_ip=["1.1.1.1", "2.2.2.2", "3.3.3.3"], client_port=[1, 2, 3], server_ip=["9.9.9.9"] * 3, start_ts=[10.0, 20.0, 30.0]))
    table = pd.DataFrame(dict(pcap_sha256=["abc"] * 4, client_ip=["1.1.1.1", "2.2.2.2", "2.2.2.2", "3.3.3.3"], client_port=[1, 2, 2, 3],
                              server_ip=["9.9.9.9"] * 4, start_ts=[10.0, 20.0, 20.0, 31.0], orig_flow_id=[5, 6, 7, 8], label=["benign", "benign", "probing", "benign"]))
    m = match_to_original(conn, table, pcap_sha256="abc")
    assert m.match.tolist() == ["matched", "ambiguous", "unmatched"]                                              # 후보 여러 개는 채점 제외, 시작 시각이 다르면 실패
    assert m.actual.tolist()[0] == "benign" and pd.isna(m.actual.tolist()[1])
    with pytest.raises(ValueError):                                                                               # 다른 파일에 대응표를 적용하면 거절
        match_to_original(conn, table, pcap_sha256="zzz")


def test_score_connections_denominators_and_confusion():
    conn = pd.DataFrame(dict(
        status=["reached_k", "reached_k", "short_closed_padded", "short_unconfirmed", "midstream_excluded", "reached_k", "reached_k"],
        match=["matched", "matched", "matched", "matched", "matched", "unmatched", "ambiguous"],
        actual=["benign", "benign", "probing", "benign", "benign", None, None],
        predicted=["benign", "probing", "probing", "benign", "benign", "benign", "benign"]))
    s = score_connections(conn)
    assert (s["total"], s["matched"], s["scored"]) == (7, 5, 3)                                                    # 채점 = 판정 가능 상태 ∧ 정답 매칭
    assert (s["excluded_midstream"], s["excluded_short_unconfirmed"], s["unmatched"], s["ambiguous"]) == (1, 1, 1, 1)
    assert s["short_total"] == 2 and s["short_closed_padded"] == 1 and s["short_ratio"] == pytest.approx(2 / 7)
    assert (s["benign_scored"], s["benign_false_alarms"]) == (2, 1)                                               # 정상 오탐 건수/분모
    cm = s["confusion"]                                                                                             # 행 = 실제, 열 = 예측, 표시 순서 CLASSES
    assert list(cm.index) == ["benign", "probing", "bruteforce_xml", "bruteforce_https"] and cm.loc["benign", "probing"] == 1 and cm.loc["probing", "probing"] == 1


def test_input_mismatch_between_cut_and_original_is_excluded_from_scoring_and_counted():
    from src.track_b.demo_infer import K

    lens = {f"len_{i}": [i + 1, i + 1] for i in range(K)}
    conn = pd.DataFrame(dict(client_ip=["1.1.1.1", "2.2.2.2"], client_port=[1, 2], server_ip=["9.9.9.9"] * 2, start_ts=[10.0, 20.0],
                             status=["reached_k", "reached_k"], predicted=["benign", "benign"], **lens))
    orig = {f"orig_seq_{i}": [i + 1, i + 1] for i in range(K)}
    orig["orig_seq_3"] = [4, 999]                                                                                  # 두 번째 연결만 원본 입력과 다르다
    table = pd.DataFrame(dict(pcap_sha256=["abc"] * 2, client_ip=["1.1.1.1", "2.2.2.2"], client_port=[1, 2], server_ip=["9.9.9.9"] * 2, start_ts=[10.0, 20.0],
                              orig_flow_id=[1, 2], label=["benign", "benign"], source=["s", "s"], **orig))
    m = match_to_original(conn, table, "abc")
    assert m.input_match.tolist() == [True, False]
    s = score_connections(m)
    assert (s["scored"], s["excluded_input_mismatch"]) == (1, 1)


def test_scoring_capacity_counts_per_group_before_any_prediction():
    from src.track_b.demo_infer import scoring_capacity

    conn = pd.DataFrame(dict(source=["a", "a", "a", "b"], status=["reached_k", "short_unconfirmed", "reached_k", "midstream_excluded"],
                             match=["matched", "matched", "ambiguous", "matched"], input_match=[True, True, True, True]))
    cap = scoring_capacity(conn).set_index("group")
    assert cap.loc["a", "rows"] == 3 and cap.loc["a", "scorable"] == 1 and cap.loc["b", "scorable"] == 0                # 라벨 수가 아니라 채점 가능 수


def test_short_connection_with_payload_after_closure_evidence_is_flagged():
    from src.track_b.demo_infer import late_payload

    f = flow(payload=4, start=100.0, times=[0.1, 0.2, 0.3, 3.0])                                                   # 3초에 관측한 payload가 있다
    assert late_payload(f, t_close=102.0) is True                                                                   # 종료 증거(시작 2초)보다 뒤의 payload가 모델 입력에 섞인다
    assert late_payload(f, t_close=104.0) is False
    assert late_payload(f, t_close=None) is False


def test_pcap_without_connections_returns_empty_table_instead_of_crashing(tmp_path):
    import src.track_b.demo_infer as di
    from src.common.models import make_model

    path = tmp_path / "empty.pcap"
    path.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x02\x00\x04\x00" + bytes(12) + (1).to_bytes(4, "little"))            # 헤더만 있는 classic pcap
    model = make_model("rf", seed=0, workers=1).fit(pd.DataFrame(np.zeros((8, 30)) + np.arange(8)[:, None], columns=[f"seq_{i}" for i in range(30)]),
                                                    ["benign", "probing", "bruteforce_xml", "bruteforce_https"] * 2)
    conn, timing = di.analyze_pcap(str(path), model)
    assert conn.empty and timing["connections"] == 0


def test_incomplete_original_input_columns_stop_scoring():
    from src.track_b.demo_infer import K

    conn = pd.DataFrame(dict(client_ip=["1.1.1.1"], client_port=[1], server_ip=["9.9.9.9"], start_ts=[10.0], **{f"len_{i}": [1] for i in range(K)}))
    table = pd.DataFrame(dict(pcap_sha256=["abc"], client_ip=["1.1.1.1"], client_port=[1], server_ip=["9.9.9.9"], start_ts=[10.0], orig_flow_id=[1], label=["benign"],
                              **{f"orig_seq_{i}": [1] for i in range(K - 1)}))                                       # orig_seq_9가 빠졌다
    with pytest.raises(ValueError):
        match_to_original(conn, table, "abc")
    with pytest.raises(ValueError):
        match_to_original(conn, table.drop(columns=[c for c in table if c.startswith("orig_seq_")]), "abc")        # 열이 아예 없어도 입력 검사를 못 하므로 중단
