from src.track_b.collect import match
from src.track_b.pcap_flows import Flow


def flow(port, ts):
    return Flow("10.0.0.2", port, "10.0.0.1", ts, ts, server_port=8443)


def row(port, t, cls="normal"):
    return dict(client_ip="10.0.0.2", client_port=port, server_ip="10.0.0.1", server_port=8443, t_start=t, cls=cls,
                server="S1", impl="curl")


def test_match_corrects_clock_offset_and_port_reuse():
    # 클라이언트 시계가 5초 느림. 포트 50000이 두 번 쓰였고(정상 → 공격) 시각으로만 구분된다
    logs = [row(50000, 95.0, "normal"), row(50000, 195.0, "attack"), row(50001, 100.0), row(50002, 110.0)]
    flows = [flow(50000, 100.0), flow(50001, 105.0), flow(50002, 115.0), flow(50000, 200.0), flow(50009, 120.0)]
    matched, unmatched_flows, unmatched_logs, offset = match(flows, logs, tolerance=1.0)
    assert offset == 5.0
    assert [(f.client_port, r["cls"]) for f, r in matched] == [(50000, "normal"), (50001, "normal"), (50002, "normal"),
                                                                (50000, "attack")]
    assert [f.client_port for f in unmatched_flows] == [50009] and unmatched_logs == []


def test_match_rejects_same_tuple_far_in_time():
    # 다수가 어긋남 0초로 맞는데 한 연결만 같은 4-튜플이 50초 떨어져 있으면 오매칭하지 않는다
    logs = [row(p, float(p - 49000)) for p in (50001, 50002, 50003)] + [row(50000, 50.0)]
    flows = [flow(p, float(p - 49000)) for p in (50001, 50002, 50003)] + [flow(50000, 100.0)]
    matched, unmatched_flows, _, _ = match(flows, logs, tolerance=1.0)
    assert len(matched) == 3 and [f.client_port for f in unmatched_flows] == [50000]


def test_breakdown_denominators_per_label():
    from src.track_b.collect import breakdown
    ep = dict(conns_in_episode=1, episode_requests=3, episode_ok=1, episode_fail=2)
    logs = [dict(row(50001, 1.0, "attack"), impl="requests", episode=1, **ep),
            dict(row(50002, 2.0, "attack"), impl="requests", episode=2, **ep),
            dict(row(50003, 3.0), impl="curl", episode=3, conns_in_episode=1, episode_requests=1, ok=True)]
    d = breakdown(logs, [(flow(50001, 1.0), logs[0])])
    a, n = d["attack/requests"], d["normal/curl"]
    assert (a["conns"], a["episodes"], a["requests"], a["fail"], a["matched"], a["unmatched"]) == (2, 2, 6.0, 4.0, 1, 1)
    assert (n["ok"], n["fail"], n["requests_per_conn"]) == (1.0, 0.0, 1.0)
