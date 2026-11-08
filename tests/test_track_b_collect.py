import random

import pytest

from src.track_b.collect import (CELL_SPECS, CORRECT_LEN, PASSWORD, check_quota, episode_plan, episode_problem, estimate_offset, first_request_index,
                                 breakdown, expected_groups, group_offsets, match, password_for, round_episodes, run_phase)
from src.track_b.pcap_flows import Flow


def flow(port, ts, server_port=8443):
    return Flow("10.0.0.2", port, "10.0.0.1", ts, ts, server_port=server_port)


def row(port, t, cls="normal", **extra):
    return dict(client_ip="10.0.0.2", client_port=port, server_ip="10.0.0.1", server_port=8443, t_start=t, cls=cls, server_name="S1", impl="curl", **extra)


def test_plans_identical_across_classes_except_outcome():
    # 같은 (first, m)에서 정상·공격의 요청 종류·암호 길이가 같다(길이·요청 구성으로 클래스가 갈리지 않음)
    for first, m in CELL_SPECS:
        (tn, rn), (ta, ra) = episode_plan("normal", first, m), episode_plan("attack", first, m)
        assert [(r["method"], r["pw_len"]) for r in rn] == [(r["method"], r["pw_len"]) for r in ra] and len(rn) == m
        assert all(r["pw_len"] == CORRECT_LEN for r in rn[-1:] if r["method"] == "POST")  # 마지막 POST는 10자
        assert not any(r["expect_ok"] for r in ra)  # 공격은 전부 실패


def test_cell_quota_normal_7_2_1_and_m_multiset():
    types = [episode_plan("normal", f, m)[0] for f, m in CELL_SPECS]
    assert sorted(types) == ["failonly"] * 2 + ["none"] + ["success"] * 7
    assert sorted(m for _, m in CELL_SPECS) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    assert sum(f == "GET" for f, m in CELL_SPECS if m >= 2) == 4  # m≥2 첫 요청 GET 4 / POST 4
    # 정상 성공 에피소드의 마지막 요청만 올바른 암호
    rng = random.Random(0)
    ok = [r for t, rs in (episode_plan("normal", f, m) for f, m in CELL_SPECS) for r in rs if r["expect_ok"]]
    assert len(ok) == 7 and all(password_for(r, rng) == PASSWORD for r in ok)


def test_round_has_10_per_cell_plus_browse_and_check_quota():
    eps = round_episodes(random.Random(0))
    main = [e for e in eps if e[4] == "main"]
    assert len(main) == 40 and len(eps) == 44
    rows = []
    for ep_no, (cls, impl, first, m, subset) in enumerate(main):
        ep_type, reqs = episode_plan(cls, first, m)
        rows.append(dict(server_name="S1", phase="train", round=1, impl=impl, cls=cls, subset="main", ep_type=ep_type, m=m, first=first, episode=ep_no,
                         kinds="".join(r["method"][0] for r in reqs), pw_lens=[r["pw_len"] for r in reqs], auth_ok=[r["expect_ok"] for r in reqs]))
    assert check_quota(rows) == []
    rows[0]["pw_lens"] = [99]
    assert check_quota(rows)  # 배정표가 어긋나면 위반으로 잡힌다
    rows[0]["pw_lens"] = [r["pw_len"] for r in episode_plan(rows[0]["cls"], rows[0]["first"], rows[0]["m"])[1]]
    rows[0]["auth_ok"] = [not x for x in rows[0]["auth_ok"]]
    assert any("인증 결과" in v for v in check_quota(rows))  # 인증 결과가 배정과 다르면 위반


def test_match_uses_offset_and_separates_ambiguous():
    # 클라이언트 시계가 5초 느림(offset=+5). 포트 50000이 두 번 쓰여(정상→공격) 시각으로 구분된다
    logs = [row(50000, 95.0, "normal"), row(50000, 195.0, "attack"), row(50001, 100.0)]
    flows = [flow(50000, 100.0), flow(50001, 105.0), flow(50000, 200.0), flow(50009, 120.0)]
    matched, unmatched_flows, unmatched_logs, ambiguous = match(flows, logs, offset=5.0, tolerance=1.0)
    assert [(f.client_port, r["cls"]) for f, r in matched] == [(50000, "normal"), (50001, "normal"), (50000, "attack")]
    assert [f.client_port for f in unmatched_flows] == [50009] and unmatched_logs == [] and ambiguous == []
    # 허용오차 안에 후보가 둘이면 임의로 고르지 않고 모호로 분리
    dup = [row(50000, 95.0, "normal"), row(50000, 95.4, "attack")]
    matched, _, _, ambiguous = match([flow(50000, 100.0)], dup, offset=5.0, tolerance=1.0)
    assert matched == [] and len(ambiguous) == 1 and len(ambiguous[0][1]) == 2


def test_estimate_offset_fails_without_enough_reference():
    logs = [row(p, float(p)) for p in range(50000, 50040)]
    server = [dict(client_ip="10.0.0.2", client_port=p, server_port=8443, t_accept=float(p) + 5.01) for p in range(50000, 50040)]
    offset, resid, n = estimate_offset(logs, server, min_ref=30)
    assert round(offset, 2) == 5.01 and n == 40 and max(abs(r) for r in resid) < 1e-6
    with pytest.raises(ValueError):  # 기준 연결이 부족하면 offset=0으로 대체하지 않고 불합격
        estimate_offset(logs[:10], server[:10], min_ref=30)


def test_first_request_index_candidate_or_unconfirmed():
    assert first_request_index([517, -1286, 80, -255, -255, 300, -135]) == 5  # Finished(80) 다음 양수 = 요청 후보
    assert first_request_index([517, -1286, 80, -255]) is None  # 요청이 앞 k개에 없으면 미확인


def test_episode_problem_rejects_bad_status_auth_and_curl_failure():
    _, reqs = episode_plan("normal", "POST", 2)  # POST 실패 → POST 성공
    conn = [dict(client_ip="a", client_port=1, t_start=0.0)]
    assert episode_problem(reqs, conn, [(401, False), (200, True)]) == ""
    assert "auth" in episode_problem(reqs, conn, [(401, False), (401, False)])  # 성공이어야 할 요청이 실패
    assert "status" in episode_problem(reqs, conn, [(000, False), (200, True)])  # 연결 후 상태 000
    assert "conns=2" in episode_problem(reqs, conn * 2, [(401, False), (200, True)])  # 재연결
    assert "rc=" in episode_problem(reqs, conn, [(401, False), (200, True)], note="rc=60 cert")


def test_group_offsets_per_phase_server_and_insufficient_reference():
    logs = [dict(row(p, float(p)), phase="train", server_name="S1", impl="curl", round=1) for p in range(50000, 50040)]
    server = [dict(client_ip="10.0.0.2", client_port=p, server_port=8443, t_accept=float(p) + 2.0) for p in range(50000, 50040)]
    out = group_offsets(logs, server, min_ref=30)
    assert round(out[("train", "S1")]["offset"], 3) == 2.0 and out[("train", "S1")]["n"] == 40
    with pytest.raises(ValueError):
        group_offsets(logs[:10], server[:10], min_ref=30)


def test_loss_rate_is_per_server_phase_cell_not_diluted():
    # S1 시험 칸 100연결 중 3개 손실(3%) + 다른 칸 500연결 손실 0: 합치면 0.5%로 희석되지만 칸별로는 3%라 불합격이어야 한다
    logs, matched = [], []
    for sv, phase, n in (("S1", "test", 100), ("S2", "test", 250), ("S3", "train", 250)):
        for i in range(n):
            r = dict(row(50000 + len(logs), float(len(logs)), "normal", phase=phase, subset="main", episode=i, kinds="G", auth_ok=[False]), server_name=sv)
            logs.append(r)
            if not (sv == "S1" and i < 3):
                matched.append((flow(r["client_port"], r["t_start"]), r))
    out = breakdown(logs, matched, [], 0.0)
    assert out["S1/test/normal/curl/main"]["lost_frac"] == 0.03 and out["S2/test/normal/curl/main"]["lost_frac"] == 0.0


def test_missing_cells_and_groups_fail():
    exp_groups, exp_cells = expected_groups(1, 1)
    assert len(exp_groups) == 6 and len(exp_cells) == 6 * 4
    assert any("칸이 비어 있음" in v for v in check_quota([], exp_cells))  # 빈 자료도 위반
    logs = [dict(row(p, float(p)), phase="train", server_name="S1", impl="curl", round=1) for p in range(50000, 50040)]
    server = [dict(client_ip="10.0.0.2", client_port=p, server_port=8443, t_accept=float(p) + 1.0) for p in range(50000, 50040)]
    with pytest.raises(ValueError):  # 시험·다른 서버 그룹에 기준 연결이 하나도 없으면 전체 offset으로 대체하지 않고 불합격
        group_offsets(logs, server, min_ref=30, expected=exp_groups)


def test_partial_connections_survive_exception(monkeypatch, tmp_path):
    import json as _json
    import src.track_b.collect as c

    def fake(impl, host, port, reqs, rng, conns):
        conns.append(dict(client_ip="10.0.0.2", client_port=51234, t_start=1.0))  # 연결을 맺은 뒤 응답 timeout
        raise TimeoutError

    monkeypatch.setattr(c, "do_episode", fake)
    monkeypatch.setattr(c, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    log, att = c.Log(tmp_path / "l.jsonl"), c.Log(tmp_path / "a.jsonl")
    args = type("A", (), {"host": "10.0.0.1"})()
    with pytest.raises(RuntimeError):  # 쿼터를 못 채우면 중단
        run_phase(args, "S1", "train", [[("normal", "curl", "GET", 1, "main")]], random.Random(0), log, att)
    rows = [_json.loads(l) for l in (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["conns"] == [["10.0.0.2", 51234]] and rows[0]["reason"].startswith("error:")
