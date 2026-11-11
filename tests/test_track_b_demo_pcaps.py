import struct

import pandas as pd

from src.track_b.demo_pcaps import build_match_table, cut_from

HEADER = b"\xd4\xc3\xb2\xa1" + b"\x00" * 20


def write_pcap(path, times, size=100):
    with open(path, "wb") as f:
        f.write(HEADER)
        for t in times:
            f.write(struct.pack("<IIII", int(t), int((t % 1) * 1e6), size, size) + bytes(size))


def test_cut_from_skips_packets_before_start_and_stops_at_size(tmp_path):
    src, dst = tmp_path / "a.pcap", tmp_path / "b.pcap"
    write_pcap(src, [100.0, 101.0, 200.0, 201.0, 202.0, 203.0])
    size, n = cut_from(src, dst, min_bytes=24 + 2 * 116, start_ts=150.0)
    assert n == 2 and size == 24 + 2 * 116                                                                         # 150초 이전 두 패킷은 건너뛰고 크기에서 멈춘다
    assert dst.read_bytes()[:24] == HEADER
    _, n_all = cut_from(src, tmp_path / "c.pcap", min_bytes=10_000, start_ts=None)
    assert n_all == 6


def test_match_table_uses_only_test_rows_of_that_pcap_inside_the_cut_window():
    seq = {f"seq_{i}": [i] * 5 for i in range(30)}
    data = pd.DataFrame(dict(pcap=["P", "P", "P", "Q", "P"], part=["test", "train", "excluded", "test", "test"], flow_id=[1, 2, 3, 4, 5], label=["benign"] * 5,
                             source=["s"] * 5, client_ip=["1.1.1.1"] * 5, client_port=[1, 2, 3, 4, 5], server_ip=["9.9.9.9"] * 5, start_ts=[10.0, 11.0, 12.0, 13.0, 99.0],
                             payload_packets=[12] * 5, **seq))
    t = build_match_table(data, "P", lo=9.0, hi=50.0, sha="abc")
    assert t.orig_flow_id.tolist() == [1]                                                                          # 학습·제외 행, 다른 pcap, 창 밖 행은 대응표에 없다
    assert set(t.pcap_sha256) == {"abc"} and "orig_seq_9" in t and "source" in t
