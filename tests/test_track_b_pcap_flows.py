import socket
import struct
import tempfile
import unittest
from pathlib import Path

from src.track_b import pcap_flows as pf

CLIENT, SERVER, OTHER = "10.0.0.1", "20.0.0.2", "30.0.0.3"


def frame(src, sport, dst, dport, flags=0x18, seq=1000, payload=0, vlan=False, pad=0):
    """이더넷+IPv4+TCP 프레임. payload는 길이만 지정한다(내용은 0 바이트)."""
    tcp = struct.pack(">HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, 65535, 0, 0) + bytes(payload)
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst)) + tcp
    eth = bytes(12) + (b"\x81\x00\x00\x01\x08\x00" if vlan else b"\x08\x00")
    return eth + ip + bytes(pad)


def write_pcap(packets):
    """packets: (ts, frame 바이트) 목록 → 임시 pcap 경로."""
    data = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for ts, raw in packets:
        micros = round(ts * 1e6)
        data += struct.pack("<IIII", micros // 1_000_000, micros % 1_000_000, len(raw), len(raw)) + raw
    handle = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
    handle.write(data)
    handle.close()
    return Path(handle.name)


def run(packets):
    path = write_pcap(packets)
    try:
        return pf.assemble_flows(path)
    finally:
        path.unlink()


def handshake(ts, port=40000, client=CLIENT):
    return [(ts, frame(client, port, SERVER, 443, flags=0x02, seq=1, payload=0)),
            (ts + 0.01, frame(SERVER, 443, client, port, flags=0x12, seq=5000)),
            (ts + 0.02, frame(client, port, SERVER, 443, flags=0x10, seq=2))]


class AssemblyTests(unittest.TestCase):
    def test_signed_lengths_in_order_and_zero_padding(self):
        packets = handshake(100.0) + [
            (100.10, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (100.20, frame(SERVER, 443, CLIENT, 40000, seq=5001, payload=200)),
            (100.30, frame(CLIENT, 40000, SERVER, 443, seq=102, payload=50)),
        ]
        flows, _, _ = run(packets)
        self.assertEqual(len(flows), 1)
        flow = flows[0]
        self.assertEqual(flow.seq, [100, -200, 50])
        self.assertEqual(flow.payload_packets, 3)
        self.assertEqual(flow.times, [0.1, 0.2, 0.3])
        self.assertFalse(flow.midstream)
        row = pf.flow_rows(flows, "x.pcap", dict(day="d", attacker=CLIENT, victim=SERVER, attack="a"))[0]
        self.assertEqual([row[f"seq_{i}"] for i in range(5)], [100, -200, 50, 0, 0])
        self.assertEqual(row["label"], "a")

    def test_ethernet_padding_and_vlan_are_not_payload(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, flags=0x10, seq=2, pad=6)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=70, vlan=True)),
        ]
        flows, _, _ = run(packets)
        self.assertEqual(flows[0].seq, [70])
        self.assertEqual(flows[0].payload_packets, 1)

    def test_retransmission_is_not_counted(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.5, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.6, frame(CLIENT, 40000, SERVER, 443, seq=102, payload=40)),
        ]
        flow = run(packets)[0][0]
        self.assertEqual(flow.seq, [100, 40])
        self.assertEqual(flow.retransmissions, 1)

    def test_seq_number_wraparound_is_not_a_retransmission(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=0xFFFFFFF0, payload=10)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=0xFFFFFFFA, payload=20)),
        ]
        self.assertEqual(run(packets)[0][0].seq, [10, 20])

    def test_out_of_order_late_segment_is_new_data_not_retransmission(self):
        # (2,100) -> (202,100) -> (102,100): 세 구간이 겹치지 않으므로 3개여야 한다 (Codex P1 재현)
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=202, payload=100)),
            (10.3, frame(CLIENT, 40000, SERVER, 443, seq=102, payload=100)),
        ]
        flow = run(packets)[0][0]
        self.assertEqual((flow.seq, flow.payload_packets, flow.retransmissions), ([100, 100, 100], 3, 0))

    def test_exact_duplicate_after_gap_fill_and_covered_subrange_are_retransmissions(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=202, payload=100)),
            (10.3, frame(CLIENT, 40000, SERVER, 443, seq=102, payload=100)),
            (10.4, frame(CLIENT, 40000, SERVER, 443, seq=102, payload=100)),  # 완전 중복
            (10.5, frame(CLIENT, 40000, SERVER, 443, seq=50, payload=100)),   # 이미 받은 구간 안
        ]
        flow = run(packets)[0][0]
        self.assertEqual((flow.payload_packets, flow.retransmissions), (3, 2))

    def test_partial_overlap_with_new_bytes_is_counted(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=52, payload=100)),  # 52~152, 새 바이트 있음
        ]
        flow = run(packets)[0][0]
        self.assertEqual((flow.payload_packets, flow.retransmissions), (2, 0))

    def test_out_of_order_across_seq_wraparound(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=0xFFFFFFF0, payload=10)),
            (10.2, frame(CLIENT, 40000, SERVER, 443, seq=0x00000004, payload=10)),   # 순환 뒤 구간이 먼저
            (10.3, frame(CLIENT, 40000, SERVER, 443, seq=0xFFFFFFFA, payload=10)),   # 사이 구간이 늦게
            (10.4, frame(CLIENT, 40000, SERVER, 443, seq=0xFFFFFFFA, payload=10)),   # 완전 중복
        ]
        flow = run(packets)[0][0]
        self.assertEqual((flow.payload_packets, flow.retransmissions), (3, 1))

    def test_directions_are_tracked_independently(self):
        packets = handshake(10.0) + [
            (10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)),
            (10.2, frame(SERVER, 443, CLIENT, 40000, seq=2, payload=100)),  # 같은 번호라도 반대 방향
        ]
        self.assertEqual(run(packets)[0][0].seq, [100, -100])

    def test_corrupt_tcp_data_offset_and_ip_version_are_skipped_and_counted(self):
        good = frame(CLIENT, 40000, SERVER, 443, seq=2, payload=100)
        bad_offset = bytearray(good)
        bad_offset[14 + 20 + 12] = 0          # TCP data offset 0
        bad_version = bytearray(good)
        bad_version[14] = 0x65                # IP 버전 6, 헤더 길이 20
        packets = handshake(10.0) + [(10.1, bytes(bad_offset)), (10.2, bytes(bad_version)), (10.3, good)]
        flows, skipped, _ = run(packets)
        self.assertEqual(flows[0].seq, [100])
        self.assertEqual(skipped.get("bad_tcp_offset"), 1)
        self.assertEqual(skipped.get("bad_ip_version"), 1)

    def test_syn_after_close_starts_new_flow_and_idle_timeout_splits(self):
        packets = (handshake(10.0) + [(10.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=10)),
                                      (10.2, frame(CLIENT, 40000, SERVER, 443, flags=0x11, seq=12))]
                   + handshake(11.0) + [(11.1, frame(CLIENT, 40000, SERVER, 443, seq=2, payload=20))]
                   + [(11.1 + pf.IDLE_TIMEOUT + 1, frame(CLIENT, 40000, SERVER, 443, seq=22, payload=30))])
        flows, _, _ = run(packets)
        self.assertEqual([flow.seq for flow in flows], [[10], [20], [30]])
        self.assertTrue(flows[2].midstream)

    def test_midstream_flow_is_marked_and_only_443_is_kept(self):
        packets = [(5.0, frame(SERVER, 443, CLIENT, 41000, seq=9, payload=300)),
                   (5.1, frame(CLIENT, 41000, OTHER, 80, seq=1, payload=999))]
        flows, _, tcp = run(packets)
        self.assertEqual(tcp, 2)
        self.assertEqual(len(flows), 1)
        self.assertEqual((flows[0].seq, flows[0].midstream), ([-300], True))

    def test_only_first_30_payload_packets_are_stored_but_all_are_counted(self):
        packets = handshake(1.0) + [
            (1.1 + i * 0.01, frame(CLIENT, 40000, SERVER, 443, seq=2 + 10 * i, payload=10)) for i in range(35)]
        flow = run(packets)[0][0]
        self.assertEqual((len(flow.seq), flow.payload_packets), (30, 35))

    def test_unsupported_capture_is_rejected(self):
        path = Path(tempfile.NamedTemporaryFile(suffix=".pcap", delete=False).name)
        path.write_bytes(b"\x0a\x0d\x0d\x0a" + bytes(40))
        try:
            with self.assertRaises(ValueError):
                list(pf.iter_tcp_packets(path))
        finally:
            path.unlink()


class PrefixTests(unittest.TestCase):
    def test_prefix_input_keeps_30_columns_and_k_equal_30_is_identity(self):
        flow = pf.Flow("c", 1, "s", 0.0, 0.0, seq=[5, -6, 7, -8, 9, 10], payload_packets=6)
        self.assertEqual(pf.prefix_input(flow, 3), [5, -6, 7] + [0] * 27)
        self.assertEqual(pf.prefix_input(flow, 30), [5, -6, 7, -8, 9, 10] + [0] * 24)
        self.assertEqual(pf.prefix_input(flow, 5)[:5], [5, -6, 7, -8, 9])

    def test_prefix_report_uses_payload_packet_count_not_stored_prefix(self):
        rows = [dict(pcap="p", label="a", payload_packets=n) for n in (2, 5, 40)]
        report = {row["k"]: row for row in pf.prefix_report(rows, ks=(5, 20, 30))}
        self.assertEqual((report[5]["reached_k_rows"], report[5]["short_rows"]), (2, 1))
        self.assertEqual((report[30]["reached_k_rows"], report[30]["short_rows"]), (1, 2))
        self.assertAlmostEqual(report[20]["short_fraction"], 2 / 3, places=5)
        self.assertEqual(report[5]["rows"], 3)
        self.assertEqual(report[5]["padding_policy"], pf.PADDING_POLICY)


if __name__ == "__main__":
    unittest.main()
