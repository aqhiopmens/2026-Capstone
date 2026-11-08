import struct
import unittest

from src.track_b.tls_audit import describe_packet, parse_records, reassemble


def record(rtype, body):
    return bytes([rtype, 3, 3]) + struct.pack(">H", len(body)) + body


def hs(msg_type, payload):
    return bytes([msg_type]) + len(payload).to_bytes(3, "big") + payload


class TlsAuditTests(unittest.TestCase):
    def test_parse_records_finds_handshake_messages_and_stops_on_bad_header(self):
        ch = record(22, hs(1, b"x" * 10))
        sh_cert = record(22, hs(2, b"y" * 4) + hs(11, b"z" * 20))
        data = ch + sh_cert + bytes([99, 3, 3, 0, 5]) + b"junk"
        recs = parse_records(data)
        self.assertEqual([(r[1], r[3]) for r in recs], [("Handshake", ["ClientHello"]), ("Handshake", ["ServerHello", "Certificate"])])
        self.assertEqual(recs[1][0], len(ch))

    def test_reassemble_orders_segments_drops_retransmission_and_maps_packets(self):
        rec = record(22, hs(1, b"a" * 30))
        a, b = rec[:20], rec[20:]
        streams, placed = reassemble([(1, 1000, a), (1, 1000, a), (1, 1020, b)])  # 두 번째는 완전한 재전송
        self.assertEqual(streams[1], rec)
        self.assertEqual(placed, [(1, 0, 20), (1, 20, len(rec) - 20)])
        desc0, bound0 = describe_packet(placed[0], {1: parse_records(streams[1])})
        desc1, bound1 = describe_packet(placed[1], {1: parse_records(streams[1])})
        self.assertIn("ClientHello", desc0)
        self.assertEqual((bound0, bound1), ("경계 시작", "레코드 중간에서 시작(분할)"))  # 레코드가 두 패킷에 걸치면 둘째는 분할로 센다


if __name__ == "__main__":
    unittest.main()


class TlsAuditBoundaryTests(unittest.TestCase):
    def test_no_record_overlap_is_unconfirmed_not_boundary(self):
        self.assertEqual(describe_packet((1, 0, 9), {1: []}), ("해석 불가(레코드 밖)", "미확인"))

    def test_packet_starting_inside_unparsed_gap_is_unconfirmed(self):
        recs = {1: parse_records(b"\x00" * 10 + record(22, hs(1, b"a" * 5)))}  # 앞 10바이트가 헤더로 해석되지 않아 레코드 없음
        self.assertEqual(describe_packet((1, 0, 30), recs)[1], "미확인")

    def test_reorder_before_first_packet_and_gap_bytes_are_counted(self):
        stats = {}
        streams, placed = reassemble([(1, 1000, b"a" * 10), (1, 990, b"b" * 5), (1, 1030, b"c" * 10)], stats)
        self.assertEqual((stats["pre_base"], stats["gap_bytes"], len(placed)), (1, 25, 3))
        self.assertEqual(placed[1][1], 0)  # 재정렬 패킷이 스트림 맨 앞으로

    def test_truncated_final_record_is_flagged(self):
        rec = record(22, hs(1, b"a" * 30))[:20]
        self.assertFalse(parse_records(rec)[0][4])
        self.assertIn("잘림", describe_packet((1, 0, 20), {1: parse_records(rec)})[0])
