import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.track_b.perf_measure import cut_pcap, pctl, single_latency


def fake_pcap(path, sizes):
    with open(path, "wb") as f:
        f.write(b"\xd4\xc3\xb2\xa1" + b"\0" * 20)
        for n in sizes:
            f.write(struct.pack("<IIII", 0, 0, n, n) + b"x" * n)


class PerfTests(unittest.TestCase):
    def test_cut_pcap_stops_at_first_packet_boundary_reaching_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "a.pcap", Path(d) / "b.pcap"
            fake_pcap(src, [100, 100, 100, 100])  # 헤더 24, 레코드마다 116
            size, packets = cut_pcap(src, dst, min_bytes=24 + 2 * 116)
            self.assertEqual((size, packets), (24 + 2 * 116, 2))  # 정확히 임계에 닿는 두 번째 패킷에서 멈춘다
            size, packets = cut_pcap(src, dst, min_bytes=24 + 2 * 116 + 1)
            self.assertEqual(packets, 3)  # 넘기려면 세 번째 패킷까지 포함(패킷 경계)
            self.assertEqual(dst.stat().st_size, size)
            size, packets = cut_pcap(src, dst, min_bytes=10**9)
            self.assertEqual(packets, 4)  # 파일이 임계보다 작으면 전체

    def test_single_latency_times_each_flow_and_percentile(self):
        calls = []
        model = SimpleNamespace(predict=lambda x: calls.append(len(x)))
        flows = [SimpleNamespace(seq=[100, -200, 50, -60, 70]) for _ in range(5)]
        ms = single_latency(model, flows, 5, warmup=2)
        self.assertEqual(len(ms), 5)
        self.assertEqual(len(calls), 7)  # 워밍업 2 + 측정 5, 모두 1행 입력
        self.assertTrue(all(c == 1 for c in calls) and (ms >= 0).all())
        self.assertAlmostEqual(pctl([1, 2, 3, 4, 100], 50), 3.0)


if __name__ == "__main__":
    unittest.main()
