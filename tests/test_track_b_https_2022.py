import unittest

import numpy as np
import pandas as pd

from src.track_b.https_2022 import condition_rows, exact_ci, metrics_2022, split_2022
from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.pcap_flows import K_MAX


def make_flows(starts, labels, victims=None, seq0=None):
    n = len(starts)
    f = pd.DataFrame(dict(pcap="p", flow_id=np.arange(n), label=labels, server_ip=victims or ["184.0.48.150"] * n,
                          payload_packets=10, start_ts=np.asarray(starts, dtype=float), duration=1.0))
    for i in range(K_MAX):
        f[f"seq_{i}"] = 0
    f["seq_0"] = seq0 if seq0 is not None else 100
    return f


class Eval2022Tests(unittest.TestCase):
    def test_exact_ci_edges(self):
        lo, hi = exact_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3084, places=3)
        self.assertEqual(exact_ci(10, 10)[1], 1.0)
        self.assertTrue(np.isnan(exact_ci(0, 0)[0]))

    def test_split_test_starts_after_window_end_and_excludes_earlier_rows(self):
        # 공격 60개(1초 간격 -> 한 묶음), 정상 하나는 창 시작 전, 하나는 창 안, 하나는 창 끝(+600s) 뒤
        starts = list(range(1000, 1060)) + [500, 1030.5, 1059 + 1 + 600 + 1]
        labels = [ATTACK] * 60 + [BENIGN] * 3
        flows = make_flows(starts, labels)
        s = split_2022(flows, n_max=50)
        self.assertEqual(s["window"][0], 1000.0)
        self.assertEqual(s["window"][1], 1059 + 1 + 600)  # 한 묶음이라 60개 전체의 종료 + 600s
        test = flows[s["test_mask"]]
        self.assertEqual(list(test.label), [BENIGN])  # 창 이후 정상 1개만
        self.assertEqual(s["before_window_rows"], 1)
        self.assertEqual(s["inside_attack_rows"], 60)

    def test_condition_rows_counts_fp_unseen_prefix_and_nan_when_no_benign(self):
        # 시험 행 4개: 공격 2(앞 값 100 = 지원과 같음, 200 = 지원에 없음), 정상 2
        flows = make_flows([0, 1, 2, 3], [ATTACK, ATTACK, BENIGN, BENIGN], seq0=[100, 200, 100, 300])
        split = dict(support=flows.iloc[[0]], test_mask=np.ones(4, dtype=bool))
        make_x = lambda f: f[[f"seq_{i}" for i in range(5)]]
        pred = np.array([True, False, True, False])
        rows, det = condition_rows(flows, split, {1: pred}, make_x, 5, "seq", 0, (1,))
        r = rows[0]
        self.assertEqual((r["test_attack_rows"], r["test_attack_tp"], r["test_recall"]), (2, 1, 0.5))
        self.assertEqual((r["test_benign_rows"], r["test_benign_fp"], r["test_benign_fpr"]), (2, 1, 0.5))
        self.assertEqual(r["test_unseen_prefix_rows"], 1)  # seq_0=200인 공격만 지원에 없던 앞 k개 값
        self.assertEqual(r["test_recall_unseen_prefix"], 0.0)
        self.assertEqual(len(det), 4)
        only_attack = make_flows([0, 1], [ATTACK, ATTACK])
        split2 = dict(support=only_attack.iloc[[0]], test_mask=np.ones(2, dtype=bool))
        r2 = condition_rows(only_attack, split2, {1: np.array([True, True])}, make_x, 5, "seq", 0, (1,))[0][0]
        self.assertTrue(np.isnan(r2["test_benign_fpr"]))  # 정상 분모 0 -> 계산 불가

    def test_metrics_2022_counts_only_test_rows(self):
        p = pd.DataFrame(dict(actual=[ATTACK, ATTACK, BENIGN, BENIGN, ATTACK], pred_attack=[1, 0, 1, 0, 1], in_test=[1, 1, 1, 1, 0],
                              kind="seq", k=5, seed=0, n=0))
        r = metrics_2022(p).iloc[0]
        self.assertEqual((r.tp, r.tn, r.fp, r.fn, r.rows), (1, 1, 1, 1, 4))
        self.assertAlmostEqual(r.accuracy, 0.5)
        self.assertAlmostEqual(r.macro_f1, 0.5)


if __name__ == "__main__":
    unittest.main()
