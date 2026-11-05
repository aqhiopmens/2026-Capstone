import unittest

import numpy as np
import pandas as pd

from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.https_unseen import NEW_VICTIM, attack_score, budget_threshold, diagnose, oracle_recall


class UnseenDiagnosisTests(unittest.TestCase):
    def test_attack_score_restores_probability_from_predicted(self):
        s = attack_score([ATTACK, BENIGN, ATTACK], [0.9, 0.9, 0.5])
        self.assertTrue(np.allclose(s, [0.9, 0.1, 0.5]))

    def test_oracle_recall_uses_fpr_budget_and_strict_greater(self):
        neg = np.arange(100) / 100          # 0.00 .. 0.99
        pos = np.array([0.99, 0.985, 0.5])  # 예산 5%면 문턱 0.94 -> 앞 둘만 초과
        self.assertAlmostEqual(oracle_recall(pos, neg, 0.05), 2 / 3)
        self.assertAlmostEqual(oracle_recall(pos, np.zeros(100), 0.05), 1.0)  # 정상이 모두 0이면 0.5도 잡힘

    def test_diagnose_verdicts_for_separable_and_inseparable_scores(self):
        rows = []
        for _ in range(2):  # separable: 공격이 높은 점수
            rows += [dict(feature_group="g1", model="rf", actual=ATTACK, predicted=ATTACK, max_probability=0.95,
                          server_ip=NEW_VICTIM, status_key_payload="x")] * 5
            rows += [dict(feature_group="g1", model="rf", actual=BENIGN, predicted=BENIGN, max_probability=0.95,
                          server_ip="s", status_key_payload="single_agree")] * 40
        rows += [dict(feature_group="g2", model="rf", actual=ATTACK, predicted=BENIGN, max_probability=1.0,
                      server_ip=NEW_VICTIM, status_key_payload="x")] * 10
        rows += [dict(feature_group="g2", model="rf", actual=BENIGN, predicted=BENIGN, max_probability=0.5,
                      server_ip="s", status_key_payload="unmatched")] * 40
        d = diagnose(pd.DataFrame(rows))
        v = d.set_index(["feature_group", "benign_set"]).verdict
        self.assertEqual(v[("g1", "benign_all")], "threshold_problem")
        self.assertEqual(v[("g2", "benign_all")], "representation_problem")  # 공격이 정상보다 더 정상처럼 점수
        self.assertIn(("g2", "benign_csv_unmatched"), v.index)


class BudgetThresholdTests(unittest.TestCase):
    def test_strict_integer_budget_excludes_overflowing_tie_block(self):
        scores = np.r_[np.full(10, 5.0), np.arange(90) * 0.01]  # 최대 5.0이 10행 동률
        cut = budget_threshold(scores, 2, 100)  # 100행의 2% = 2행 < 동률 블록 10행
        self.assertEqual((cut["allowed"], cut["flagged"], cut["boundary_tie_rows"]), (2, 0, 10))
        cut = budget_threshold(scores, 10, 100)  # 예산 10행이 블록과 같아 모두 표시
        self.assertEqual(cut["flagged"], 10)

    def test_budget_never_exceeded_and_integer_arithmetic(self):
        scores = np.arange(4455, dtype=float)
        self.assertEqual(budget_threshold(scores, 2, 100)["allowed"], 89)
        self.assertEqual(budget_threshold(scores, 5, 1000)["allowed"], 22)
        self.assertEqual(budget_threshold(scores, 2, 100)["flagged"], 89)
        with self.assertRaises(ValueError):
            budget_threshold([], 2, 100)
        with self.assertRaises(ValueError):
            budget_threshold([1.0, np.nan], 2, 100)


class B1MetricsTests(unittest.TestCase):
    def test_confusion_counts_accuracy_and_macro_f1_treat_both_attack_victims_as_positive(self):
        from src.track_b.https_unseen import b1_metrics

        rows = [("attack", 1), ("attack", 0), ("a242", 1), ("benign", 0), ("benign", 0), ("benign", 1)]
        p = pd.DataFrame(dict(role=[r for r, _ in rows], pred_attack=[v for _, v in rows], kind="seq", k=5, seed=0, n=0))
        m = b1_metrics(p).iloc[0]
        self.assertEqual((m.tp, m.tn, m.fp, m.fn), (2, 2, 1, 1))
        self.assertAlmostEqual(m.accuracy, 4 / 6)
        self.assertAlmostEqual(m.macro_f1, (2 * 2 / (2 * 2 + 1 + 1) + 2 * 2 / (2 * 2 + 1 + 1)) / 2)


class B1SplitTests(unittest.TestCase):
    def frame(self):
        attacks = pd.DataFrame(dict(start_ts=[0.0, 1.0, 2.0, 100.0, 101.0, 2000.0], duration=[1.0] * 6, flow_id=range(6)))
        attacks = attacks.assign(label=ATTACK, server_ip=NEW_VICTIM, day="2021-04-16", pcap="p", payload_packets=5)
        benign = pd.DataFrame(dict(start_ts=[-50.0, 5.0, 300.0, 700.0, 3000.0], duration=[1.0] * 5, flow_id=range(10, 15)))
        benign = benign.assign(label=BENIGN, server_ip="s", day="2021-04-16", pcap="p", payload_packets=5)
        return pd.concat([attacks, benign], ignore_index=True)

    def test_window_covers_first_bundles_plus_ten_minutes_and_is_common_to_all_n(self):
        from src.track_b.https_unseen import b1_split, support_window

        f = self.frame()
        w0, w1, ordered, bundle = support_window(f[f.label == ATTACK], n_max=3)
        self.assertEqual((w0, w1), (0.0, 3.0 + 600.0))  # 첫 묶음(0~3초) 종료 + 10분
        self.assertEqual(bundle.tolist(), [0, 0, 0, 1, 1, 2])
        split = b1_split(f, n_max=3)
        test_ids = set(split["test"].flow_id)
        self.assertFalse(test_ids & {0, 1, 2, 3, 4, 10, 11, 12})  # 창(0~603초) 안 행과 창 시작 전 정상(10)은 시험 제외
        self.assertEqual(test_ids, {5, 13, 14})                     # 시험 = start_ts > 창 끝(창 이후)만
        self.assertEqual(split["before_window_rows"], 1)
        self.assertEqual(split["excluded_attacks"], 5)  # 창(0~603초) 안 공격 5행(0,1,2,100,101)
        self.assertEqual(sorted(split["window_benign"].flow_id), [11, 12])  # 5초, 300초 정상


if __name__ == "__main__":
    unittest.main()
