import unittest

import numpy as np
import pandas as pd

from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.https_unseen import NEW_VICTIM, attack_score, diagnose, oracle_recall


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


if __name__ == "__main__":
    unittest.main()
