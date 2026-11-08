import unittest

import numpy as np
import pandas as pd

from src.track_b.early_eval import SEQ_COLUMNS, mask3
from src.track_b.mask3 import BIG_DELTA, pair_metrics


class Mask3Tests(unittest.TestCase):
    def test_mask3_zeroes_only_first_three_columns_and_keeps_rest(self):
        x = pd.DataFrame([list(range(1, 31)), [0, 0, 0, 0, 0] + [0] * 25], columns=SEQ_COLUMNS)
        y = mask3(x)
        self.assertTrue((y.iloc[:, :3] == 0).all().all())
        self.assertEqual(y.iloc[0, 3:].tolist(), list(range(4, 31)))  # 열을 당기지 않고 나머지는 그대로
        self.assertEqual(x.iloc[0, 0], 1)  # 원본은 바뀌지 않는다

    def test_pair_metrics_pairs_by_k_seed_n_and_flags_big_change(self):
        base = pd.DataFrame(dict(k=[5, 5], seed=[0, 1], n=[0, 0], tp=[100, 100], tn=[50, 50], fp=[0, 0], fn=[0, 0],
                                 accuracy=[1.0, 1.0], macro_f1=[1.0, 1.0]))
        new = base.assign(tp=[40, 60], fn=[60, 40], accuracy=[0.7, 0.8], macro_f1=[0.6, 0.8])
        rows, summ = pair_metrics(base, new, "t")
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(summ.d_macro_f1.iloc[0], -0.3)
        self.assertAlmostEqual(summ.d_recall.iloc[0], -0.5)
        self.assertTrue(summ.big_change.iloc[0])
        self.assertEqual(BIG_DELTA, 0.05)


if __name__ == "__main__":
    unittest.main()


def test_role_recall_splits_victims_and_checks_rows():
    import pandas as pd
    from src.track_b.mask3 import role_recall
    def frame(a):
        return pd.DataFrame(dict(k=5, n=0, seed=0, role=["attack"] * 2 + ["a242"] * 2 + ["benign"], pred_attack=a + [0]))
    out = role_recall(frame([1, 0, 1, 1]), frame([0, 0, 1, 1])).set_index("role")
    assert out.loc[".88.81", "recall_seq"] == 0.5 and out.loc[".88.81", "recall_mask3"] == 0.0
    assert out.loc[".242.104", "d_recall"] == 0.0 and (out.rows == 2).all()
