import unittest

import numpy as np
import pandas as pd

from src.track_b.four_class import BENIGN, PROBING, class_scores, probing_parts


class FourClassTests(unittest.TestCase):
    def test_probing_time_blocks_exclude_boundary_crossing_and_gap_rows(self):
        # 전체 구간 0~10000초 -> T=7000, 간격 600초 -> 시험은 시작 >= 7600
        frame = pd.DataFrame(dict(start_ts=[0.0, 10.0, 6900.0, 7100.0, 7700.0, 9999.0], duration=[10.0, 10.0, 200.0, 50.0, 10.0, 1.0]))
        part, t = probing_parts(frame, fraction=0.7, gap=600.0)
        self.assertAlmostEqual(t, 7000.0)
        # 2번(끝 7100 > T, 경계 가로지름)과 3번(간격 안)은 제외, 앞 둘은 학습, 뒤 둘은 시험
        self.assertEqual(part.tolist(), ["train", "train", "excluded", "excluded", "test", "test"])

    def test_class_scores_macro_ignores_absent_classes(self):
        scores, macro = class_scores(np.array([BENIGN, BENIGN, PROBING]), np.array([BENIGN, PROBING, PROBING]))
        self.assertEqual(scores.set_index("label").support[PROBING], 1)
        self.assertAlmostEqual(macro, (2 / 3 + 2 / 3) / 2)  # 정상 f1 2/3, probing f1 2/3, 지원 0인 클래스는 평균에서 제외


if __name__ == "__main__":
    unittest.main()
