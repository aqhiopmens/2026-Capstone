import unittest

import numpy as np

from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.knn import knn1_predict, nearest_l1


class KnnTests(unittest.TestCase):
    def test_nearest_l1_min_distance(self):
        d = nearest_l1([[0, 0], [10, 10]], [[1, 1], [9, 12], [10, 10]])
        self.assertEqual(d.tolist(), [2, 3, 0])

    def test_predict_attack_when_strictly_closer_and_tie_goes_to_benign_with_exact_flag(self):
        x_train = np.array([[100, 0], [200, 0]])
        y_train = np.array([ATTACK, BENIGN])
        test = np.array([[110, 0], [150, 0], [200, 0], [190, 0]])  # 공격에 가까움 / 동률 / 정확 일치(정상) / 정상에 가까움
        pred, exact = knn1_predict(x_train, y_train, test)
        self.assertEqual(pred.tolist(), [True, False, False, False])  # 동률(150)은 정상
        self.assertEqual(exact.tolist(), [False, False, True, False])

    def test_empty_attack_set_never_predicts_attack(self):
        pred, _ = knn1_predict(np.array([[1, 1]]), np.array([BENIGN]), np.array([[1, 1], [5, 5]]))
        self.assertFalse(pred.any())


if __name__ == "__main__":
    unittest.main()
