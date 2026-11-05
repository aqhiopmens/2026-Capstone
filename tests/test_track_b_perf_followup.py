import unittest
from types import SimpleNamespace

import numpy as np

from src.track_b.perf_followup import build_input, summarize, time_variant, variants


class FollowupTests(unittest.TestCase):
    def test_variants_are_fixed_set(self):
        names = [v[0] for v in variants()]
        self.assertEqual(names, ["rf100_df", "rf100_np", "rf50_df", "rf50_np", "rf25_df", "rf25_np", "lgbm_df"])

    def test_build_input_types_and_prefix(self):
        flow = SimpleNamespace(seq=[100, -200, 50, -60, 70, 80])
        df, arr = build_input(flow, 3, "df"), build_input(flow, 3, "np")
        self.assertEqual(df.shape, (1, 30))
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr[0, :4].tolist(), [100, -200, 50, 0])  # 앞 k=3개만 남고 나머지 0

    def test_time_variant_measures_three_times_per_flow_after_warmup(self):
        calls = []
        model = SimpleNamespace(predict=lambda x: calls.append(1))
        flows = [SimpleNamespace(seq=[1, 2, 3, 4, 5]) for _ in range(12)]
        ms = time_variant(model, "np", flows, 5)
        self.assertEqual(ms.shape, (12, 3))
        self.assertEqual(len(calls), 10 + 12 * 2)  # 워밍업 10 + 플로우당 predict 2회(미리 만든 입력, 직접 합계)
        self.assertTrue((ms >= 0).all())
        s = summarize(ms[:, 2])
        self.assertEqual(s["over_10ms"], int((ms[:, 2] > 10).sum()))


if __name__ == "__main__":
    unittest.main()
