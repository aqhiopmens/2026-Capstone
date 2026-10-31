import unittest

import numpy as np
import pandas as pd

from src.track_b import early_eval as ee


def flows(rows):
    data = []
    for label, day, seq, total, times in rows:
        row = dict(label=label, day=day, payload_packets=total)
        padded = list(seq) + [0] * (30 - len(seq))
        row.update({f"seq_{i}": padded[i] for i in range(30)})
        row.update({f"t_{i}": (times[i] if i < len(times) else np.nan) for i in range(30)})
        data.append(row)
    return pd.DataFrame(data)


class FeatureTests(unittest.TestCase):
    def test_seq_keeps_30_columns_and_zeros_after_k(self):
        df = flows([("benign", "d", [5, -6, 7, -8, 9, 10], 6, [])])
        x = ee.prefix_features(df, 3, "seq")
        self.assertEqual(x.shape, (1, 30))
        self.assertEqual(x.iloc[0].tolist(), [5, -6, 7] + [0] * 27)

    def test_counts_use_only_the_first_k_packets(self):
        df = flows([("benign", "d", [5, -6, 7, -8, 9, 10], 6, [])])
        counts = ee.prefix_features(df, 3, "counts").iloc[0].to_dict()
        self.assertEqual(counts, dict(n_seen=3, total_abs=18, total_up=12, total_down=6))
        self.assertEqual(ee.prefix_features(df, 5, "counts").iloc[0].n_seen, 5)
        with self.assertRaises(ValueError):
            ee.prefix_features(df, 3, "other")

    def test_prefix_features_ignore_values_after_k(self):
        a = flows([("benign", "d", [1, -2, 3, 100, 200], 5, [])])
        b = flows([("benign", "d", [1, -2, 3, -7, 9], 5, [])])
        for kind in ee.FEATURE_KINDS:
            self.assertTrue(ee.prefix_features(a, 3, kind).equals(ee.prefix_features(b, 3, kind)))


class SubsetTests(unittest.TestCase):
    def test_missing_class_gives_nan_recall_and_zero_division_f1(self):
        scores = ee.subset_scores(["benign", "benign"], ["benign", "bruteforce_xml"])
        self.assertTrue(np.isnan(scores["recall_bruteforce_xml"]))
        self.assertEqual((scores["support_benign"], scores["support_bruteforce_xml"]), (2, 0))
        self.assertAlmostEqual(scores["recall_benign"], 0.5)
        self.assertAlmostEqual(scores["accuracy"], 0.5)

    def test_empty_subset_is_all_nan(self):
        scores = ee.subset_scores([], [])
        self.assertEqual(scores["rows"], 0)
        self.assertTrue(np.isnan(scores["accuracy"]) and np.isnan(scores["macro_f1"]))


class DiagnosticsTests(unittest.TestCase):
    def test_diversity_shared_and_majority_baseline(self):
        rows = []
        for day in ("2021-04-12", "2021-04-17"):
            rows += [("benign", day, [10, -20, 30, -40, 50], 5, []), ("benign", day, [11, -20, 30, -40, 50], 5, []),
                     ("bruteforce_xml", day, [99, -1, 2, -3, 4], 5, [])]
        d = ee.diagnostics(flows(rows), ks=(5,))
        diversity = d[d.kind == "prefix_diversity"].iloc[0]
        self.assertEqual((diversity.attack_distinct, diversity.benign_distinct, diversity.shared), (1, 2, 0))
        majority = d[d.kind == "majority"]
        self.assertEqual(set(majority.name), {"benign"})
        self.assertTrue(np.allclose(majority.accuracy, 2 / 3))
        first = d[(d.kind == "single_feature") & (d.name == "seq_0")]
        self.assertTrue((first.accuracy == 1.0).all())

    def test_n_seen_alone_cannot_beat_majority_when_all_flows_reach_k(self):
        rows = []
        for day in ("2021-04-12", "2021-04-17"):
            rows += [("benign", day, [1, 2, 3], 3, []), ("benign", day, [4, 5, 6], 3, []),
                     ("bruteforce_xml", day, [7, 8, 9], 3, [])]
        d = ee.diagnostics(flows(rows), ks=(3,))
        n_seen = d[(d.kind == "single_feature") & (d.name == "n_seen")]
        self.assertTrue(np.allclose(n_seen.accuracy, 2 / 3))


class LatencyTests(unittest.TestCase):
    def test_latency_counts_only_flows_that_reached_k_and_uses_total_payload_packets(self):
        df = flows([("a", "d", [1, 2, 3], 3, [0.1, 0.2, 0.3]),
                    ("a", "d", [1, 2], 2, [0.1, 0.2]),
                    ("a", "d", [1] * 30, 50, [0.01 * i for i in range(30)])])
        table = ee.latency_table(df, ks=(3, 5)).set_index("k")
        self.assertEqual((table.loc[3, "reached_k_rows"], table.loc[3, "short_rows"]), (2, 1))
        self.assertEqual(table.loc[5, "reached_k_rows"], 1)
        self.assertAlmostEqual(table.loc[3, "latency_median_s"], (0.3 + 0.02) / 2)
        self.assertAlmostEqual(table.loc[5, "short_fraction"], 2 / 3, places=5)


if __name__ == "__main__":
    unittest.main()
