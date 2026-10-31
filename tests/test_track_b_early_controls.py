import unittest

import numpy as np
import pandas as pd

from src.track_b import early_controls as ec
from src.track_b.pcap_flows import K_MAX


def flows(rows):
    data = []
    for seq in rows:
        padded = list(seq) + [0] * (K_MAX - len(seq))
        data.append({f"seq_{i}": v for i, v in enumerate(padded)})
    return pd.DataFrame(data)


class ControlFeatureTests(unittest.TestCase):
    def test_mask_first_zeroes_only_seq_0_and_keeps_columns(self):
        x = ec.control_features(flows([[5, -6, 7, -8, 9, 10]]), 3, "mask_first", 0)
        self.assertEqual(list(x.iloc[0, :6]), [0, -6, 7, 0, 0, 0])

    def test_abs_drops_sign_after_cutting_to_k(self):
        x = ec.control_features(flows([[5, -6, 7, -8]]), 3, "abs", 0)
        self.assertEqual(list(x.iloc[0, :5]), [5, 6, 7, 0, 0])

    def test_shuffle_cuts_first_so_values_after_k_never_enter(self):
        df = flows([[1, 2, 3, 4, 5, 6, 7, 8]])
        for seed in range(20):
            row = ec.control_features(df, 3, "shuffle", seed).iloc[0].to_numpy()
            self.assertEqual(sorted(row[:3]), [1, 2, 3])
            self.assertTrue((row[3:] == 0).all())

    def test_shuffle_keeps_padding_positions_and_is_row_id_and_seed_deterministic(self):
        df = flows([[1, -2, 3], [4, -5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        a = ec.control_features(df, 10, "shuffle", 1)
        self.assertTrue((a.iloc[0, 3:] == 0).all())
        self.assertEqual(sorted(a.iloc[0, :3]), [-2, 1, 3])
        self.assertTrue(a.equals(ec.control_features(df, 10, "shuffle", 1)))
        self.assertFalse(a.equals(ec.control_features(df, 10, "shuffle", 2)))

    def test_shuffle_is_independent_per_row(self):
        df = flows([list(range(1, 11))] * 6)
        x = ec.control_features(df, 10, "shuffle", 0).to_numpy()
        self.assertGreater(len({tuple(r) for r in x}), 1)


class CompareTests(unittest.TestCase):
    def test_delta_counts_newly_wrong_and_right_on_same_rows(self):
        control = pd.DataFrame(dict(row_id=[1, 2, 3, 4], actual=["benign", "benign", "bruteforce_xml", "bruteforce_xml"],
                                    predicted=["benign", "bruteforce_xml", "bruteforce_xml", "benign"],
                                    reached_k=[True, True, True, False]))
        base = control[["row_id", "actual"]].assign(predicted=["benign", "benign", "benign", "benign"])
        rows = {r["subset"]: r for r in ec.compare_to_baseline(control, base)}
        self.assertEqual((rows["all"]["newly_wrong"], rows["all"]["newly_right"]), (1, 1))
        self.assertEqual(rows["reached_k"]["rows"], 3)
        self.assertEqual(rows["short"]["rows"], 1)
        self.assertAlmostEqual(rows["all"]["delta_accuracy"], 0.5 - 0.5)

    def test_mismatched_rows_or_labels_raise(self):
        control = pd.DataFrame(dict(row_id=[1], actual=["benign"], predicted=["benign"], reached_k=[True]))
        with self.assertRaises(ValueError):
            ec.compare_to_baseline(control, pd.DataFrame(dict(row_id=[2], actual=["benign"], predicted=["benign"])))
        with self.assertRaises(ValueError):
            ec.compare_to_baseline(control, pd.DataFrame(dict(row_id=[1], actual=["bruteforce_xml"], predicted=["benign"])))


if __name__ == "__main__":
    unittest.main()
