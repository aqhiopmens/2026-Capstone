import unittest

import numpy as np
import pandas as pd

from src.track_b.https_eval import ATTACK, BENIGN, SOURCE_ONLY, partitions_for, scored, subset_masks


def flows():
    return pd.DataFrame(dict(
        day=["d1", "d1", "d1", "d2", "d2", "d2"],
        label=[BENIGN, ATTACK, ATTACK, BENIGN, ATTACK, BENIGN],
        server_ip=["s", SOURCE_ONLY, "other", "s", "other", "s"],
        payload_packets=[3, 4, 5, 0, 6, 7]))


class HttpsEvalTests(unittest.TestCase):
    def test_partitions_exclude_payload0_and_other_victims(self):
        f = flows()
        self.assertEqual(partitions_for(f, "d1", "d2", "all").tolist(),
                         ["train", "train", "train", "excluded", "test", "test"])
        self.assertEqual(partitions_for(f, "d1", "d2", "only_242104").tolist(),
                         ["train", "train", "excluded", "excluded", "excluded", "test"])
        with self.assertRaises(ValueError):
            partitions_for(f, "d1", "d2", "nope")

    def test_subset_masks_split_benign_by_csv_status_and_attack_by_victim(self):
        test = flows().assign(midstream=[0, 0, 1, 0, 0, 1],
                              status_key_payload=["single_agree", "conflict", "unmatched", "multi_agree",
                                                  "conflict", "unmatched"])
        m = subset_masks(test, k=5)
        self.assertEqual(test.index[m["benign_csv_confirmed"]].tolist(), [0, 3])
        self.assertEqual(test.index[m["benign_csv_unmatched"]].tolist(), [5])
        self.assertEqual(test.index[m["attack_242104"]].tolist(), [1])
        self.assertEqual(test.index[m["attack_88_81"]].tolist(), [2, 4])
        self.assertEqual(test.index[m["short"]].tolist(), [0, 1, 3])

    def test_macro_f1_is_blank_when_a_class_is_missing(self):
        one_class = scored([BENIGN, BENIGN], [BENIGN, ATTACK])
        self.assertTrue(np.isnan(one_class["macro_f1"]))
        self.assertEqual(one_class["recall_benign"], 0.5)
        both = scored([BENIGN, ATTACK], [BENIGN, ATTACK])
        self.assertEqual(both["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
