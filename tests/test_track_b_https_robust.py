import unittest

import numpy as np
import pandas as pd

from src.track_b.https_robust import KINDS, benign_overlap, transform


def flows(rows):
    columns = {f"seq_{i}": [r[i] if i < len(r) else 0 for r in rows] for i in range(30)}
    return pd.DataFrame(columns)


class HttpsRobustTests(unittest.TestCase):
    def test_logb_uses_plus_one_so_one_byte_is_not_padding(self):
        x = transform(flows([[1, 2, 3, 4, 7, 8, -1, -2896, 304, 308, 0, 5]]), 12, "logb").iloc[0, :12].tolist()
        self.assertEqual(x, [1, 2, 2, 3, 3, 4, -1, -12, 9, 9, 0, 3])

    def test_transform_only_touches_first_k_and_keeps_zero_padding(self):
        x = transform(flows([[304, -810, 80, 146, -55, 999]]), 3, "logb")
        self.assertEqual(x.iloc[0, :6].tolist(), [9, -10, 7, 0, 0, 0])
        self.assertTrue((x.iloc[0, 6:] == 0).all())

    def test_304_and_308_share_a_bucket_but_dir_drops_sizes(self):
        a, b = transform(flows([[304, -2896], [308, -2896]]), 2, "logb").to_numpy()[:, :2]
        self.assertEqual(a.tolist(), b.tolist())
        d = transform(flows([[304, -2896, 80]]), 3, "dir").iloc[0, :4].tolist()
        self.assertEqual(d, [1, -1, 1, 0])
        self.assertEqual(KINDS, ("logb", "dir"))

    def test_benign_overlap_counts_shared_servers(self):
        f = pd.DataFrame(dict(label=["benign"] * 5, payload_packets=[3] * 5,
                              day=["2021-04-11", "2021-04-11", "2021-04-16", "2021-04-16", "2021-04-16"],
                              server_ip=["a", "b", "a", "a", "c"]))
        row = benign_overlap(f).iloc[0]
        self.assertEqual((row.shared_servers, row.test_flows_to_shared_servers), (1, 2))


if __name__ == "__main__":
    unittest.main()
