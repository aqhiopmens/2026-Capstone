import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.track_b.https_flows import ATTACK, BENIGN, label_flow
from src.track_b.https_flows_2022 import CAPTURES_2022, read_csv_2022


class Https2022Tests(unittest.TestCase):
    def test_label_rule_uses_attacker_client_ip_not_victim_alone(self):
        capture = next(iter(CAPTURES_2022.values()))
        self.assertEqual(label_flow("203.178.143.112", "184.0.48.150", capture), ATTACK)
        self.assertEqual(label_flow("203.178.143.112", "8.8.8.8", capture), BENIGN)
        self.assertEqual(label_flow("140.213.164.207", "184.0.48.150", capture), BENIGN)  # 정상도 피해자 서버로 감

    def test_csv_loader_keeps_443_and_maps_columns(self):
        frame = pd.DataFrame(dict(originh=["a", "b"], originp=[1, 2], responh=["v", "v"], responp=[443, 80],
                                  attack_category=["Bruteforce", "Benign"], fwd_data_pkts_tot=[3, 1], bwd_data_pkts_tot=[4, 1]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            frame.to_csv(path, index=False)
            out = read_csv_2022(path)
        self.assertEqual(len(out), 1)
        self.assertEqual((out.traffic_category.iloc[0], int(out.data_pkts.iloc[0])), ("Bruteforce", 7))


if __name__ == "__main__":
    unittest.main()
