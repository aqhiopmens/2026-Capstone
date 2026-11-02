import unittest

import pandas as pd

from src.track_b.https_flows import ATTACK, BENIGN, label_flow, match_status, reaudit


class HttpsLabelAuditTests(unittest.TestCase):
    def test_match_status_is_symmetric_across_labels(self):
        self.assertEqual(match_status(BENIGN, []), "unmatched")
        self.assertEqual(match_status(BENIGN, ["Benign"]), "single_agree")
        self.assertEqual(match_status(BENIGN, ["Benign", "Background"]), "multi_agree")
        self.assertEqual(match_status(ATTACK, ["Bruteforce"]), "single_agree")
        self.assertEqual(match_status(ATTACK, ["Bruteforce", "Benign"]), "conflict")
        self.assertEqual(match_status(ATTACK, ["Benign"]), "single_disagree")

    def test_label_uses_capture_rule_with_two_victims(self):
        capture = dict(attacker="a", victims=("v1", "v2"))
        self.assertEqual(label_flow("a", "v2", capture), ATTACK)
        self.assertEqual(label_flow("a", "other", capture), BENIGN)
        self.assertEqual(label_flow("b", "v1", capture), BENIGN)

    def test_reaudit_counts_key_and_payload_criteria(self):
        flows = pd.DataFrame([
            dict(pcap="p", flow_id=0, label=ATTACK, client_ip="a", client_port=1, server_ip="v", payload_packets=5),
            dict(pcap="p", flow_id=1, label=BENIGN, client_ip="c", client_port=2, server_ip="s", payload_packets=3),
            dict(pcap="p", flow_id=2, label=BENIGN, client_ip="c", client_port=3, server_ip="s", payload_packets=3)])
        csv = pd.DataFrame([
            dict(originh="a", originp=1, responh="v", traffic_category="Bruteforce", data_pkts=5),
            dict(originh="a", originp=1, responh="v", traffic_category="Benign", data_pkts=5),
            dict(originh="c", originp=2, responh="s", traffic_category="Benign", data_pkts=9)])
        status, table = reaudit(flows, csv)
        self.assertEqual(status.status_key_payload.tolist(), ["conflict", "unmatched", "unmatched"])
        self.assertEqual(status.status_key.tolist(), ["conflict", "single_agree", "unmatched"])
        benign = table[table.label == BENIGN].iloc[0]
        self.assertEqual((benign.key_flows, benign.key_payload_flows), (1, 0))


if __name__ == "__main__":
    unittest.main()
