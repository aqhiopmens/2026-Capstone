"""원본 재전처리의 데이터 손실·분할 누수를 막는 계약 검사."""

import unittest

import pandas as pd

from src.track_a.preprocess import (
    EXCLUDED, HELD_OUT, feature_groups, encode_frame, row_roles,
    pattern_ids, build_assignment, validate_assignment,
)


def fixture():
    columns = feature_groups()["all"]
    records = []
    for cls, label in enumerate(("benign", "BuerLoader_None_TLS_CC", sorted(HELD_OUT)[0], sorted(EXCLUDED)[0])):
        for i in range(200):
            row = dict.fromkeys(columns, 0)
            row.update({"Label": label, "tls_version": "0x0303", "tls_chosen_cipher_suit": "0x1301",
                        "0": cls * 1000 + i + 1, "IP_total_bytes": 100 + i})
            records.append(row)
    return pd.DataFrame(records)


class EncodingTests(unittest.TestCase):
    def test_hex_and_numeric_values_preserved_without_fit(self):
        raw = fixture()
        raw["tls_cipher_suites_4"] = raw["tls_cipher_suites_4"].astype(object)
        raw.loc[0, "tls_cipher_suites_4"] = "0x0035"
        raw.loc[1, "tls_cipher_suites_4"] = "53"
        raw.loc[0, "IP_total_bytes"] = 10**15
        raw.loc[1] = raw.loc[0]
        encoded = encode_frame(raw)
        self.assertEqual(encoded.loc[0, "tls_version"], 771)
        self.assertEqual(encoded.loc[0, "tls_cipher_suites_4"], 53)
        self.assertEqual(encoded.loc[0, "IP_total_bytes"], 10**15)
        self.assertEqual(len(encoded), len(raw))
        self.assertTrue(encoded.loc[0].equals(encoded.loc[1]))

    def test_unknown_hex_token_is_not_silently_zeroed(self):
        raw = fixture()
        raw.loc[0, "tls_version"] = "broken"
        with self.assertRaises(ValueError):
            encode_frame(raw)

    def test_fractional_tls_identifier_is_not_truncated(self):
        raw = fixture()
        raw["tls_cipher_suites_4"] = raw["tls_cipher_suites_4"].astype(float)
        raw.loc[0, "tls_cipher_suites_4"] = 1.5
        with self.assertRaises(ValueError):
            encode_frame(raw)

    def test_nonfinite_and_nonintegral_sequence_rejected(self):
        for column, value in (("0", 1.5), ("IP_total_bytes", float("inf"))):
            raw = fixture()
            raw[column] = raw[column].astype(float)
            raw.loc[0, column] = value
            with self.assertRaises(ValueError):
                encode_frame(raw)

    def test_negative_direction_is_valid_but_negative_stats_are_quarantined(self):
        f = encode_frame(fixture())
        f["IP_fwd_arrtime_diff_Min"] = f["IP_fwd_arrtime_diff_Min"].astype(float)
        f.loc[0, "0"] = -100
        f.loc[1, "IP_fwd_arrtime_diff_Min"] = -0.5
        roles = row_roles(f)
        self.assertEqual(roles.iloc[0], "known")
        self.assertEqual(roles.iloc[1], "invalid")
        # A zero duration alone is not sufficient evidence of corruption.
        self.assertEqual(roles.iloc[2], "known")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.f = encode_frame(fixture())
        self.patterns = pattern_ids(self.f)

    def test_group_split_is_complete_reproducible_and_disjoint(self):
        a, _ = build_assignment(self.f, self.patterns, "pattern_group", 0)
        b, _ = build_assignment(self.f, self.patterns, "pattern_group", 0)
        pd.testing.assert_frame_equal(a, b)
        validate_assignment(self.f, a, "pattern_group")
        self.assertEqual(len(a), len(self.f))
        self.assertTrue(a.loc[a.Label.isin(HELD_OUT), "partition"].eq("unknown").all())
        self.assertTrue(a.loc[a.Label.isin(EXCLUDED), "partition"].eq("excluded").all())

    def test_held_out_row_in_train_is_rejected(self):
        a, _ = build_assignment(self.f, self.patterns, "pattern_group", 0)
        a.loc[a.Label.isin(HELD_OUT), "partition"] = "train"
        with self.assertRaises(ValueError):
            validate_assignment(self.f, a, "pattern_group")

    def test_shared_pattern_across_labels_stays_together(self):
        # Same pattern in two known classes must remain one global group.
        self.f.loc[200, "0"] = self.f.loc[0, "0"]
        p = pattern_ids(self.f)
        a, _ = build_assignment(self.f, p, "pattern_group", 1)
        self.assertEqual(a.loc[0, "partition"], a.loc[200, "partition"])

    def test_pattern_overlap_with_unknown_is_rejected(self):
        self.f.loc[400, "0"] = self.f.loc[0, "0"]
        with self.assertRaises(ValueError):
            build_assignment(self.f, pattern_ids(self.f), "pattern_group", 0)

    def test_invalid_assignment_or_too_small_class_is_rejected(self):
        a, _ = build_assignment(self.f, self.patterns, "random", 0)
        a.loc[0, "source_row"] = a.loc[1, "source_row"]
        with self.assertRaises(ValueError):
            validate_assignment(self.f, a, "random")
        small = self.f.iloc[:40].copy()
        with self.assertRaises(ValueError):
            build_assignment(small, pattern_ids(small), "random", 0)


if __name__ == "__main__":
    unittest.main()
