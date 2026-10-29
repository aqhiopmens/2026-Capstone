"""순서 대조의 값 보존과 조회 기준선의 시험 라벨 누출을 검사한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.track_a.sequence_controls import hash_baseline, load_baseline, shuffle_packets


class ShuffleTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame([list(range(1, 11)) + [0] * 20 for _ in range(40)],
                              columns=[str(i) for i in range(30)], index=range(100, 140))
        self.x.iloc[:, 1] *= -1

    def test_multiset_direction_and_padding_are_preserved(self):
        shuffled = shuffle_packets(self.x, 0)
        np.testing.assert_array_equal(np.sort(shuffled, axis=1), np.sort(self.x, axis=1))
        np.testing.assert_array_equal(shuffled.eq(0), self.x.eq(0))
        self.assertTrue((shuffled != self.x).any().any())
        self.assertGreater(len(shuffled.drop_duplicates()), 1)  # 고정 열 순열이 아니다

    def test_row_ids_make_shuffle_reproducible_across_row_order_and_chunks(self):
        shuffled = shuffle_packets(self.x, 1)
        pd.testing.assert_frame_equal(shuffled, shuffle_packets(self.x, 1))
        pd.testing.assert_frame_equal(shuffled.iloc[10:20], shuffle_packets(self.x.iloc[10:20], 1))
        pd.testing.assert_frame_equal(shuffled, shuffle_packets(self.x.iloc[::-1], 1).iloc[::-1])
        self.assertFalse(shuffled.equals(shuffle_packets(self.x, 2)))

    def test_interior_zero_or_missing_value_requires_policy_review(self):
        for value in (0, np.nan):
            x = self.x.astype(float)
            x.iloc[0, 2] = value
            with self.assertRaises(ValueError):
                shuffle_packets(x, 0)


def hash_fixture():
    labels = pd.Series(["a", "a", "b", "b", "a", "b", "unseen"])
    partitions = pd.Series(["train"] * 4 + ["test", "test", "unknown"])
    patterns = pd.Series(["p", "q", "p", "r", "p", "new", "q"])
    return labels, partitions, patterns


class HashTests(unittest.TestCase):
    def run_hash(self, labels, partitions, patterns, split="random"):
        return hash_baseline(labels, partitions, patterns, known_labels=["a", "b"], split=split, seed=0)

    def test_train_only_conflict_and_unseen_fallback(self):
        y, p, g = hash_fixture()
        result = self.run_hash(y, p, g)
        self.assertEqual(result.predictions.predicted.tolist(), ["a", "a", "a"])
        self.assertEqual(result.predictions.lookup_hit.tolist(), [True, False, True])
        self.assertEqual(result.metrics["lookup_coverage"], .5)
        self.assertEqual(result.metadata["ambiguous_train_patterns"], 1)
        self.assertEqual(result.metadata["train_patterns"], 3)
        self.assertEqual(result.confusion["count"].sum(), 2)
        self.assertNotIn("max_probability", result.predictions)

    def test_changing_test_and_unknown_labels_cannot_change_lookup_predictions(self):
        y, p, g = hash_fixture()
        a = self.run_hash(y, p, g)
        y.iloc[4:7] = ["b", "a", "another-unseen"]
        b = self.run_hash(y, p, g)
        pd.testing.assert_series_equal(a.predictions.predicted, b.predictions.predicted)
        self.assertEqual(a.metadata["model_params"], b.metadata["model_params"])

    def test_group_test_is_all_unseen_and_scores_include_fallback_errors(self):
        y, p, g = hash_fixture()
        g.iloc[4:7] = ["new1", "new2", "new3"]
        result = self.run_hash(y, p, g, "pattern_group")
        self.assertEqual(result.metrics["lookup_coverage"], 0)
        self.assertIsNone(result.metrics["lookup_hit_accuracy"])
        self.assertAlmostEqual(result.metrics["accuracy"], .5)
        self.assertAlmostEqual(result.metrics["macro_f1"], 1 / 3)

    def test_group_overlap_is_not_permitted(self):
        y, p, g = hash_fixture()
        with self.assertRaises(ValueError):
            self.run_hash(y, p, g, "pattern_group")


class BaselineTests(unittest.TestCase):
    def test_cross_machine_or_incomplete_baseline_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "outputs/results/baseline"
            folder.mkdir(parents=True)
            manifest = dict(complete=True, completed_runs=48, machine="other-machine",
                            source_sha256={}, code_sha256={})
            for changes in ({}, dict(machine="unit-test", complete=False),
                            dict(machine="unit-test", source_sha256={"raw": "changed"})):
                (folder / "manifest.json").write_text(json.dumps({**manifest, **changes}), encoding="utf-8")
                with patch("src.track_a.sequence_controls.ROOT", root), patch(
                        "src.track_a.sequence_controls.baseline_code_hashes", return_value={}):
                    with self.assertRaises(ValueError):
                        load_baseline("baseline", machine="unit-test", hashes={}, workers=2)


if __name__ == "__main__":
    unittest.main()
