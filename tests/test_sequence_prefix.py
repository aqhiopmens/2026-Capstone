"""관측 범위·저장 예측 정합·부분집합 분모와 실패 시 중단을 검사한다."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.track_a.sequence_prefix_analysis import (
    audit_prefix, prefix_input, score_subsets, summarize_subsets, validate_predictions,
)
from src.track_a.sequence_prefix_experiment import conditions, load_baseline_predictions, run_batch


def sequence_fixture():
    return pd.DataFrame([[10, -20, 30, -40, 50, tail] + [0] * 24 for tail in (6, 7, 8, 9)],
                        columns=[str(i) for i in range(30)], index=[10, 11, 12, 13])


class PrefixTests(unittest.TestCase):
    def test_observation_budget_preserves_sign_padding_and_source(self):
        x = sequence_fixture()
        original = x.copy(deep=True)
        for k in (5, 10, 20, 30):
            out = prefix_input(x, k)
            pd.testing.assert_frame_equal(out.iloc[:, :k], x.iloc[:, :k])
            self.assertTrue(out.iloc[:, k:].eq(0).all().all())
            self.assertEqual(out.shape, x.shape)
        pd.testing.assert_frame_equal(prefix_input(x, 30), original)
        pd.testing.assert_frame_equal(x, original)
        changed = x.copy()
        changed.iloc[:, 5] = 999
        pd.testing.assert_frame_equal(prefix_input(x, 5), prefix_input(changed, 5))

    def test_invalid_sequence_or_k_fails(self):
        x = sequence_fixture()
        for k in (0, 31, True, 5.0):
            with self.assertRaises(ValueError):
                prefix_input(x, k)
        for value in (0, 1.5, np.nan):
            changed = x.astype(float)
            changed.iloc[0, 2] = value
            with self.assertRaises(ValueError):
                prefix_input(changed, 5)
        with self.assertRaises(ValueError):
            prefix_input(x.iloc[::-1].set_axis([1, 1, 2, 3]), 5)

    def test_original_group_disjoint_can_have_shared_prefix_and_short_counts(self):
        x = sequence_fixture()
        labels = pd.Series(["a", "a", "b", "u"], index=x.index)
        parts = pd.Series(["train", "test", "test", "unknown"], index=x.index)
        masks5, report5 = audit_prefix(x, labels, parts, split="pattern_group", seed=0, k=5)
        masks30, report30 = audit_prefix(x, labels, parts, split="pattern_group", seed=0, k=30)
        self.assertTrue(masks5["seen"].all())
        self.assertFalse(masks30["seen"].loc[parts.eq("test")].any())
        test5 = report5.query("partition == 'test' and label == '__all__'").iloc[0]
        test30 = report30.query("partition == 'test' and label == '__all__'").iloc[0]
        self.assertEqual((test5.rows, test5.reached_k_rows, test5.short_rows), (2, 2, 0))
        self.assertEqual((test30.rows, test30.reached_k_rows, test30.short_rows), (2, 0, 2))


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.pred = pd.DataFrame(dict(row_id=[10, 11, 12, 13, 14], partition=["test"] * 4 + ["unknown"],
            actual=["a", "a", "b", "c", "u"], predicted=["a", "b", "b", "c", "a"], max_probability=[.8] * 5))
        self.masks = dict(reached=pd.Series([True, False, False, False, True], index=range(10, 15)),
                          seen=pd.Series([True] * 5, index=range(10, 15)))
        self.identity = dict(machine="unit", split="random", seed=0, model="rf", k=5)

    def reports(self):
        return score_subsets(self.pred, known_labels=["a", "b", "c"], masks=self.masks, identity=self.identity)

    def test_missing_classes_fixed_macro_and_empty_subset_are_not_hidden(self):
        scores, classes, matrices = self.reports()
        all_row = scores.query("subset == 'all'").iloc[0]
        self.assertAlmostEqual(all_row.macro_f1, f1_score(["a", "a", "b", "c"], ["a", "b", "b", "c"], average="macro"))
        reached = scores.query("subset == 'reached_k'").iloc[0]
        self.assertAlmostEqual(reached.macro_f1, 1 / 3)
        self.assertEqual(reached.present_classes, 1)
        missing = classes.query("subset == 'reached_k' and label == 'b'").iloc[0]
        self.assertTrue(pd.isna(missing.recall))
        self.assertEqual(missing.f1, 0)
        self.assertTrue(classes.query("subset == 'reached_k' and label == 'a'").iloc[0].low_support)
        empty = scores.query("subset == 'prefix_unseen'").iloc[0]
        self.assertEqual(empty.rows, 0)
        self.assertTrue(pd.isna(empty.macro_f1))
        self.assertEqual(matrices.query("subset == 'all'")["count"].sum(), 4)  # unknown은 분모 밖
        for pair in (("reached_k", "short"), ("prefix_seen", "prefix_unseen")):
            self.assertEqual(matrices.loc[matrices.subset.isin(pair), "count"].sum(), 4)

    def test_subset_summary_keeps_empty_runs_and_rejects_duplicate_or_missing_seeds(self):
        scores = self.reports()[0]
        frame = pd.concat([scores.assign(seed=seed) for seed in (0, 1, 2)])
        summary = summarize_subsets(frame)
        self.assertEqual(summary.query("subset == 'prefix_unseen'").iloc[0].valid_runs, 0)
        with self.assertRaises(ValueError):
            summarize_subsets(pd.concat([frame, scores]))
        with self.assertRaises(ValueError):
            summarize_subsets(frame.loc[frame.seed.ne(2)])

    def test_saved_predictions_require_exact_ids_labels_assignments_and_probabilities(self):
        labels = pd.Series(["a", "a", "b", "c", "u"], index=range(10, 15))
        parts = pd.Series(["test"] * 4 + ["unknown"], index=labels.index)
        assignments = pd.DataFrame(dict(row_id=labels.index, partition=parts.to_numpy()))
        validate_predictions(self.pred, assignments, labels, parts, ["a", "b", "c"])
        for column, value in (("actual", "wrong"), ("partition", "train"), ("max_probability", 1.1)):
            pred = self.pred.copy()
            pred.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_predictions(pred, assignments, labels, parts, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            validate_predictions(self.pred.iloc[:-1], assignments, labels, parts, ["a", "b", "c"])


class DriverTests(unittest.TestCase):
    def test_conditions_and_failed_fit_leave_incomplete_manifest_and_no_summary(self):
        self.assertEqual(len(set(conditions())), 36)
        x = sequence_fixture()
        frame = x.assign(Label=["a", "a", "b", "u"])
        rows = pd.DataFrame(dict(pattern_id=["p0", "p1", "p2", "p3"]))
        parts = ["train", "test", "test", "unknown"]
        splits = pd.DataFrame({f"{split}_{seed}": parts for split in ("random", "pattern_group") for seed in (0, 1, 2)}, index=x.index)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, DATA_DIR=str(root / "data"), MACHINE="unit"):
                from src.common import config
            with patch.object(config, "DATA_DIR", root / "data"), patch.object(config, "MACHINE", "unit"), patch(
                    "src.track_a.sequence_prefix_experiment.ROOT", root / "repo"), patch(
                    "src.track_a.sequence_prefix_experiment.load_inputs", return_value=(frame, rows, splits, ["a", "b"], {}, {})), patch(
                    "src.track_a.sequence_prefix_experiment.load_baseline", return_value=(pd.DataFrame(), {})), patch(
                    "src.track_a.sequence_prefix_experiment.load_baseline_predictions", return_value=({}, {})), patch(
                    "src.track_a.sequence_prefix_experiment.code_hashes", return_value={}), patch(
                    "src.track_a.sequence_prefix_experiment.subprocess.check_output", return_value="unit"), patch(
                    "src.track_a.sequence_prefix_experiment.run_classification", side_effect=RuntimeError("fit failed")):
                with self.assertRaisesRegex(RuntimeError, "fit failed"):
                    run_batch(run_id="interrupted", baseline_run_id="baseline")
                manifest = json.loads((root / "repo/outputs/results/interrupted/manifest.json").read_text(encoding="utf-8"))
                self.assertFalse(manifest["complete"])
                self.assertEqual(manifest["completed_runs"], 0)
                self.assertFalse((root / "repo/outputs/results/interrupted/summary.csv").exists())
                with self.assertRaises(FileExistsError):
                    run_batch(run_id="interrupted", baseline_run_id="baseline")


if __name__ == "__main__":
    unittest.main()
