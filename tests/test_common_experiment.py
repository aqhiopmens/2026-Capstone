"""누수 방지·평가 분모·산출물 격리를 확인한다. 정식 실험은 실행하지 않는다."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common.evaluation import evaluate_classification
from src.common.experiment import run_classification
from src.common.models import make_model
from src.common.results import save_run, sha256_file, summarize_runs
from src.common.splits import split_validation, validate_partitions


def fixture():
    labels = pd.Series(["a"] * 300 + ["b"] * 300 + ["unseen"] * 20)
    partitions = pd.Series("train", index=labels.index)
    partitions.iloc[240:300] = "test"
    partitions.iloc[540:600] = "test"
    partitions.iloc[600:] = "unknown"
    groups = pd.Series(np.arange(len(labels)) // 2, index=labels.index)
    x = pd.DataFrame({"signal": np.r_[np.zeros(300), np.ones(300), np.full(20, 99.)],
                      "noise": np.arange(len(labels)) % 7})
    return x, labels, partitions, groups


class SplitTests(unittest.TestCase):
    def test_both_validation_splits_preserve_external_rows_and_all_classes(self):
        _, y, p, g = fixture()
        for split in ("random", "pattern_group"):
            a = split_validation(y, p, split=split, seed=0, groups=g)
            b = split_validation(y, p, split=split, seed=0, groups=g)
            pd.testing.assert_series_equal(a, b)
            pd.testing.assert_series_equal(a[p.ne("train")], p[p.ne("train")])
            self.assertEqual(set(y[a.eq("validation")]), {"a", "b"})
            validate_partitions(y, a, ["a", "b"], g if split == "pattern_group" else None)

    def test_cross_label_shared_group_is_not_split(self):
        _, y, p, g = fixture()
        g.iloc[300:302] = g.iloc[0]
        result = split_validation(y, p, split="pattern_group", seed=2, groups=g)
        self.assertEqual(result.iloc[0], result.iloc[300])

    def test_external_overlap_and_missing_group_fail(self):
        _, y, p, g = fixture()
        g.iloc[600] = g.iloc[0]
        with self.assertRaises(ValueError):
            split_validation(y, p, split="pattern_group", seed=0, groups=g)
        with self.assertRaises(ValueError):
            split_validation(y, p, split="pattern_group", seed=0)

    def test_small_classes_and_repeated_validation_fail(self):
        _, y, p, _ = fixture()
        with self.assertRaises(ValueError):
            split_validation(y, p, split="random", seed=0, minimum=60)
        p = split_validation(y, p, split="random", seed=0)
        with self.assertRaises(ValueError):
            split_validation(y, p, split="random", seed=0)

    def test_alignment_and_known_unknown_contamination_fail(self):
        _, y, p, _ = fixture()
        with self.assertRaises(ValueError):
            validate_partitions(y, p.iloc[::-1], ["a", "b"])
        p.iloc[0] = "unknown"
        with self.assertRaises(ValueError):
            validate_partitions(y, p, ["a", "b"])


class EvaluationTests(unittest.TestCase):
    def test_imbalance_does_not_turn_macro_f1_into_accuracy(self):
        actual = ["a"] * 9 + ["b"]
        predicted = ["a"] * 10
        scores, cm = evaluate_classification(actual, predicted, labels=["b", "a"])
        self.assertAlmostEqual(scores["accuracy"], .9)
        self.assertAlmostEqual(scores["macro_f1"], (18 / 19) / 2)
        self.assertEqual(len(cm), 4)
        self.assertEqual(cm["count"].sum(), 10)
        self.assertEqual(cm.iloc[0].to_dict(), dict(actual="b", predicted="b", count=0))

    def test_missing_class_and_unknown_predictions_fail(self):
        with self.assertRaises(ValueError):
            evaluate_classification(["a"], ["a"], labels=["a", "b"])
        with self.assertRaises(ValueError):
            evaluate_classification(["a", "b"], ["a", "unseen"], labels=["a", "b"])


class RunnerTests(unittest.TestCase):
    def test_real_models_fit_only_training_rows_and_unknown_not_scored(self):
        x, y, p, g = fixture()
        p = split_validation(y, p, split="pattern_group", seed=0, groups=g)
        for name in ("rf", "lgbm"):
            estimator = make_model(name, seed=0, workers=1)
            original_fit = estimator.fit
            fitted = []

            def spy_fit(fit_x, fit_y):
                fitted.append(fit_x.index.tolist())
                self.assertEqual(set(fit_y), {"a", "b"})
                return original_fit(fit_x, fit_y)

            with patch("src.common.experiment.make_model", return_value=estimator), patch.object(
                    estimator, "fit", side_effect=spy_fit):
                result = run_classification(x, y, p, known_labels=["a", "b"],
                                            model_name=name, seed=0, split="pattern_group",
                                            groups=g, workers=1)
            self.assertEqual(fitted, [x.index[p.eq("train")].tolist()])
            self.assertEqual(result.metrics["test_rows"], 120)
            self.assertEqual(result.metrics["unknown_rows"], 20)
            self.assertEqual(result.confusion["count"].sum(), 120)
            test = result.predictions.query("partition == 'test'")
            self.assertAlmostEqual(result.metrics["macro_f1"], f1_score(
                test.actual, test.predicted, average="macro"))
            self.assertEqual(len(result.assignments), len(x))
            self.assertTrue(result.predictions.max_probability.between(0, 1).all())

    def test_metadata_feature_and_nonfinite_rejected_before_fit(self):
        x, y, p, _ = fixture()
        for bad_x in (x.assign(Label=y), x.assign(signal=np.inf), x.iloc[::-1]):
            with patch("src.common.experiment.make_model") as factory:
                with self.assertRaises(ValueError):
                    run_classification(bad_x, y, p, known_labels=["a", "b"],
                                       model_name="rf", seed=0, split="random")
                factory.assert_not_called()


class StorageTests(unittest.TestCase):
    def test_roundtrip_separation_and_no_overwrite(self):
        x, y, p, _ = fixture()
        result = run_classification(x, y, p, known_labels=["a", "b"],
                                    model_name="rf", seed=0, split="random", workers=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            source.write_text("test data\n", encoding="utf-8")
            with patch.dict(os.environ, DATA_DIR=str(root / "data"), MACHINE="unit-test"):
                from src.common import config
            with patch.object(config, "DATA_DIR", root / "data"), patch.object(
                    config, "MACHINE", "unit-test"), patch(
                    "src.common.results.__file__", str(root / "repo/src/common/results.py")), patch(
                    "src.common.results.subprocess.check_output", side_effect=lambda args, **kw:
                    "abc123\n" if args[1] == "rev-parse" else ""):
                args = dict(run_id="synthetic", track="A", experiment="unit-test",
                            feature_group="synthetic", split_seed=1,
                            source_hashes={"input": sha256_file(source)}, note="not research results")
                report, data = save_run(result, **args)
                self.assertEqual({f.name for f in report.iterdir()},
                                 {"metrics.csv", "confusion.csv", "metadata.json"})
                saved = pd.read_csv(data / "predictions.csv")
                pd.testing.assert_frame_equal(saved, result.predictions)
                self.assertEqual(len(pd.read_csv(data / "assignments.csv")), len(x))
                metric = pd.read_csv(report / "metrics.csv").iloc[0]
                self.assertEqual(metric.machine, "unit-test")
                self.assertEqual(metric.seed, 0)
                self.assertEqual(metric.split_seed, 1)
                with self.assertRaises(FileExistsError):
                    save_run(result, **args)
                with self.assertRaises(ValueError):
                    save_run(result, **{**args, "run_id": "../escape"})
                with patch.object(config, "DATA_DIR", root / "repo/data"):
                    with self.assertRaises(ValueError):
                        save_run(result, **{**args, "run_id": "unsafe-data"})

    def test_summary_requires_three_distinct_runs_per_environment(self):
        rows = pd.DataFrame([dict(track="A", experiment="unit-test", split="random",
                                 feature_group="all", model="rf", machine="unit-test",
                                 sklearn="test", lightgbm="test", torch="", seed=i,
                                 split_seed=i, macro_f1=score, accuracy=score)
                             for i, score in enumerate((.6, .7, .8))])
        summary = summarize_runs(rows).iloc[0]
        self.assertAlmostEqual(summary.macro_f1_mean, .7)
        self.assertAlmostEqual(summary.macro_f1_std, .1)
        for invalid in (rows.iloc[:2], pd.concat([rows, rows.iloc[:1]]),
                        rows.assign(machine=["unit-test", "unit-test", "another-machine"])):
            with self.assertRaises(ValueError):
                summarize_runs(invalid)


if __name__ == "__main__":
    unittest.main()
