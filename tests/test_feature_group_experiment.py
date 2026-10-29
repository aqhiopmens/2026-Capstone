"""정식 실행기의 입력 손상·배정 변경·불완전 실행을 검사한다."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.track_a.feature_group_experiment import conditions, run_batch, validate_tables
from src.track_a.preprocess import build_assignment, encode_frame, pattern_ids, row_roles
from test_track_a_preprocess import fixture


def tables():
    frame = encode_frame(fixture())
    ids = pattern_ids(frame)
    rows = pd.DataFrame(dict(source_row=range(len(frame)), Label=frame.Label,
                             pattern_id=ids, role=row_roles(frame)))
    splits = pd.DataFrame(dict(source_row=range(len(frame))))
    for split in ("random", "pattern_group"):
        for seed in (0, 1, 2):
            assignment, _ = build_assignment(frame, ids, split, seed)
            splits[f"{split}_{seed}"] = assignment.partition
    return frame.reset_index(names="source_row"), rows, splits


class InputTests(unittest.TestCase):
    def setUp(self):
        self.features, self.rows, self.splits = tables()

    def test_input_contract_preserves_all_features_rows_and_six_splits(self):
        validated = validate_tables(self.features, self.rows, self.splits)
        pd.testing.assert_frame_equal(validated, self.features.drop(columns="source_row"))
        self.assertEqual(len(set(conditions())), 48)

    def test_reordered_or_duplicated_row_ids_fail(self):
        for key in ("features", "rows", "splits"):
            data = dict(features=self.features.copy(), rows=self.rows.copy(), splits=self.splits.copy())
            data[key].loc[1, "source_row"] = 0
            with self.assertRaises(ValueError):
                validate_tables(**data)
        with self.assertRaises(ValueError):
            validate_tables(self.features.iloc[::-1], self.rows, self.splits)

    def test_changed_role_label_or_sequence_fails(self):
        for column, value in (("role", "unknown"), ("Label", "changed"), ("pattern_id", "changed")):
            rows = self.rows.copy()
            rows.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_tables(self.features, rows, self.splits)
        changed = self.features.copy()
        changed.loc[0, "0"] = 100000
        with self.assertRaises(ValueError):
            validate_tables(changed, self.rows, self.splits)

    def test_valid_but_different_random_allocation_is_not_the_claimed_seed(self):
        changed = self.splits.copy()
        mask = self.rows.Label.eq(self.rows.Label.iloc[0])
        train = changed.index[mask & changed.random_0.eq("train")][0]
        test = changed.index[mask & changed.random_0.eq("test")][0]
        changed.loc[train, "random_0"], changed.loc[test, "random_0"] = "test", "train"
        with self.assertRaisesRegex(ValueError, "시드"):
            validate_tables(self.features, self.rows, changed)

    def test_failed_training_leaves_incomplete_manifest_and_no_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            with patch.dict(os.environ, DATA_DIR=str(root / "data"), MACHINE="unit-test"):
                from src.common import config
            f = self.features.drop(columns="source_row")
            with patch.object(config, "DATA_DIR", root / "data"), patch.object(config, "MACHINE", "unit-test"), patch(
                    "src.track_a.feature_group_experiment.ROOT", repo), patch(
                    "src.track_a.feature_group_experiment.load_inputs",
                    return_value=(f, self.rows, self.splits, ["benign", "BuerLoader_None_TLS_CC"], {}, {})), patch(
                    "src.track_a.feature_group_experiment.code_hashes", return_value={}), patch(
                    "src.track_a.feature_group_experiment.subprocess.check_output", return_value="test\n"), patch(
                    "src.track_a.feature_group_experiment.run_classification", side_effect=RuntimeError("fit failed")):
                with self.assertRaisesRegex(RuntimeError, "fit failed"):
                    run_batch(run_id="interrupted")
                report = repo / "outputs/results/interrupted"
                manifest = json.loads((report / "manifest.json").read_text())
                self.assertFalse(manifest["complete"])
                self.assertEqual(manifest["completed_runs"], 0)
                self.assertFalse((report / "summary.csv").exists())
                with self.assertRaises(FileExistsError):
                    run_batch(run_id="interrupted")


if __name__ == "__main__":
    unittest.main()
