"""집계 후 새 체크아웃·원본 기록 보존·불일치 시 중단을 검사한다."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from src.track_a.result_metadata import compact_batch, metadata_bytes, read_archive


class MetadataArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.folder = root / "batch"
        self.folder.mkdir()
        self.original = {}
        metrics, matrices = [], []
        for seed in (0, 1):
            child = f"batch_rf_s{seed}"
            path = root / child
            path.mkdir()
            identity = dict(split="random", seed=seed, model="rf", feature_group="sequence")
            meta = dict(identity=identity, note="은빈·원본 기록", params={"workers": 2})
            raw = (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()
            if seed:
                raw = raw.replace(b"\n", b"\r\n")
            self.original[child] = raw
            (path / "metadata.json").write_bytes(raw)
            metric = pd.DataFrame([{**identity, "macro_f1": .75}])
            matrix = pd.DataFrame([{**identity, "actual": "a", "predicted": "a", "count": 3}])
            metric.to_csv(path / "metrics.csv", index=False)
            matrix.to_csv(path / "confusion.csv", index=False)
            metrics.append(metric)
            matrices.append(matrix)
        pd.concat(metrics).to_csv(self.folder / "metrics.csv", index=False)
        pd.concat(matrices).to_csv(self.folder / "confusion.csv", index=False)
        manifest = dict(run_id="batch", complete=True, expected_runs=2, completed_runs=2,
                        run_ids=list(self.original))
        (self.folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_exact_metadata_survives_without_child_folders_and_is_idempotent(self):
        path = compact_batch(self.folder)
        before = path.read_bytes()
        archive = json.loads(before)
        for child, raw in self.original.items():
            self.assertEqual(metadata_bytes(archive["runs"][child]), raw)
            shutil.rmtree(self.folder.parent / child)
        self.assertEqual(len(read_archive(self.folder)), 2)
        compact_batch(self.folder)
        self.assertEqual(path.read_bytes(), before)

    def test_changed_child_score_or_confusion_aborts_before_archive(self):
        for name, column in (("metrics.csv", "macro_f1"), ("confusion.csv", "count")):
            with self.subTest(name=name):
                path = self.folder.parent / "batch_rf_s0" / name
                original = path.read_bytes()
                frame = pd.read_csv(path)
                frame[column] = 999
                frame.to_csv(path, index=False)
                with self.assertRaises(AssertionError):
                    compact_batch(self.folder)
                self.assertFalse((self.folder / "run_metadata.json").exists())
                path.write_bytes(original)

    def test_tampered_manifest_or_aggregates_are_rejected(self):
        compact_batch(self.folder)
        for name in ("manifest.json", "metrics.csv", "confusion.csv"):
            with self.subTest(name=name):
                path = self.folder / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    read_archive(self.folder)
                path.write_bytes(original)

    def test_tampered_metadata_is_rejected(self):
        path = compact_batch(self.folder)
        archive = json.loads(path.read_text())
        archive["runs"]["batch_rf_s0"]["metadata"]["params"]["workers"] = 99
        path.write_text(json.dumps(archive))
        with self.assertRaises(ValueError):
            read_archive(self.folder)

    def test_path_traversal_and_incomplete_batch_are_rejected(self):
        path = self.folder / "manifest.json"
        original = json.loads(path.read_text())
        for update in ({"run_ids": ["../outside", "batch_rf_s1"]}, {"complete": False}):
            path.write_text(json.dumps({**original, **update}))
            with self.assertRaises(ValueError):
                compact_batch(self.folder)


if __name__ == "__main__":
    unittest.main()
