"""완료된 시퀀스 대조의 클래스별 점수·변환 후 겹침을 집계한다. 재학습 없음."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.common.results import sha256_file
from src.track_a.preprocess import feature_groups
from src.track_a.sequence_controls import shuffle_packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("잘못된 run-id")
    from src.common.config import DATA_DIR, MACHINE
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs/results" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    baseline_id = manifest["baseline_run_id"]
    if not manifest["complete"] or not re.fullmatch(r"[A-Za-z0-9_-]+", baseline_id):
        raise ValueError("완료된 올바른 배치가 필요합니다")
    baseline = source.parent / baseline_id
    original_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    folder = DATA_DIR / "processed" / f"track_a_{original_manifest['preprocess_run_id']}"
    paths = {name: folder / name for name in ("features.csv", "rows.csv", "splits.csv")}
    paths.update(confusion=source / "confusion.csv", original_confusion=baseline / "confusion.csv",
                 manifest=source / "manifest.json", original_manifest=baseline / "manifest.json")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if any(hashes[name] != manifest["source_sha256"][name] for name in ("features.csv", "rows.csv", "splits.csv")):
        raise ValueError("분석 입력이 실험 당시와 다릅니다")
    out = source.parent / f"{args.run_id}_audit"
    if out.exists():
        raise FileExistsError("분석 결과를 덮어쓰지 않습니다")
    current = pd.read_csv(paths["confusion"])
    old = pd.read_csv(paths["original_confusion"])
    old = old.loc[old.feature_group.eq("sequence")]
    scores = []
    for split in ("random", "pattern_group"):
        for seed in (0, 1, 2):
            for model in ("rf", "lgbm"):
                parts = []
                for name, table in (("shuffled", current), ("original", old)):
                    part = table[(table.split == split) & (table.seed == seed) & (table.model == model)]
                    matrix = part.pivot(index="actual", columns="predicted", values="count").sort_index().sort_index(axis=1)
                    a = matrix.to_numpy()
                    parts.append(pd.DataFrame(dict(label=matrix.index, test_rows=a.sum(1),
                        **{f"{name}_f1": 2 * np.diag(a) / (a.sum(0) + a.sum(1)),
                           f"{name}_recall": np.diag(a) / a.sum(1)})).set_index("label"))
                joined = parts[0].join(parts[1].drop(columns="test_rows")).reset_index()
                joined["split"], joined["seed"], joined["model"] = split, seed, model
                joined["delta_f1"] = joined.shuffled_f1 - joined.original_f1
                scores.append(joined)
    x = pd.read_csv(paths["features.csv"], usecols=feature_groups()["sequence"]).loc[:, feature_groups()["sequence"]]
    registry, splits = pd.read_csv(paths["rows.csv"]), pd.read_csv(paths["splits.csv"])
    _, bags = np.unique(np.sort(x.to_numpy(), axis=1), axis=0, return_inverse=True)
    overlaps, coverage = [], []
    for seed in (0, 1, 2):
        _, ids = np.unique(shuffle_packets(x, seed).to_numpy(), axis=0, return_inverse=True)
        for split in ("random", "pattern_group"):
            p = splits[f"{split}_{seed}"]
            train, test = p.eq("train").to_numpy(), p.eq("test").to_numpy()
            seen, seen_bags = set(ids[train]), set(bags[train])
            hits = np.isin(ids, list(seen))
            overlaps.append(dict(split=split, seed=seed, transformed_overlap_patterns=len(seen & set(ids[test])),
                                 transformed_test_hit_rows=int(hits[test].sum()), test_rows=int(test.sum())))
            for label in manifest["known_labels"]:
                mask = test & registry.Label.eq(label).to_numpy()
                count = int(np.isin(bags[mask], list(seen_bags)).sum())
                coverage.append(dict(split=split, seed=seed, label=label, test_rows=int(mask.sum()),
                                     bag_hit_rows=count, bag_coverage=count / int(mask.sum())))
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("분석 중 입력이 바뀌었습니다")
    out.mkdir(parents=True)
    pd.concat(scores, ignore_index=True).to_csv(out / "paired_class_scores.csv", index=False)
    pd.DataFrame(overlaps).to_csv(out / "post_shuffle_overlap.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "bag_coverage.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(source_run_id=args.run_id, machine=MACHINE,
        source_sha256=hashes, audit_code_sha256=sha256_file(__file__), model_training=False,
        complete=True), indent=2) + "\n", encoding="utf-8")
    print("AUDIT COMPLETE:", out)


if __name__ == "__main__":
    main()
