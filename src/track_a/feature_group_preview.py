"""발표용 예비 비교. 캡디1 정제본의 전처리 누수를 제거한 정식 실험이 아니다.

Run: .venv/bin/python src/track_a/feature_group_preview.py [--model lgbm]
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


EXCLUDED = {
    "Shifu.A_None_TLS_CC", "Banker_None_TLS_CC", "Drixed_None_TLS_CC",
    "Vawtrak.C_None_TLS_CC", "golistmero_golistmero_TLS_scan",
}
HELD_OUT = {
    "nessus_nessus_TLS_scan", "arachni_arachni_TLS_scan",
    "Totbrick_None_TLS_CC", "Tiggre_None_TLS_CC", "TrickBotCC_None_TLS_CC",
}


def make_group_split(x, y, groups, seed):
    """성능을 보지 않고 최소 행 수와 시험 비율만으로 폴드를 선택한다.

    크기가 큰 그룹을 우선 배정하고, 같은 크기 그룹의 순서는 시드로 바꾼다.
    SGKF shuffle=True의 단순 첫 폴드는 이 데이터에서 클래스가 누락된다.
    """
    permuted = np.random.default_rng(seed).permutation(groups.max() + 1)[groups]
    labels = sorted(set(y))
    candidates = []
    for fold, (train, test) in enumerate(StratifiedGroupKFold(5, shuffle=False).split(x, y, permuted)):
        if all(min(int((y[train] == c).sum()), int((y[test] == c).sum())) >= 30 for c in labels):
            candidates.append((abs(len(test) / len(y) - 0.2), fold, train, test))
    if not candidates:
        raise ValueError("그룹 분할에서 모든 클래스의 학습·시험 30행 조건을 만족하는 폴드가 없습니다.")
    _, fold, train, test = min(candidates, key=lambda candidate: candidate[:2])
    print(f"GROUP seed={seed} selected_fold={fold} test_fraction={len(test)/len(y):.4f}", flush=True)
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf", "lgbm"], default="rf")
    args = parser.parse_args()
    import lightgbm

    source = Path(os.environ["DATA_DIR"]).expanduser() / "processed/malicious_tls_cleaned.csv"
    original = pd.read_csv(source)
    sequence = [str(i) for i in range(30)]
    fingerprint = [c for c in original if c.startswith("tls_")]
    stats = [c for c in original if c.startswith(("IP_", "payload_"))]
    all_features = [c for c in original if c != "Label"]
    features = {"all": all_features, "fingerprint": fingerprint, "stats": stats, "sequence": sequence}
    assert [len(v) for v in features.values()] == [117, 9, 78, 30]
    assert set(fingerprint + stats + sequence) == set(all_features)
    assert EXCLUDED | HELD_OUT <= set(original.Label)
    frame = original.loc[~original.Label.isin(EXCLUDED | HELD_OUT)].copy()
    labels = sorted(frame.Label.unique())
    assert len(labels) == 13 and "benign" in labels
    assert np.isfinite(frame[all_features].to_numpy()).all()
    # Exact tuples preserve identity without a hash collision risk.
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(frame[sequence]))
    y = frame.Label.to_numpy()
    assert not set(y) & (EXCLUDED | HELD_OUT)
    x = frame[all_features]
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    out = Path(__file__).resolve().parents[2] / "outputs/results"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"feature_group_preview_{date}_{args.model}"
    rows, counts, matrices = [], [], []
    allocations = pd.DataFrame({"source_row": frame.index, "pattern_id": groups})
    note = "캡디1 정제본, 재전처리 전, 예비; 18종 중 5종 보류, 알려진 13종; 전처리 누수 잔존"
    for seed in (0, 1, 2):
        random_split = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=seed)
        group_split = make_group_split(x, y, groups, seed)
        for split, (train, test) in (("random", random_split), ("pattern_group", group_split)):
            assert not set(train) & set(test)
            assert len(train) + len(test) == len(frame)
            assert set(y[train]) == set(labels) == set(y[test])
            overlap = len(set(groups[train]) & set(groups[test]))
            if split == "pattern_group":
                assert overlap == 0
            allocation = np.full(len(frame), "test", dtype=object)
            allocation[train] = "train"
            allocations[f"{split}_{seed}"] = allocation
            for label in labels:
                ntrain, ntest = int((y[train] == label).sum()), int((y[test] == label).sum())
                assert min(ntrain, ntest) >= 30, (split, seed, label, ntrain, ntest)
                counts.append(dict(split=split, seed=seed, label=label, train_rows=ntrain, test_rows=ntest,
                                   train_patterns=len(set(groups[train][y[train] == label])),
                                   test_patterns=len(set(groups[test][y[test] == label])),
                                   below_30=min(ntrain, ntest) < 30, overlap_patterns=overlap))
            pd.DataFrame(counts).to_csv(out / f"{stem}_class_counts.csv", index=False)
            print(f"SPLIT {split} seed={seed} train={len(train)} test={len(test)} overlap={overlap}", flush=True)
            for group_name, columns in features.items():
                if args.model == "rf":
                    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=4, class_weight="balanced")
                else:
                    model = lightgbm.LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=4,
                                                    class_weight="balanced", verbosity=-1,
                                                    deterministic=True, force_col_wise=True)
                start = time.monotonic()
                model.fit(x.iloc[train][columns], y[train])
                pred = model.predict(x.iloc[test][columns])
                row = dict(date=date, track="A", experiment="feature_group_preview", split=split,
                           feature_group=group_name, model=args.model, seed=seed,
                           macro_f1=f1_score(y[test], pred, labels=labels, average="macro", zero_division=0),
                           accuracy=accuracy_score(y[test], pred), machine="eunbin-mac",
                           sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, torch="", note=note,
                           n_features=len(columns), train_rows=len(train), test_rows=len(test),
                           train_seconds=time.monotonic()-start)
                rows.append(row)
                cm = confusion_matrix(y[test], pred, labels=labels)
                for i, actual in enumerate(labels):
                    for j, predicted in enumerate(labels):
                        matrices.append(dict(split=split, feature_group=group_name, model=args.model,
                                             seed=seed, actual=actual, predicted=predicted, count=int(cm[i, j])))
                pd.DataFrame(rows).to_csv(out / f"{stem}.csv", index=False)
                pd.DataFrame(matrices).to_csv(out / f"{stem}_confusion.csv", index=False)
                print(f"{len(rows)}/24 {group_name}: macro_f1={row['macro_f1']:.6f} accuracy={row['accuracy']:.6f} {row['train_seconds']:.1f}s", flush=True)
    # Per-row data stays outside Git, under DATA_DIR.
    allocation_dir = Path(os.environ["DATA_DIR"]).expanduser() / "processed" / f"preview_{date}"
    allocation_dir.mkdir(exist_ok=True)
    allocations.to_csv(allocation_dir / f"{stem}_splits.csv", index=False)
    result = pd.DataFrame(rows)
    summary = result.groupby(["model", "feature_group", "split"], sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), runs=("seed", "count")).reset_index()
    assert len(result) == 24 and summary.runs.eq(3).all()
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    metadata = dict(source="processed/malicious_tls_cleaned.csv", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_rows=len(original), used_rows=len(frame), labels=labels, excluded=sorted(EXCLUDED),
                    held_out_candidates=sorted(HELD_OUT), seed_changes_split_and_model=True,
                    group_split="Seed-permuted group IDs; StratifiedGroupKFold(5, shuffle=False); among folds with >=30 train/test rows per class, choose closest overall test fraction to 20%, tie: fold index; no model scores used",
                    std="sample standard deviation (ddof=1)", rf_workers=4, note=note)
    (out / f"{stem}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
