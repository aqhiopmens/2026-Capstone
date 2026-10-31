"""초기 패킷 입력과 행 배정을 고정한 감사·부분집합 평가. 학습·파일 쓰기는 하지 않는다."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.track_a.preprocess import feature_groups, pattern_ids

KS = (5, 10, 20)
SUBSETS = ("all", "reached_k", "short", "prefix_seen", "prefix_unseen")


def prefix_input(sequence, k):
    if type(k) is not int or k not in (*KS, 30):
        raise ValueError("k는 5/10/20/30이어야 합니다")
    if (list(sequence.columns) != feature_groups()["sequence"] or not sequence.index.is_unique
            or sequence.empty):
        raise ValueError("고유 행 ID와 시퀀스 30개 컬럼이 필요합니다")
    values = sequence.to_numpy()
    if (not np.isfinite(values).all() or not np.equal(values, np.trunc(values)).all()
            or (np.maximum.accumulate(values == 0, axis=1) & (values != 0)).any()):
        raise ValueError("유한 정수 시퀀스와 뒤쪽 0 패딩이 필요합니다")
    result = sequence.copy()
    result.iloc[:, k:] = 0
    return result


def audit_prefix(sequence, labels, partitions, *, split, seed, k):
    if not sequence.index.equals(labels.index) or not labels.index.equals(partitions.index):
        raise ValueError("감사 대상 행 정렬 오류")
    ids = pattern_ids(prefix_input(sequence, k))
    train_ids = set(ids.loc[partitions.eq("train")])
    seen = ids.isin(train_ids)
    reached = sequence.ne(0).sum(axis=1).ge(k)
    records = []
    for part in ("train", "test", "unknown"):
        part_mask = partitions.eq(part)
        for label in ("__all__", *sorted(labels.loc[part_mask].unique())):
            mask = part_mask if label == "__all__" else part_mask & labels.eq(label)
            n = int(mask.sum())
            local_ids = set(ids.loc[mask])
            records.append(dict(split=split, seed=seed, k=k, partition=part, label=label,
                rows=n, reached_k_rows=int((mask & reached).sum()), short_rows=int((mask & ~reached).sum()),
                short_fraction=float((mask & ~reached).sum() / n) if n else np.nan,
                prefix_patterns=len(local_ids), train_prefix_patterns=len(train_ids),
                shared_prefix_patterns=len(local_ids & train_ids),
                prefix_seen_rows=int((mask & seen).sum()),
                prefix_seen_fraction=float((mask & seen).sum() / n) if n else np.nan,
                padding_policy="30 signed lengths; after k zero; original trailing zeros retained"))
    return dict(reached=reached, seen=seen), pd.DataFrame(records)


def validate_predictions(predictions, assignments, labels, partitions, known_labels):
    required = {"row_id", "partition", "actual", "predicted", "max_probability"}
    if set(predictions.columns) != required or set(assignments.columns) != {"row_id", "partition"}:
        raise ValueError("행별 저장본 컬럼 오류")
    if (assignments.row_id.tolist() != labels.index.tolist()
            or assignments.partition.tolist() != partitions.tolist()):
        raise ValueError("행별 배정이 원본과 다릅니다")
    expected = partitions.index[partitions.isin(["test", "unknown"])]
    ids = predictions.row_id
    if ids.duplicated().any() or set(ids) != set(expected):
        raise ValueError("행별 예측 ID 중복·누락·시험 외 행 포함")
    ordered = predictions.set_index("row_id").loc[expected]
    if (ordered.partition.tolist() != partitions.loc[expected].tolist()
            or ordered.actual.tolist() != labels.loc[expected].tolist()
            or not set(ordered.predicted) <= set(known_labels)
            or not np.isfinite(ordered.max_probability).all()
            or not ordered.max_probability.between(0, 1).all()):
        raise ValueError("예측 라벨·배정·확률 오류")
    return ordered.rename_axis("row_id").reset_index()


def score_subsets(predictions, *, known_labels, masks, identity):
    test = predictions.loc[predictions.partition.eq("test")].set_index("row_id")
    if (test.empty or not test.index.is_unique or not set(test.actual) <= set(known_labels)
            or not set(test.predicted) <= set(known_labels)):
        raise ValueError("시험 부분집합 평가 대상 오류")
    if any(not test.index.isin(mask.index).all() or mask.isna().any()
           or mask.dtype != bool for mask in masks.values()):
        raise ValueError("감사 마스크 누락·자료형 오류")
    reached, seen = (masks[key].loc[test.index] for key in ("reached", "seen"))
    selections = dict(all=pd.Series(True, index=test.index), reached_k=reached, short=~reached,
                      prefix_seen=seen, prefix_unseen=~seen)
    scores, classes, matrices, counts = [], [], [], {}
    for subset, mask in selections.items():
        chosen = test.loc[mask]
        n = len(chosen)
        matrix = (confusion_matrix(chosen.actual, chosen.predicted, labels=known_labels)
                  if n else np.zeros((len(known_labels), len(known_labels)), dtype=int))
        counts[subset] = matrix
        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)
        tp = matrix.diagonal()
        denominator = support + predicted_support
        f1 = np.divide(2 * tp, denominator, out=np.zeros(len(known_labels)), where=denominator != 0)
        key = dict(**identity, subset=subset)
        scores.append(dict(**key, rows=n, fraction=n / len(test),
            macro_f1=float(f1.mean()) if n else np.nan,
            accuracy=float(tp.sum() / n) if n else np.nan,
            present_classes=int((support > 0).sum()), missing_classes=int((support == 0).sum()),
            low_support_classes=int(((support > 0) & (support < 30)).sum()),
            label_policy="fixed known labels; zero_division=0; empty subset scores NA"))
        for i, label in enumerate(known_labels):
            classes.append(dict(**key, label=label, support=int(support[i]),
                predicted_support=int(predicted_support[i]),
                recall=float(tp[i] / support[i]) if support[i] else np.nan,
                f1=float(f1[i]) if n else np.nan, low_support=bool(0 < support[i] < 30)))
            for j, predicted in enumerate(known_labels):
                matrices.append(dict(**key, actual=label, predicted=predicted, count=int(matrix[i, j])))
    for left, right in (("reached_k", "short"), ("prefix_seen", "prefix_unseen")):
        if not np.array_equal(counts[left] + counts[right], counts["all"]):
            raise ValueError("부분집합 합산 행렬이 전체와 다릅니다")
    return pd.DataFrame(scores), pd.DataFrame(classes), pd.DataFrame(matrices)


def summarize_subsets(scores):
    keys = ["machine", "split", "model", "k", "subset"]
    if scores.duplicated(keys + ["seed"]).any():
        raise ValueError("부분집합 조건·시드 중복")
    records = []
    for key, group in scores.groupby(keys, sort=False, dropna=False):
        if set(group.seed) != {0, 1, 2}:
            raise ValueError("부분집합 집계는 세 시드 모두 필요합니다")
        records.append(dict(zip(keys, key)) | dict(runs=3,
            valid_runs=int(group.macro_f1.notna().sum()),
            rows_min=int(group.rows.min()), rows_max=int(group.rows.max()),
            present_classes_min=int(group.present_classes.min()),
            present_classes_max=int(group.present_classes.max()),
            macro_f1_mean=group.macro_f1.mean(), macro_f1_std=group.macro_f1.std(ddof=1),
            accuracy_mean=group.accuracy.mean(), accuracy_std=group.accuracy.std(ddof=1)))
    return pd.DataFrame(records)
