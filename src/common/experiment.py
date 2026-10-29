"""한 조건의 알려진 클래스 기준선. 파일 입출력·외부 분할·문턱 조정은 하지 않는다."""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from src.common.evaluation import evaluate_classification
from src.common.models import make_model
from src.common.splits import validate_partitions


@dataclass
class RunResult:
    model: object
    metrics: dict
    confusion: pd.DataFrame
    predictions: pd.DataFrame
    assignments: pd.DataFrame
    metadata: dict


def run_classification(x, labels, partitions, *, known_labels, model_name, seed,
                       split, groups=None, workers=2):
    """train만 fit한다. validation/unknown 예측은 알려진 시험 지표에서 제외한다."""
    if (not x.index.equals(labels.index) or not x.index.is_unique
            or not x.columns.is_unique or x.shape[1] == 0):
        raise ValueError("특징·라벨의 행 ID/순서 또는 특징 컬럼 중복·누락 오류")
    if set(x.columns) & {"Label", "source_row", "row_id", "pattern_id", "role", "partition"}:
        raise ValueError("라벨·행 ID·분할 메타데이터를 특징에 포함하지 마세요")
    if split == "pattern_group" and groups is None:
        raise ValueError("그룹 분할은 그룹 ID가 필수입니다")
    validate_partitions(labels, partitions, known_labels,
                        groups if split == "pattern_group" else None)
    active = partitions.isin(["train", "validation", "test", "unknown"])
    if not np.isfinite(x.loc[active].to_numpy(dtype=float)).all():
        raise ValueError("대상 특징에 결측·무한값이 있습니다. 자동 보완하지 않습니다")
    model = make_model(model_name, seed=seed, workers=workers)
    train = partitions.eq("train")
    start = time.monotonic()
    model.fit(x.loc[train], labels.loc[train])
    elapsed = time.monotonic() - start
    prediction_parts = []
    for part in ("validation", "test", "unknown"):
        mask = partitions.eq(part)
        if not mask.any():
            continue
        probabilities = model.predict_proba(x.loc[mask])
        prediction_parts.append(pd.DataFrame(dict(
            row_id=x.index[mask], partition=part, actual=labels.loc[mask].to_numpy(),
            predicted=model.classes_[probabilities.argmax(axis=1)],
            max_probability=probabilities.max(axis=1))))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    test = predictions.loc[predictions.partition.eq("test")]
    metrics, confusion = evaluate_classification(test.actual, test.predicted, labels=known_labels)
    metrics.update(n_features=x.shape[1], train_rows=int(train.sum()),
                   validation_rows=int(partitions.eq("validation").sum()), test_rows=len(test),
                   unknown_rows=int(partitions.eq("unknown").sum()), train_seconds=elapsed)
    metadata = dict(labels=list(known_labels), feature_columns=list(x.columns),
                    model_name=model_name, model_seed=seed, split=split,
                    model_params=model.get_params(), fit_partition="train",
                    preprocessing="none; no scaling, imputation, feature selection or row trimming",
                    unknown="predictions only; excluded from known-class scores; no rejection threshold fitted")
    assignments = pd.DataFrame(dict(row_id=x.index, partition=partitions.to_numpy()))
    return RunResult(model, metrics, confusion, predictions, assignments, metadata)
