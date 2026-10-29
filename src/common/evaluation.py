"""고정 클래스 전체로 macro-F1·정확도·혼동 행렬을 계산한다."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_classification(actual, predicted, *, labels):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or predicted.ndim != 1 or not len(actual) or len(actual) != len(predicted):
        raise ValueError("실제·예측 라벨은 같은 길이의 비어 있지 않은 1차원이어야 합니다")
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError("고정 클래스 목록 오류")
    if set(actual) != set(labels) or not set(predicted) <= set(labels):
        raise ValueError("평가 대상은 고정 클래스 전체이며 미확인 라벨을 포함할 수 없습니다")
    matrix = confusion_matrix(actual, predicted, labels=labels)
    confusion = pd.DataFrame([
        dict(actual=a, predicted=p, count=int(matrix[i, j]))
        for i, a in enumerate(labels) for j, p in enumerate(labels)
    ])
    metrics = dict(macro_f1=float(f1_score(actual, predicted, labels=labels,
                                        average="macro", zero_division=0)),
                   accuracy=float(accuracy_score(actual, predicted)))
    return metrics, confusion
