"""kNN(최근접 1개) 적응 대조(안건 10 확정안): 같은 지원 행·시험 행에서 RF와 비교한다.

거리 L1(맨해튼), 스케일링 없음(부호 있는 앞 k개 길이 그대로), 이웃 1개. 가장 가까운 공격 행까지의 거리 d_a와 가장 가까운 정상 행까지의 거리 d_b를 비교해
d_a < d_b이면 공격, 같으면 정상(보수적 동률 규칙). 거리 0(정확한 키 일치) 여부를 같이 돌려줘 '정확 조회'와 '미관측 값'을 구분한다.
"""

import numpy as np

from src.track_b.https_flows import ATTACK


def nearest_l1(train, test, chunk=1500):
    """각 시험 행에서 학습 행 집합까지의 최소 L1 거리(정수 배열). train이 비면 매우 큰 값."""
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    if len(train) == 0:
        return np.full(len(test), np.iinfo(np.int64).max // 4, dtype=np.int64)
    out = np.empty(len(test), dtype=np.int64)
    for i in range(0, len(test), chunk):
        block = test[i:i + chunk]
        out[i:i + chunk] = np.abs(block[:, None, :] - train[None, :, :]).sum(axis=2).min(axis=1)
    return out


def knn1_predict(x_train, y_train, x_test):
    """(공격 예측 bool 배열, 정확 키 일치 bool 배열)."""
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    atk = y_train == ATTACK
    da, db = nearest_l1(x_train[atk], x_test), nearest_l1(x_train[~atk], x_test)
    return da < db, np.minimum(da, db) == 0
