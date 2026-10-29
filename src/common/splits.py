"""학습 내부 검증 분할. 시간 분할은 호출자가 명시적으로 배정한다."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


PARTITIONS = {"train", "validation", "test", "unknown", "excluded", "invalid"}


def validate_partitions(labels, partitions, known_labels, groups=None):
    if not labels.index.is_unique or not labels.index.equals(partitions.index):
        raise ValueError("라벨과 분할의 고유 행 ID 및 순서가 일치해야 합니다")
    if labels.isna().any() or not partitions.isin(PARTITIONS).all():
        raise ValueError("결측 라벨 또는 잘못된 분할")
    known = set(known_labels)
    if len(known) < 2 or len(known) != len(known_labels):
        raise ValueError("고정 클래스 목록은 중복 없는 2종 이상이어야 합니다")
    for part in ("train", "test", "validation"):
        selected = labels[partitions.eq(part)]
        if part == "validation" and selected.empty:
            continue
        if set(selected) != known:
            raise ValueError(f"{part}에 고정 클래스 전체가 필요합니다")
    if set(labels[partitions.eq("unknown")]) & known:
        raise ValueError("unknown에 알려진 클래스가 있습니다")
    if groups is not None:
        if not groups.index.equals(labels.index) or groups.isna().any():
            raise ValueError("그룹의 행 ID·순서 또는 결측값 오류")
        seen = set()
        for part in ("train", "validation", "test", "unknown"):
            current = set(groups[partitions.eq(part)])
            if seen & current:
                raise ValueError("학습·검증·시험·보류 사이에 그룹이 겹칩니다")
            seen.update(current)


def split_validation(labels, partitions, *, split, seed, groups=None, minimum=30):
    """기존 train만 80/20으로 나누며 외부 test/unknown 배정은 보존한다.

    그룹 폴드는 클래스별 최소 행 수를 만족하는 후보 중 검증 비율 20%에
    가장 가까운 것(동률은 폴드 번호순)을 고른다. 모델 점수를 보지 않는다.
    """
    known_labels = sorted(labels[partitions.eq("train")].unique())
    if split not in {"random", "pattern_group"}:
        raise ValueError("random/pattern_group만 지원; 시간 검증은 직접 배정하세요")
    if split == "pattern_group" and groups is None:
        raise ValueError("패턴 그룹 분할에는 전역 그룹 ID가 필요합니다")
    if partitions.eq("validation").any() or minimum < 1:
        raise ValueError("검증셋 중복 분리 또는 잘못된 최소 행 수")
    checked_groups = groups if split == "pattern_group" else None
    validate_partitions(labels, partitions, known_labels, checked_groups)
    positions = np.flatnonzero(partitions.eq("train"))
    y = labels.iloc[positions].to_numpy()

    def valid(train, val):
        return all(min(int((y[train] == c).sum()), int((y[val] == c).sum())) >= minimum
                   for c in known_labels)

    if split == "random":
        train, val = train_test_split(np.arange(len(y)), test_size=.2,
                                     stratify=y, random_state=seed)
        if not valid(train, val):
            raise ValueError("학습·검증의 클래스별 최소 행 수 부족")
    else:
        ids, _ = pd.factorize(groups.iloc[positions], sort=True)
        shuffled = np.random.default_rng(seed).permutation(ids.max() + 1)[ids]
        candidates = []
        for fold, (train, val) in enumerate(StratifiedGroupKFold(5, shuffle=False).split(
                np.zeros(len(y)), y, shuffled)):
            if valid(train, val):
                candidates.append((abs(len(val) / len(y) - .2), fold, val))
        if not candidates:
            raise ValueError("클래스별 최소 행 수를 만족하는 검증 그룹 폴드가 없습니다")
        _, _, val = min(candidates, key=lambda candidate: candidate[:2])
    result = partitions.copy()
    result.iloc[positions[val]] = "validation"
    validate_partitions(labels, result, known_labels, checked_groups)
    return result
