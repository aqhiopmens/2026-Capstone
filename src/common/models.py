"""두 트랙이 공유하는 고정 RF·LightGBM 기준선. 트리에는 스케일러를 쓰지 않는다."""

from sklearn.ensemble import RandomForestClassifier


def make_model(name, *, seed, workers=2):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers는 양의 정수여야 합니다")
    common = dict(n_estimators=100, random_state=seed, n_jobs=workers,
                  class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(**common)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**common, verbosity=-1, deterministic=True, force_col_wise=True)
    raise ValueError(f"지원하지 않는 모델: {name}")
