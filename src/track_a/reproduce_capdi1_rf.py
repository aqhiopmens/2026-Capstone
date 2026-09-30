"""캡디1 Malicious_TLS RF 결과 재현 (notebooks/models/baseline_rf.ipynb의 run_rf와 같은 설정).

확인용이며 [실험 1] 기준선이 아니다. 캡디1 정제본·특징 선택본·시드 1개 조건이라 plan.md의 실험 조건
(재전처리, 18종 중 13종, 특징군 4조합 × 시드 3회, 혼동 행렬, Colab 실행)과 다르다.
[실험 1] 기준선은 src/common/ 공통 코드로 따로 만든다 (docs/decisions.md 2026-09-30).

실행: uv run python -m src.track_a.reproduce_capdi1_rf
성공 기준: 정확도가 캡디1 값 ±0.1%p 안, 그리고 68개 >= 117개 방향 유지. 학습 시간은 기계가 달라 비교하지 않는다.
"""
import csv
from datetime import date
from pathlib import Path

import lightgbm
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common.config import DATA_DIR

CAPDI1_ACC = {117: 0.9903, 68: 0.9911}
TOLERANCE = 0.001  # ±0.1%p


def run_rf(df, feature_cols):
    y = LabelEncoder().fit_transform(df["Label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")


def main():
    processed = DATA_DIR / "processed"
    full = pd.read_csv(processed / "malicious_tls_cleaned.csv", low_memory=False)
    feat = pd.read_csv(processed / "malicious_tls_features_final.csv", low_memory=False)
    # 캡디1과 같은 열 순서를 쓴다 (RF는 열 순서가 바뀌면 결과가 달라짐)
    selected = (processed / "malicious_tls_selected_features.txt").read_text(encoding="utf-8").split()

    runs = {
        117: run_rf(full, [c for c in full.columns if c != "Label"]),
        68: run_rf(feat, selected),
    }

    ok = True
    rows = []
    for n, (acc, f1) in runs.items():
        within = abs(acc - CAPDI1_ACC[n]) <= TOLERANCE
        ok &= within
        print(f"특징 {n}개: 정확도 {acc:.4f} (캡디1 {CAPDI1_ACC[n]:.4f}, {'통과' if within else '실패'}), macro-F1 {f1:.4f}")
        rows.append({
            "date": date.today().isoformat(), "track": "A", "experiment": "capdi1_reproduce",
            "split": "random", "feature_group": f"capdi1_{n}", "model": "rf", "seed": 42,
            "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "machine": "kyle-win",
            "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "torch": "",
            "note": "캡디1 정제본·특징 선택본 그대로 사용",
        })
    direction = runs[68][0] >= runs[117][0]
    ok &= direction
    print(f"방향(68개 >= 117개): {'유지' if direction else '뒤집힘'}")

    out = Path("outputs/results/capdi1_reproduce.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("재현 성공" if ok else "재현 실패")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
