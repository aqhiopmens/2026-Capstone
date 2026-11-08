"""kNN(최근접 1개) 적응 대조(안건 10 확정안): 같은 지원 행·시험 행에서 RF와 비교한다.

거리 L1(맨해튼), 스케일링 없음(부호 있는 앞 k개 길이 그대로), 이웃 1개. 가장 가까운 공격 행까지의 거리 d_a와 가장 가까운 정상 행까지의 거리 d_b를 비교해
d_a < d_b이면 공격, 같으면 정상(보수적 동률 규칙). 거리 0(정확한 키 일치) 여부를 같이 돌려줘 '정확 조회'와 '미관측 값'을 구분한다.
"""

import numpy as np
import pandas as pd

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


def pair_rf_knn(rf, knn):
    """같은 (k, seed, n)의 RF·kNN 혼동 지표 짝과 (k, n) 시드 평균."""
    both = knn.merge(rf, on=["k", "seed", "n"], suffixes=("_knn", "_rf"), validate="one_to_one")
    for m in ("accuracy", "macro_f1", "fp", "fn"):
        both[f"d_{m}"] = both[f"{m}_knn"] - both[f"{m}_rf"]
    for s in ("rf", "knn"):
        both[f"recall_{s}"] = both[f"tp_{s}"] / (both[f"tp_{s}"] + both[f"fn_{s}"])
    both["d_recall"] = both.recall_knn - both.recall_rf
    cols = ["macro_f1_rf", "macro_f1_knn", "d_macro_f1", "recall_rf", "recall_knn", "d_recall", "fp_rf", "fp_knn", "d_fp"]
    return both, both.groupby(["k", "n"])[cols].mean().reset_index()


def exact_split(pred, attack_mask):
    """kNN 예측에서 시험 공격 행을 정확 키 일치(거리 0)와 미관측 값으로 나눈 recall·분모. pred: pred_attack·exact 열이 있는 DataFrame."""
    rows = []
    for (k, n), g in pred[attack_mask(pred)].groupby(["k", "n"]):
        e, x = g[g.exact == 1], g[g.exact == 0]
        rows.append(dict(k=k, n=n, attack_rows=len(g) // g.seed.nunique(), exact_rows=len(e) / g.seed.nunique(), recall_exact=e.pred_attack.mean() if len(e) else float("nan"),
                         unseen_rows=len(x) / g.seed.nunique(), recall_unseen=x.pred_attack.mean() if len(x) else float("nan")))
    return pd.DataFrame(rows)


def compare_main(args):
    import json
    from pathlib import Path

    from src.common.config import DATA_DIR
    from src.common.results import sha256_file
    from src.track_b.https_2022 import metrics_2022
    from src.track_b.https_unseen import b1_metrics

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_knn_compare_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    pdir = DATA_DIR / "predictions"
    files = dict(b1_rf=pdir / f"track_b_https_b1_{args.b1_rf}" / "b1_test_predictions.csv.gz", b1_knn=pdir / f"track_b_https_b1_{args.b1_knn}" / "b1_test_predictions.csv.gz",
                 e2022_rf=pdir / f"track_b_https2022_eval_{args.e2022_rf}" / "eval2022_predictions.csv.gz",
                 e2022_knn=pdir / f"track_b_https2022_eval_{args.e2022_knn}" / "eval2022_predictions.csv.gz")
    d = {k: pd.read_csv(v) for k, v in files.items()}
    b1r, b1k = d["b1_rf"][d["b1_rf"].kind == "seq"], d["b1_knn"]
    key = ["pcap", "flow_id", "role"]
    if not b1r[key].drop_duplicates().sort_values(key).reset_index(drop=True).equals(b1k[key].drop_duplicates().sort_values(key).reset_index(drop=True)):
        raise ValueError("#39: RF와 kNN의 시험 행이 일치하지 않습니다")
    er, ek = d["e2022_rf"][d["e2022_rf"].kind == "seq"], d["e2022_knn"]
    key2 = ["pcap", "flow_id", "actual"]
    if not er[er.in_test == 1][key2].drop_duplicates().sort_values(key2).reset_index(drop=True).equals(ek[ek.in_test == 1][key2].drop_duplicates().sort_values(key2).reset_index(drop=True)):
        raise ValueError("#40: RF와 kNN의 시험 행이 일치하지 않습니다")
    r39, s39 = pair_rf_knn(b1_metrics(b1r).drop(columns="kind"), b1_metrics(b1k).drop(columns="kind"))
    r40, s40 = pair_rf_knn(metrics_2022(er).drop(columns="kind"), metrics_2022(ek).drop(columns="kind"))
    x39 = exact_split(b1k, lambda p: p.role != "benign")
    x40 = exact_split(ek[ek.in_test == 1], lambda p: p.actual == ATTACK)
    out.mkdir(parents=True)
    for name, frame in (("paired_39_conditions", r39.assign(experiment="39_88_81")), ("paired_39_summary", s39), ("paired_40_conditions", r40.assign(experiment="40_2022")),
                        ("paired_40_summary", s40), ("exact_split_39", x39), ("exact_split_40", x40)):
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, **{k: dict(path=str(v.relative_to(DATA_DIR)), sha256=sha256_file(v)) for k, v in files.items()},
        code_sha256={"src/track_b/knn.py": sha256_file(Path(__file__))},
        note="저장 예측 짝 비교(재학습 없음), 같은 시험 행 키 확인. kNN은 L1 최근접 1개, 동률은 정상. 정확 키 일치(거리 0) 여부로 시험 공격 행을 나눠 recall 병기"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print("#39", s39.round(4).to_string(index=False))
    print("#40", s40.round(4).to_string(index=False))
    print("exact39", x39.round(4).to_string(index=False))
    print("exact40", x40.round(4).to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    for name in ("b1-rf", "b1-knn", "e2022-rf", "e2022-knn"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--run-id", required=True)
    compare_main(parser.parse_args())
