"""2022 HTTPS 평가(안건 6 확정안): 2021 04-11 학습 -> 2022 04-10 시험, 2022 공격 첫 N개를 oracle 지원으로 추가.

고정(성능 확인 전, 안건 6): 학습 기준선 = 2021 04-11 HTTPS 플로우 전체(payload>0, 정상+.242.104 공격), 04-12 XML·04-17은 쓰지 않는다.
정상 지원 추가 없음. 문턱·IsolationForest 없이 RF(기본 predict)만 쓰고 N마다 같은 고정 설정으로 새로 학습한다.
시험 = 2022 N=50 지원 묶음 종료+10분 뒤에 시작한 모든 플로우(모든 N 공통). N=0의 2022 전체 평가는 별도 '전이 기준'이다.
오탐은 실제 시험 정상 분모와 FP 건수를 같이 적고, 정확 이항 구간(Clopper-Pearson)은 독립 시행 가정의 참고값이다.
같은 캡처·연결 묶음의 상관 때문에 환경 간 오탐 불확실성은 추정하기 어렵다. 분모 0이면 계산 불가(NaN).

Run: uv run python -m src.track_b.https_2022 --flows-run-id 2026-11-02_v1 --flows-2022-run-id 2026-11-05_v2 --run-id <ID>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.common.results import sha256_file
from src.track_b.early_eval import prefix_features
from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.https_robust import transform
from src.track_b.https_unseen import (B1_KS, B1_NS, B1_TRAIN_DAY, BUNDLE_GAP_S, INPUTS, SEEDS, SEQ_COLUMNS, WINDOW_AFTER_S,
                                      prefix_keys, run_environment, support_window)

VICTIMS = ("184.0.48.150", "184.0.48.169")
COLS = ["pcap", "day", "flow_id", "label", "client_ip", "server_ip", "payload_packets", "start_ts", "duration"] + SEQ_COLUMNS


def exact_ci(k, n, alpha=0.05):
    """Clopper-Pearson 정확 구간(독립 시행 가정의 참고값). n=0이면 (nan, nan)."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def load_payload(path):
    f = pd.read_csv(path, usecols=COLS, keep_default_na=False)
    return f[f.payload_packets > 0].reset_index(drop=True)


def split_2022(flows, n_max=50):
    """지원 후보(공격 시간순)와 공통 시험 행. 시험 = 창 끝(N=50 묶음 종료+10분) 뒤에 시작한 모든 플로우. 창 시작 전·창 안 행은 제외."""
    attacks = flows[flows.label == ATTACK]
    w0, w1, ordered, bundle = support_window(attacks, n_max)
    test_mask = (flows.start_ts > w1).to_numpy()
    end = flows.start_ts + flows.duration
    return dict(window=(w0, w1), support=ordered, bundle=bundle, test_mask=test_mask,
                boundary_cross=(int(((flows.start_ts < w0) & (end >= w0)).sum()), int(((flows.start_ts <= w1) & (end > w1)).sum())),
                before_window_rows=int((flows.start_ts < w0).sum()),
                inside_attack_rows=int(((flows.start_ts >= w0) & (flows.start_ts <= w1) & (flows.label == ATTACK)).sum()))


def condition_rows(flows, split, pred_by_n, make_x, k, kind, seed, ns):
    """모델별 예측(전체 2022)에서 시험/전이 지표 행과 행별 예측을 만든다. pred_by_n[n] = 전체 플로우에 대한 pred_attack 배열."""
    rows, details = [], []
    attack = (flows.label == ATTACK).to_numpy()
    test = split["test_mask"]
    short = (flows.payload_packets < k).to_numpy()
    for n in ns:
        pred = pred_by_n[n].astype(bool)
        support_keys = set(prefix_keys(make_x(split["support"].head(n)))) if n else set()
        unseen = np.array([key not in support_keys for key in prefix_keys(make_x(flows))])
        row = dict(kind=kind, k=k, seed=seed, n=n)
        for name, mask in (("test", test), ("full", np.ones(len(flows), dtype=bool))):
            a, b = mask & attack, mask & ~attack
            fp = int(pred[b].sum())
            lo, hi = exact_ci(fp, int(b.sum()))
            row.update({f"{name}_attack_rows": int(a.sum()), f"{name}_attack_tp": int(pred[a].sum()),
                        f"{name}_recall": float(pred[a].mean()) if a.any() else np.nan,
                        f"{name}_benign_rows": int(b.sum()), f"{name}_benign_fp": fp,
                        f"{name}_benign_fpr": fp / int(b.sum()) if b.any() else np.nan,
                        f"{name}_benign_fpr_ci_lo": lo, f"{name}_benign_fpr_ci_hi": hi})
        u = test & attack & unseen
        row.update(test_unseen_prefix_rows=int(u.sum()), test_recall_unseen_prefix=float(pred[u].mean()) if u.any() else np.nan,
                   test_benign_short_rows=int((test & ~attack & short).sum()),
                   test_benign_short_fp=int(pred[test & ~attack & short].sum()))
        for v in VICTIMS:
            m = test & attack & (flows.server_ip == v).to_numpy()
            row.update({f"test_attack_rows_{v}": int(m.sum()), f"test_recall_{v}": float(pred[m].mean()) if m.any() else np.nan})
        rows.append(row)
        details.append(pd.DataFrame(dict(pcap=flows.pcap.to_numpy(), flow_id=flows.flow_id.to_numpy(), actual=flows.label.to_numpy(),
                                         in_test=test.astype(np.int8), unseen_prefix=unseen.astype(np.int8),
                                         pred_attack=pred.astype(np.int8))).assign(kind=kind, k=k, seed=seed, n=n))
    return rows, pd.concat(details, ignore_index=True)


def run_condition(base, flows, split, k, kind, seed, ns=B1_NS):
    from src.common.models import make_model

    make_x = (lambda f: prefix_features(f, k, "seq")) if kind == "seq" else (lambda f: transform(f, k, "logb"))
    x_all = make_x(flows)
    pred_by_n = {}
    for n in ns:
        train = pd.concat([base, split["support"].head(n)], ignore_index=True)
        model = make_model("rf", seed=seed, workers=2).fit(make_x(train), train.label)
        pred_by_n[n] = model.predict(x_all) == ATTACK
    rows, details = condition_rows(flows, split, pred_by_n, make_x, k, kind, seed, ns)
    for r in rows:
        r["train_rows"] = len(base) + r["n"]
    return rows, details


def metrics_2022(predictions):
    """저장 예측에서 공통 시험 행(in_test=1) 기준 조건(kind,k,seed,n)별 TP/TN/FP/FN·accuracy·macro-F1을 계산한다(재학습 없음). 양성=공격."""
    t = predictions[predictions.in_test == 1]
    rows = []
    for (kind, k, seed, n), g in t.groupby(["kind", "k", "seed", "n"]):
        truth, pred = (g.actual == ATTACK).to_numpy(), g.pred_attack.to_numpy().astype(bool)
        tp, tn, fp, fn = int((truth & pred).sum()), int((~truth & ~pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
        f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
        rows.append(dict(kind=kind, k=k, seed=seed, n=n, rows=len(g), tp=tp, tn=tn, fp=fp, fn=fn,
                         accuracy=(tp + tn) / len(g), macro_f1=(f1(tp, fp, fn) + f1(tn, fn, fp)) / 2))
    return pd.DataFrame(rows)


def metrics_main(run_id):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    pred_path = DATA_DIR / "predictions" / f"track_b_https2022_eval_{run_id}" / "eval2022_predictions.csv.gz"
    out = root / "outputs/results" / f"track_b_https2022_eval_{run_id}"
    if not out.exists() or (out / "metrics_per_condition.csv").exists():
        raise FileExistsError("결과 폴더가 없거나 이미 지표 파일이 있습니다")
    metrics = metrics_2022(pd.read_csv(pred_path))
    metrics.to_csv(out / "metrics_per_condition.csv", index=False)
    summary = metrics.groupby(["kind", "k", "n"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    summary.to_csv(out / "metrics_summary.csv", index=False)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    meta["posthoc_metrics"] = dict(file="metrics_per_condition.csv", computed_from=str(pred_path.relative_to(DATA_DIR)),
                                   predictions_sha256=sha256_file(pred_path), code_sha256=sha256_file(Path(__file__)),
                                   note="재학습 없이 저장 예측에서 계산(공통 시험 행, 양성=공격)")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary[["kind", "k", "n", "tp", "tn", "fp", "fn", "accuracy", "macro_f1"]].round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-of", help="기존 run-id의 저장 예측에서 혼동 건수·accuracy·macro-F1만 계산(재학습 없음)")
    parser.add_argument("--flows-run-id")
    parser.add_argument("--flows-2022-run-id")
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    if args.metrics_of:
        return metrics_main(args.metrics_of)
    if not (args.flows_run_id and args.flows_2022_run_id):
        parser.error("--flows-run-id와 --flows-2022-run-id가 필요합니다")
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https2022_eval_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_https2022_eval_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    p21 = DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"
    p22 = DATA_DIR / "processed" / f"track_b_https2022_{args.flows_2022_run_id}" / "flows.csv"
    base = load_payload(p21)
    base = base[base.day == B1_TRAIN_DAY].reset_index(drop=True)
    flows = load_payload(p22).sort_values(["start_ts", "flow_id"]).reset_index(drop=True)
    split = split_2022(flows)
    runs = [run_condition(base, flows, split, k, kind, seed) for kind in INPUTS for k in B1_KS for seed in SEEDS]
    result = pd.DataFrame([r for rows, _ in runs for r in rows])
    data_out.mkdir(parents=True)
    pd.concat([d for _, d in runs], ignore_index=True).to_csv(data_out / "eval2022_predictions.csv.gz", index=False)
    split["support"][["pcap", "flow_id", "start_ts", "duration", "server_ip"]].assign(bundle=split["bundle"]).to_csv(
        data_out / "support_order.csv", index=False, float_format="%.17g")
    test_ids = flows[split["test_mask"]][["pcap", "flow_id"]].sort_values(["pcap", "flow_id"])
    test_ids.to_csv(data_out / "common_test_rows.csv", index=False)
    out.mkdir(parents=True)
    result.to_csv(out / "eval2022.csv", index=False)
    summary = result.groupby(["kind", "k", "n"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    lf = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    sup = split["support"]
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, flows_2022_run_id=args.flows_2022_run_id, ns=list(B1_NS), ks=list(B1_KS),
        seeds=list(SEEDS), inputs=list(INPUTS), train_day=B1_TRAIN_DAY, train_rows=len(base), train_label_counts=base.label.value_counts().to_dict(),
        flows_2022_rows_payload_gt0=len(flows), flows_2022_label_counts=flows.label.value_counts().to_dict(),
        bundle_gap_s=BUNDLE_GAP_S, window_after_s=WINDOW_AFTER_S, window=list(split["window"]),
        support_bundle_counts=pd.Series(split["bundle"][:max(B1_NS)]).nunique(), first50_victims=sup.head(50).server_ip.value_counts().to_dict(),
        common_test_rows=len(test_ids), common_test_victim_counts=flows[split["test_mask"] & (flows.label == ATTACK).to_numpy()].server_ip.value_counts().to_dict(),
        common_test_benign_rows=int((split["test_mask"] & (flows.label == BENIGN).to_numpy()).sum()),
        before_window_rows_excluded=split["before_window_rows"], inside_window_attack_rows_excluded=split["inside_attack_rows"],
        boundary_cross=dict(start_before_w0_and_end_ge_w0=split["boundary_cross"][0], start_le_w1_and_end_gt_w1=split["boundary_cross"][1]),
        common_test_rows_sha256=sha256_file(data_out / "common_test_rows.csv"), common_test_rows_sha256_lf=lf(data_out / "common_test_rows.csv"),
        predictions_files={f.name: sha256_file(f) for f in sorted(data_out.iterdir())}, environment=run_environment(root),
        flows_sha256=sha256_file(p21), flows_2022_sha256=sha256_file(p22),
        code_sha256={"src/track_b/https_2022.py": sha256_file(Path(__file__))},
        note="안건 6 확정안: 문턱·IsolationForest 없음, RF 기본 predict, 정상 지원 추가 없음. oracle 지원 N개 라벨(같은 캡처 안 효과, 일반화 근거 아님). "
             "FP 구간은 독립 시행 가정의 참고값이며 환경 간 오탐 불확실성은 추정하지 못한다. 공격자 1명·잠정 라벨 기반 탐색적 결과"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    cols = ["kind", "k", "n", "test_attack_rows", "test_recall", "test_unseen_prefix_rows", "test_recall_unseen_prefix",
            "test_benign_rows", "test_benign_fp", "full_recall", "full_benign_fp"]
    print(summary[cols].round(4).to_string(index=False))
    print("test rows", len(test_ids), "window", split["window"], "DONE", out, flush=True)


if __name__ == "__main__":
    main()
