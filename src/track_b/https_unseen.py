"""처음 보는 공격(.88.81)을 저장 예측 점수로 진단한다 (단계 0, 학습 없음).

저장 예측(max_probability, predicted)에서 P(attack)을 복원해 .88.81 공격 대 04-16 정상을 AUC·점수 분포·동률 비율·
시험 점수 상한 recall@정상 오탐으로 본다. 상한은 시험 점수로 문턱을 골랐을 때의 값이라 **진단 전용**이며 탐지 성능이 아니다.
판정 기준(사전 고정, 안건 5): AUC>=0.9 이고 상한(5%)>=0.5 -> 문턱 문제(표현은 충분), AUC<=0.6 -> 표현 문제, 그 사이 보류.

단계 A(`suspect`): 정상만 학습한 IsolationForest의 이상 점수로 두 칸(의심/정상) 판정. 04-17 검증 정상에서 문턱을 정하고
04-16 시험에 적용하는 **오프라인 전이 평가**(검증이 시험보다 미래)이며 점수는 악성 확률이 아니다. 상위 0.5%는 '고위험 우선 검토' 순위일 뿐 공격 판정이 아니다.

Run: uv run python -m src.track_b.https_unseen diag --predictions <csv> [--predictions <csv> ...] --run-id <ID>
     uv run python -m src.track_b.https_unseen suspect --flows-run-id 2026-11-02_v1 --xml-flows-run-id 2026-10-31_v3 --run-id <ID>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from src.common.results import sha256_file
from src.track_b.early_eval import prefix_features
from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.https_robust import transform
from src.track_b.pcap_flows import K_MAX

NEW_VICTIM = "128.199.88.81"
SPLIT = "time_0411to0416_all"
COLUMNS = ["split", "feature_group", "model", "seed", "actual", "predicted", "max_probability", "server_ip",
           "payload_packets", "status_key_payload"]
GROUP_KEYS = ["feature_group", "model"]
FPRS = (0.01, 0.05)


def attack_score(predicted, max_probability):
    """이진 분류의 predicted와 max_probability에서 P(attack)을 복원한다(predicted가 공격이면 p, 아니면 1-p)."""
    p = np.asarray(max_probability, dtype=float)
    return np.where(np.asarray(predicted) == ATTACK, p, 1.0 - p)


def oracle_recall(pos, neg, fpr):
    """정상의 상위 fpr 비율만 오탐으로 허용하는 가장 낮은 문턱(score > t)에서의 공격 recall. 시험 점수를 쓰는 진단값."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    budget = int(np.floor(fpr * len(neg)))
    ordered = np.sort(neg)[::-1]
    threshold = ordered[budget] if budget < len(neg) else -np.inf
    return float((pos > threshold).mean())


def diagnose(frame):
    """(feature_group, model)별로 3시드를 합쳐 .88.81 공격 대 정상 진단 행을 낸다. 정상 집합별(전체, CSV 확인, 미매칭)로 따로."""
    rows = []
    for (group, model), part in frame.groupby(GROUP_KEYS):
        pos = part[(part.actual == ATTACK) & (part.server_ip == NEW_VICTIM)]
        for name, mask in (("benign_all", part.actual == BENIGN),
                           ("benign_csv_confirmed", (part.actual == BENIGN) & part.status_key_payload.isin(["single_agree", "multi_agree"])),
                           ("benign_csv_unmatched", (part.actual == BENIGN) & (part.status_key_payload == "unmatched"))):
            neg = part[mask]
            if pos.empty or neg.empty:
                continue
            sp, sn = attack_score(pos.predicted, pos.max_probability), attack_score(neg.predicted, neg.max_probability)
            labels = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
            auc = float(roc_auc_score(labels, np.r_[sp, sn]))
            row = dict(feature_group=group, model=model, benign_set=name, attack_rows=len(sp), benign_rows=len(sn), auc=round(auc, 6),
                       attack_score_median=float(np.median(sp)), attack_score_p90=float(np.quantile(sp, 0.9)),
                       benign_score_median=float(np.median(sn)), benign_score_p99=float(np.quantile(sn, 0.99)),
                       attack_share_score_le_0_01=float((sp <= 0.01).mean()), benign_share_score_le_0_01=float((sn <= 0.01).mean()),
                       attack_tied_with_benign_max=float((sp <= sn.max()).mean() if len(sn) else np.nan))
            for fpr in FPRS:
                row[f"oracle_recall_fpr{int(fpr * 100)}"] = round(oracle_recall(sp, sn, fpr), 6)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["verdict"] = np.where((out.auc >= 0.9) & (out.oracle_recall_fpr5 >= 0.5), "threshold_problem",
                              np.where(out.auc <= 0.6, "representation_problem", "inconclusive"))
    return out


def load_split(path, split=SPLIT):
    parts = [chunk[chunk.split == split] for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=500_000,
                                                                keep_default_na=False, float_precision="round_trip")]
    return pd.concat(parts, ignore_index=True)


SEQ_COLUMNS = [f"seq_{i}" for i in range(K_MAX)]
TRAIN_DAYS, VALID_DAY, TEST_DAY = ("2021-04-11", "2021-04-12"), "2021-04-17", "2021-04-16"
SUSPECT_BUDGET, PRIORITY_BUDGET = (2, 100), (5, 1000)  # 검증 정상 대비 2%, 0.5% (정수 연산)
SEEDS = (0, 1, 2)
INPUTS = ("seq", "logb")
SUSPECT_KS = (5, 10)


def budget_threshold(scores, numerator, denominator):
    """높을수록 의심인 점수에서, strict `score > tau`로 표시되는 행이 정수 예산 n×num//den 이하가 되는 가장 작은 tau.

    동률 블록이 예산을 넘치면 그 블록은 통째로 표시하지 않는다(tau가 블록 값이 된다). 반환: tau, 예산, 실제 표시 수, 경계 동률 행 수.
    A의 `select_threshold`는 percent {1,5,10}·[0,1] 확률·낮을수록 거절만 받아 못 쓰므로 같은 규칙(strict, 정수 예산, 동률 블록 제외)으로 따로 둔다.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("검증 점수가 비었거나 유한하지 않습니다")
    allowed = scores.size * numerator // denominator
    candidates = np.r_[-np.inf, np.unique(scores)]
    tau = next(c for c in candidates if (scores > c).sum() <= allowed)
    return dict(tau=float(tau), allowed=int(allowed), flagged=int((scores > tau).sum()),
                boundary_tie_rows=int((scores == tau).sum()))


def anomaly_scores(model, x):
    return -model.score_samples(x)  # 클수록 정상에서 먼 점


def suspect_run(flows, k, kind, seed):
    """한 (입력, k, 시드) 조건. flows는 payload>0만. 학습=04-11+04-12 정상, 검증=04-17 정상, 시험=04-16 정상+.88.81 공격(+.242.104 공격은 따로)."""
    make = (lambda f: prefix_features(f, k, "seq")) if kind == "seq" else (lambda f: transform(f, k, "logb"))
    benign = flows.label == BENIGN
    train, valid = flows[benign & flows.day.isin(TRAIN_DAYS)], flows[benign & (flows.day == VALID_DAY)]
    test = flows[flows.day == TEST_DAY]
    if set(train.day) & {VALID_DAY, TEST_DAY} or train.empty or valid.empty:
        raise ValueError("학습·검증·시험 날짜 분리 오류")
    model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=2).fit(make(train))
    val_scores = anomaly_scores(model, make(valid))
    cut = budget_threshold(val_scores, *SUSPECT_BUDGET)
    top = budget_threshold(val_scores, *PRIORITY_BUDGET)
    groups = {"test_benign": test[test.label == BENIGN],
              "test_88_81_attack": test[(test.label == ATTACK) & (test.server_ip == NEW_VICTIM)],
              "test_242_104_attack": test[(test.label == ATTACK) & (test.server_ip != NEW_VICTIM)]}
    row = dict(kind=kind, k=k, seed=seed, valid_rows=len(valid), valid_flagged=cut["flagged"], valid_allowed=cut["allowed"],
               boundary_tie_rows=cut["boundary_tie_rows"], tau=cut["tau"], top_tau=top["tau"], top_flagged=top["flagged"],
               top_boundary_tie_rows=top["boundary_tie_rows"], valid_88_81_benign_rows=int((valid.server_ip == NEW_VICTIM).sum()))
    for name, g in groups.items():
        s = anomaly_scores(model, make(g)) if len(g) else np.array([])
        row[f"{name}_rows"] = len(g)
        row[f"{name}_flag_rate"] = float((s > cut["tau"]).mean()) if len(g) else np.nan
        row[f"{name}_priority_rate"] = float((s > top["tau"]).mean()) if len(g) else np.nan
        if name == "test_benign":
            short = (g.payload_packets < k).to_numpy()
            row["test_benign_short_flag_rate"] = float((s[short] > cut["tau"]).mean()) if short.any() else np.nan
            row["test_benign_short_rows"] = int(short.sum())
    row["fpr_valid_minus_test"] = row["valid_flagged"] / row["valid_rows"] - row["test_benign_flag_rate"]
    return row


def load_flows(flows_run_id, xml_run_id, data_dir):
    cols = ["pcap", "day", "flow_id", "label", "client_ip", "server_ip", "payload_packets", "midstream"] + SEQ_COLUMNS
    a = pd.read_csv(data_dir / "processed" / f"track_b_https_{flows_run_id}" / "flows.csv", usecols=cols, keep_default_na=False)
    b = pd.read_csv(data_dir / "processed" / f"track_b_{xml_run_id}" / "flows.csv", usecols=cols, keep_default_na=False)
    b = b.assign(label=b.label.replace({"bruteforce_xml": ATTACK}))
    flows = pd.concat([a, b], ignore_index=True)
    return flows[flows.payload_packets > 0].reset_index(drop=True)


def suspect_main(args):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_suspect_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows = load_flows(args.flows_run_id, args.xml_flows_run_id, DATA_DIR)
    result = pd.DataFrame([suspect_run(flows, k, kind, seed) for kind in INPUTS for k in SUSPECT_KS for seed in SEEDS])
    out.mkdir(parents=True)
    result.to_csv(out / "suspect.csv", index=False)
    summary = result.groupby(["kind", "k"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, xml_flows_run_id=args.xml_flows_run_id,
        train_days=list(TRAIN_DAYS), valid_day=VALID_DAY, test_day=TEST_DAY, suspect_budget_percent=SUSPECT_BUDGET[0],
        priority_budget_permille=PRIORITY_BUDGET[0], inputs=list(INPUTS), ks=list(SUSPECT_KS), seeds=list(SEEDS),
        flows_sha256=sha256_file(DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"),
        xml_flows_sha256=sha256_file(DATA_DIR / "processed" / f"track_b_{args.xml_flows_run_id}" / "flows.csv"),
        code_sha256={"src/track_b/https_unseen.py": sha256_file(Path(__file__))},
        note="오프라인 전이 평가(04-17 검증 문턱을 04-16에 적용). 이상 점수는 악성 확률이 아니며 상위 0.5%는 고위험 우선 검토 순위일 뿐 "
             "공격 판정이 아니다. 탐색적 결과"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summary[["kind", "k", "valid_rows", "valid_flagged", "boundary_tie_rows", "test_benign_flag_rate", "test_88_81_attack_flag_rate",
                   "test_88_81_attack_priority_rate", "test_242_104_attack_flag_rate", "fpr_valid_minus_test"]].round(4).to_string(index=False))
    print("DONE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("suspect")
    r.add_argument("--flows-run-id", required=True)
    r.add_argument("--xml-flows-run-id", required=True)
    r.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    d = sub.add_parser("diag")
    d.add_argument("--predictions", action="append", required=True)
    d.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    if args.cmd == "suspect":
        return suspect_main(args)
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_unseen_diag_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    frames = [load_split(p) for p in args.predictions]
    result = diagnose(pd.concat(frames, ignore_index=True))
    out.mkdir(parents=True)
    result.to_csv(out / "diagnosis.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, split=SPLIT, new_victim=NEW_VICTIM, fprs=list(FPRS),
        predictions=[dict(path=Path(p).name, parent=Path(p).parent.name, sha256=sha256_file(Path(p))) for p in args.predictions],
        rule="AUC>=0.9 and oracle_recall_fpr5>=0.5 -> threshold_problem; AUC<=0.6 -> representation_problem; else inconclusive",
        note="저장 예측만 사용(학습 없음). oracle recall은 시험 점수로 문턱을 고른 상한이며 진단 전용, 탐지 성능이 아니다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(result[["feature_group", "model", "benign_set", "auc", "oracle_recall_fpr1", "oracle_recall_fpr5", "attack_share_score_le_0_01", "verdict"]]
          .to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
