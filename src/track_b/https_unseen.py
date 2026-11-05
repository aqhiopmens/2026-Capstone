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
import hashlib
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
    details = [pd.DataFrame(dict(pcap=valid.pcap.to_numpy(), flow_id=valid.flow_id.to_numpy(), role="valid", score=val_scores))]
    for name, g in groups.items():
        s = anomaly_scores(model, make(g)) if len(g) else np.array([])
        details.append(pd.DataFrame(dict(pcap=g.pcap.to_numpy(), flow_id=g.flow_id.to_numpy(), role=name, score=s)))
        row[f"{name}_rows"] = len(g)
        row[f"{name}_flag_rate"] = float((s > cut["tau"]).mean()) if len(g) else np.nan
        row[f"{name}_priority_rate"] = float((s > top["tau"]).mean()) if len(g) else np.nan
        if name == "test_benign":
            short = (g.payload_packets < k).to_numpy()
            row["test_benign_short_flag_rate"] = float((s[short] > cut["tau"]).mean()) if short.any() else np.nan
            row["test_benign_short_rows"] = int(short.sum())
    row["fpr_valid_minus_test"] = row["valid_flagged"] / row["valid_rows"] - row["test_benign_flag_rate"]
    detail = pd.concat(details, ignore_index=True)
    detail = detail.assign(kind=kind, k=k, seed=seed, suspect=detail.score > cut["tau"], priority=detail.score > top["tau"],
                           tau=cut["tau"], top_tau=top["tau"])
    return row, detail


def load_flows(flows_run_id, xml_run_id, data_dir):
    cols = ["pcap", "day", "flow_id", "label", "client_ip", "server_ip", "payload_packets", "midstream",
            "start_ts", "duration"] + SEQ_COLUMNS
    a = pd.read_csv(data_dir / "processed" / f"track_b_https_{flows_run_id}" / "flows.csv", usecols=cols, keep_default_na=False)
    b = pd.read_csv(data_dir / "processed" / f"track_b_{xml_run_id}" / "flows.csv", usecols=cols, keep_default_na=False)
    b = b.assign(label=b.label.replace({"bruteforce_xml": ATTACK}))
    flows = pd.concat([a, b], ignore_index=True)
    return flows[flows.payload_packets > 0].reset_index(drop=True)


ENV_DEPENDENCIES = ("src/track_b/https_unseen.py", "src/track_b/early_eval.py", "src/track_b/https_robust.py", "src/track_b/https_eval.py",
                    "src/track_b/https_flows.py", "src/track_b/pcap_flows.py", "src/common/models.py", "src/common/experiment.py")


def run_environment(root):
    """실행 당시의 머신·버전·코드 SHA·모델 전체 파라미터. 과거 실행에 소급하지 않도록 실행 시점에 기록한다."""
    import platform
    import subprocess
    from importlib.metadata import version

    from src.common.config import MACHINE
    from src.common.models import make_model

    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    return dict(machine=MACHINE, python=platform.python_version(), platform=platform.platform(),
                packages={n: version(n) for n in ("scikit-learn", "numpy", "pandas", "lightgbm")},
                git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
                dependency_sha256={d: sha256_file(root / d) for d in ENV_DEPENDENCIES},
                rf_params=make_model("rf", seed=0).get_params(),
                isolation_forest_params=IsolationForest(n_estimators=100, random_state=0, n_jobs=2).get_params())


B1_NS = (0, 5, 10, 20, 50)
B1_KS = (5, 10)
BUNDLE_GAP_S, WINDOW_AFTER_S = 5.0, 600.0
B1_TRAIN_DAY = "2021-04-11"


def support_window(attacks, n_max=50, gap=BUNDLE_GAP_S, after=WINDOW_AFTER_S):
    """첫 n_max개 공격이 속한 연결 묶음(시작 시각 간격 > gap초면 새 묶음) 전체와 그 종료 후 after초까지의 시간 창.

    반환: (창 시작, 창 끝, 지원 후보 순서(start_ts 순 행 위치), 묶음 id 배열). 종료 시각 = start_ts + duration.
    """
    a = attacks.sort_values(["start_ts", "flow_id"]).reset_index(drop=True)
    bundle = (a.start_ts.diff().fillna(0) > gap).cumsum().to_numpy()
    first_bundles = np.unique(bundle[:n_max])
    in_bundle = np.isin(bundle, first_bundles)
    start = float(a.start_ts[in_bundle].min())
    end = float((a.start_ts + a.duration)[in_bundle].max()) + after
    return start, end, a, bundle


def b1_split(flows, n_max=50):
    """공통 시험 행과 지원 후보. 시험 = 창 밖 04-16 정상과 창 밖 공격(.88.81, .242.104). 창 안 행은 모든 N에서 시험 제외."""
    day = flows[flows.day == TEST_DAY]
    attacks = day[(day.label == ATTACK) & (day.server_ip == NEW_VICTIM)]
    w0, w1, ordered, _ = support_window(attacks, n_max)
    end = day.start_ts + day.duration
    inside = (day.start_ts >= w0) & (day.start_ts <= w1)
    cross = int(((day.start_ts < w0) & (end >= w0)).sum()), int(((day.start_ts <= w1) & (end > w1)).sum())
    test = day[day.start_ts > w1]  # 합의한 '창 이후' 시험. 창 시작 전 행(정상 105)과 창 안 행은 모두 제외
    before = int((day.start_ts < w0).sum())
    window_benign = day[inside & (day.label == BENIGN)]
    return dict(test=test, support_attacks=ordered, window_benign=window_benign, window=(w0, w1), boundary_cross=cross,
                before_window_rows=before, before_window_cross_rows=cross[0],
                excluded_attacks=int((inside & (day.label == ATTACK) & (day.server_ip == NEW_VICTIM)).sum()))


def prefix_keys(x):
    return [tuple(r) for r in x.to_numpy().tolist()]


def b1_run(flows, k, kind, seed, split, ns=B1_NS):
    """N개의 .88.81 공격 라벨(oracle 지원)과 의심 칸 정상 소수를 학습에 추가했을 때 공통 시험의 recall·오탐. 학습 기준선은 04-11 단독."""
    from src.common.models import make_model

    make = (lambda f: prefix_features(f, k, "seq")) if kind == "seq" else (lambda f: transform(f, k, "logb"))
    base = flows[flows.day == B1_TRAIN_DAY]
    benign = flows.label == BENIGN
    train_a = flows[benign & flows.day.isin(TRAIN_DAYS)]
    valid = flows[benign & (flows.day == VALID_DAY)]
    iso = IsolationForest(n_estimators=100, random_state=seed, n_jobs=2).fit(make(train_a))
    tau = budget_threshold(anomaly_scores(iso, make(valid)), *SUSPECT_BUDGET)["tau"]
    wb = split["window_benign"]
    flagged_benign = wb[anomaly_scores(iso, make(wb)) > tau] if len(wb) else wb  # 사람이 의심 칸에서 확인한 정상(모든 N 공통)
    sa = split["support_attacks"]
    first50 = sa.head(max(ns))
    policy = dict(first50_attacks=len(first50),
                  first50_flagged=int((anomaly_scores(iso, make(first50)) > tau).sum()),
                  reviewed_benign=len(flagged_benign), window_benign=len(wb))
    test = split["test"]
    t_attack = test[(test.label == ATTACK) & (test.server_ip == NEW_VICTIM)]
    t_242 = test[(test.label == ATTACK) & (test.server_ip != NEW_VICTIM)]
    t_benign = test[test.label == BENIGN]
    xt = {name: make(g) for name, g in (("attack", t_attack), ("a242", t_242), ("benign", t_benign))}
    rows, details = [], []
    roles = pd.concat([
        pd.DataFrame(dict(pcap=first50.pcap.to_numpy(), flow_id=first50.flow_id.to_numpy(), role="support_attack_order",
                          order=np.arange(1, len(first50) + 1), score=anomaly_scores(iso, make(first50)))),
        pd.DataFrame(dict(pcap=flagged_benign.pcap.to_numpy(), flow_id=flagged_benign.flow_id.to_numpy(), role="support_benign",
                          order=0, score=anomaly_scores(iso, make(flagged_benign)) if len(flagged_benign) else np.array([])))],
        ignore_index=True).assign(kind=kind, k=k, seed=seed, tau=tau)
    groups_b = {"attack": t_attack, "a242": t_242, "benign": t_benign}
    for n in ns:
        support = sa.head(n)
        train = pd.concat([base, support, flagged_benign], ignore_index=True)
        model = make_model("rf", seed=seed, workers=2).fit(make(train), train.label)
        pred = {name: model.predict(x) for name, x in xt.items()}
        details.append(pd.concat([pd.DataFrame(dict(pcap=g.pcap.to_numpy(), flow_id=g.flow_id.to_numpy(), role=name,
                                                    pred_attack=(pred[name] == ATTACK).astype(np.int8)))
                                  for name, g in groups_b.items()], ignore_index=True).assign(kind=kind, k=k, seed=seed, n=n))
        new = prefix_keys(make(support)) if n else []
        unseen = np.array([key not in set(new) for key in prefix_keys(xt["attack"])])
        short = (t_benign.payload_packets < k).to_numpy()
        rows.append(dict(kind=kind, k=k, seed=seed, n=n, **policy, train_rows=len(train), test_attack_rows=len(t_attack),
                         recall_88_81=float((pred["attack"] == ATTACK).mean()),
                         unseen_prefix_rows=int(unseen.sum()),
                         recall_88_81_unseen_prefix=float((pred["attack"][unseen] == ATTACK).mean()) if unseen.any() else np.nan,
                         recall_242_104=float((pred["a242"] == ATTACK).mean()) if len(t_242) else np.nan,
                         benign_test_rows=len(t_benign), benign_fpr=float((pred["benign"] == ATTACK).mean()),
                         benign_short_rows=int(short.sum()),
                         benign_short_fpr=float((pred["benign"][short] == ATTACK).mean()) if short.any() else np.nan))
    return rows, pd.concat(details, ignore_index=True), roles


def b1_main(args):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_b1_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows = load_flows(args.flows_run_id, args.xml_flows_run_id, DATA_DIR)
    split = b1_split(flows)
    runs = [b1_run(flows, k, kind, seed, split) for kind in INPUTS for k in B1_KS for seed in SEEDS]
    result = pd.DataFrame([r for rows, _, _ in runs for r in rows])
    data_out = DATA_DIR / "predictions" / f"track_b_https_b1_{args.run_id}"
    data_out.mkdir(parents=True)
    pd.concat([d for _, d, _ in runs], ignore_index=True).to_csv(data_out / "b1_test_predictions.csv.gz", index=False)
    pd.concat([r for _, _, r in runs], ignore_index=True).to_csv(data_out / "b1_support_roles.csv.gz", index=False, float_format="%.17g")
    out.mkdir(parents=True)
    result.to_csv(out / "b1.csv", index=False)
    result.groupby(["kind", "k", "n"]).mean(numeric_only=True).reset_index().to_csv(out / "summary.csv", index=False)
    test_ids = split["test"][["pcap", "flow_id"]].sort_values(["pcap", "flow_id"])
    test_ids.to_csv(data_out / "common_test_rows.csv", index=False)  # 행별 분할은 DATA_DIR에만(저장소에는 행수·정의·SHA만)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, xml_flows_run_id=args.xml_flows_run_id, ns=list(B1_NS), ks=list(B1_KS),
        seeds=list(SEEDS), inputs=list(INPUTS), train_day=B1_TRAIN_DAY, bundle_gap_s=BUNDLE_GAP_S, window_after_s=WINDOW_AFTER_S,
        window=list(split["window"]), common_test_rows=len(test_ids), common_test_rows_sha256=sha256_file(data_out / "common_test_rows.csv"),
        common_test_rows_sha256_lf=hashlib.sha256((data_out / "common_test_rows.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        common_test_rows_path=f"predictions/track_b_https_b1_{args.run_id}/common_test_rows.csv", environment=run_environment(root),
        test_definition="04-16 flows with start_ts > window_end (합의한 창 이후). 창 시작 전 행과 창 안 행은 제외",
        before_window_rows_excluded=split["before_window_rows"],
        predictions_files={f.name: sha256_file(f) for f in sorted(data_out.iterdir())},
        excluded_88_81_attacks=split["excluded_attacks"], window_benign_rows=len(split["window_benign"]),
        boundary_cross=dict(start_before_w0_and_end_ge_w0=split["boundary_cross"][0], start_le_w1_and_end_gt_w1=split["boundary_cross"][1]),
        flows_sha256=sha256_file(DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"),
        xml_flows_sha256=sha256_file(DATA_DIR / "processed" / f"track_b_{args.xml_flows_run_id}" / "flows.csv"),
        code_sha256={"src/track_b/https_unseen.py": sha256_file(Path(__file__))},
        note="oracle 지원 N개 라벨(사람이 발견한 결과가 아님). 04-17 검증 문턱을 04-16에 적용하는 오프라인 전이 평가. "
             "모든 N에 같은 시험 행. 탐색적 결과이며 같은 캠페인 반복 요청을 외운 효과가 일반화 근거가 아니다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    s = result.groupby(["kind", "k", "n"]).mean(numeric_only=True).reset_index()
    print(s[["kind", "k", "n", "recall_88_81", "recall_88_81_unseen_prefix", "unseen_prefix_rows", "benign_fpr", "benign_short_fpr",
             "first50_flagged", "reviewed_benign"]].round(4).to_string(index=False))
    print("test rows", len(test_ids), "excluded attacks", split["excluded_attacks"], "window", split["window"], "DONE", out, flush=True)


def suspect_main(args):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_suspect_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows = load_flows(args.flows_run_id, args.xml_flows_run_id, DATA_DIR)
    pairs = [suspect_run(flows, k, kind, seed) for kind in INPUTS for k in SUSPECT_KS for seed in SEEDS]
    result = pd.DataFrame([r for r, _ in pairs])
    data_out = DATA_DIR / "predictions" / f"track_b_https_suspect_{args.run_id}"
    data_out.mkdir(parents=True)
    pd.concat([d for _, d in pairs], ignore_index=True).to_csv(data_out / "suspect_rows.csv.gz", index=False, float_format="%.17g")
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
        rows_file=dict(path=f"predictions/track_b_https_suspect_{args.run_id}/suspect_rows.csv.gz",
                       sha256=sha256_file(data_out / "suspect_rows.csv.gz")), environment=run_environment(root),
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
    b = sub.add_parser("b1")
    b.add_argument("--flows-run-id", required=True)
    b.add_argument("--xml-flows-run-id", required=True)
    b.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
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
    if args.cmd == "b1":
        return b1_main(args)
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
