"""트랙 B HTTPS: 입력을 덜 취약하게 바꿨을 때(크기 로그 구간화, 방향만) 약한 조건이 어떻게 달라지는지 짝 비교.

토론 안건 4(11/4) 합의 범위. 기존 평가(https_eval)와 같은 분할·시험행·RF/LGBM 설정·시드를 쓰고 입력 2종만 바꾼다.
- logb: 앞 k개 값만 x=0이면 0, 그 외 sign(x)*(1+floor(log2|x|)). 구간 경계는 사전 고정(시험 분포로 정하지 않음). 나머지 칸은 0 패딩
- dir : 앞 k개 값의 부호(sign)만. 길이 의존성 진단이며 개선 입력으로 주장하지 않는다
결과는 시험에서 약한 조건을 본 뒤 고른 후속이므로 '탐색적 개선/악화'이며 독립 최종 검증이 아니다.
기준선은 이미 저장된 Colab 최종 결과(원본 seq)이고 원본은 다시 학습하지 않는다.

Run: uv run python -m src.track_b.https_robust --flows-run-id 2026-11-02_v1 --audit-run-id 2026-11-02_v2 --run-id <ID>
"""

import argparse
import json
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.results import sha256_file, summarize_runs
from src.track_b.early_eval import MODELS, SEEDS, prefix_features
from src.track_b.https_eval import DIRECTIONS, KS, SCOPES, check_inputs, load_flows, run_all
from src.track_b.pcap_flows import PADDING_POLICY

KINDS = ("logb", "dir")
BASELINE = "track_b_https_eval_2026-11-04_colab"
PAIR_SUBSETS = ("all", "attack_242104", "attack_88_81", "benign_csv_confirmed", "benign_csv_unmatched")


def transform(flows, k, kind):
    """원본 seq의 앞 k개만 변환. 0 패딩과 k 이후 칸은 그대로 0이다."""
    seq = prefix_features(flows, k, "seq")
    a = seq.to_numpy().astype(np.int64)
    mag = np.abs(a)
    if kind == "logb":
        out = np.where(mag > 0, np.sign(a) * (1 + np.floor(np.log2(np.where(mag > 0, mag, 1))).astype(np.int64)), 0)
    elif kind == "dir":
        out = np.sign(a)
    else:
        raise ValueError("알 수 없는 입력 종류")
    return pd.DataFrame(out, columns=seq.columns, index=seq.index)


def paired_tables(subsets, summary, root):
    """새 입력 결과를 같은 조건의 원본 seq(저장된 Colab 최종)와 짝 비교한다."""
    base_subsets = pd.read_csv(root / "outputs/results" / BASELINE / "subsets.csv")
    base_summary = pd.read_csv(root / "outputs/results" / BASELINE / "summary.csv")
    base_subsets = base_subsets[base_subsets.feature_kind == "seq"]
    keys = ["split", "k", "model", "subset"]
    metrics = ["macro_f1", "recall_bruteforce_https", "recall_benign"]

    def mean_by(frame):
        return frame.groupby(keys + (["feature_kind"] if "feature_kind" in frame else []))[metrics + ["rows"]].mean().reset_index()

    new_m, base_m = mean_by(subsets), mean_by(base_subsets).drop(columns="rows")
    delta = new_m.merge(base_m, on=keys, suffixes=("_new", "_seq"))
    for m in metrics:
        delta[f"d_{m}"] = delta[f"{m}_new"] - delta[f"{m}_seq"]
    delta = delta[delta.subset.isin(PAIR_SUBSETS)]
    summary = summary.assign(k=summary.feature_group.str.extract(r"k(\d+)").astype(int)[0],
                             feature_kind=summary.feature_group.str.split("_").str[0])
    base_summary = base_summary.assign(k=base_summary.feature_group.str.extract(r"k(\d+)").astype(int)[0],
                                       feature_kind=base_summary.feature_group.str.split("_").str[0])
    base_summary = base_summary[base_summary.feature_kind == "seq"]
    dsum = summary.merge(base_summary, on=["split", "k", "model"], suffixes=("_new", "_seq"))
    dsum["d_macro_f1"] = dsum.macro_f1_mean_new - dsum.macro_f1_mean_seq
    return delta, dsum[["split", "k", "model", "feature_kind_new", "macro_f1_mean_new", "macro_f1_mean_seq", "d_macro_f1"]]


def scope_pair(root):
    """저장된 원본(Colab 최종) 결과만으로: .88.81을 학습에 포함(all) vs 제외(only_242104)했을 때 같은 시험행의 차이.

    attack_242104·benign 부분집합은 두 범위의 시험행이 같다(.88.81 공격만 all에만 있음). 추가 학습 없음.
    """
    sub = pd.read_csv(root / "outputs/results" / BASELINE / "subsets.csv")
    sub = sub[sub.subset.isin(["attack_242104", "benign_csv_confirmed", "benign_csv_unmatched"])]
    sub = sub.assign(direction=sub.split.str.replace(r"_(all|only_242104)$", "", regex=True), scope=sub.scope)
    g = sub.groupby(["direction", "feature_kind", "k", "model", "subset", "scope"])[
        ["recall_bruteforce_https", "recall_benign", "rows"]].mean().reset_index()
    a = g[g.scope == "all"].drop(columns="scope")
    b = g[g.scope == "only_242104"].drop(columns="scope")
    out = a.merge(b, on=["direction", "feature_kind", "k", "model", "subset"], suffixes=("_all", "_only"))
    out["d_attack_recall"] = out.recall_bruteforce_https_all - out.recall_bruteforce_https_only
    out["d_benign_recall"] = out.recall_benign_all - out.recall_benign_only
    out["same_test_rows"] = out.rows_all == out.rows_only
    return out


def benign_overlap(flows):
    """정상 서버 IP가 학습·시험 날짜에 겹치는 정도. 식별자는 입력이 아니며 새 서버 일반화 주장을 막기 위한 수치."""
    rows = []
    benign = flows[(flows.label == "benign") & (flows.payload_packets > 0)]
    for train_day, test_day in DIRECTIONS:
        tr, te = benign[benign.day == train_day], benign[benign.day == test_day]
        shared = set(tr.server_ip) & set(te.server_ip)
        rows.append(dict(train_day=train_day, test_day=test_day, train_flows=len(tr), test_flows=len(te),
                         train_servers=tr.server_ip.nunique(), test_servers=te.server_ip.nunique(),
                         shared_servers=len(shared), test_flows_to_shared_servers=int(te.server_ip.isin(shared).sum()),
                         test_flows_to_shared_fraction=round(float(te.server_ip.isin(shared).mean()), 6)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_robust_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_https_robust_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows_path, status_path = check_inputs(root, args.flows_run_id, args.audit_run_id, DATA_DIR)
    flows = load_flows(flows_path, status_path)
    metrics, subsets, predictions = run_all(flows, workers=args.workers, kinds=KINDS, features=transform,
                                            experiment="https_robust_k")
    summary = summarize_runs(metrics)
    delta, dsum = paired_tables(subsets, summary, root)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    for name, frame in (("metrics", metrics), ("summary", summary), ("subsets", subsets),
                        ("paired_subsets", delta), ("paired_summary", dsum), ("scope_pair", scope_pair(root)),
                        ("benign_overlap", benign_overlap(flows))):
        frame.to_csv(out / f"{name}.csv", index=False)
    predictions.to_csv(data_out / "predictions.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, audit_run_id=args.audit_run_id, baseline=BASELINE,
        flows_sha256=sha256_file(flows_path), flow_status_sha256=sha256_file(status_path),
        baseline_summary_sha256=sha256_file(root / "outputs/results" / BASELINE / "summary.csv"),
        baseline_subsets_sha256=sha256_file(root / "outputs/results" / BASELINE / "subsets.csv"),
        predictions_sha256=sha256_file(data_out / "predictions.csv"),
        code_sha256={name: sha256_file(root / name) for name in (
            "src/track_b/https_robust.py", "src/track_b/https_eval.py", "src/track_b/early_eval.py",
            "src/common/experiment.py", "src/common/models.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        versions=dict(sklearn=version("scikit-learn"), lightgbm=version("lightgbm"), numpy=version("numpy"),
                      pandas=version("pandas")), ks=list(KS), models=list(MODELS), seeds=list(SEEDS),
        directions=[list(d) for d in DIRECTIONS], scopes=list(SCOPES), kinds=list(KINDS), padding_policy=PADDING_POLICY,
        runs=len(metrics),
        note="시험에서 약한 조건을 본 뒤 고른 후속이므로 탐색적 개선/악화이며 독립 최종 검증이 아니다. "
             "dir은 길이 의존성 진단이다. 새 서버 일반화 주장 금지"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(dsum.groupby(["split", "feature_kind_new"]).d_macro_f1.agg(["mean", "min", "max"]).round(4).to_string())
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
