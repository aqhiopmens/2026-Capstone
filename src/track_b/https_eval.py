"""트랙 B: HIKARI Bruteforce-HTTPS 앞 k개 페이로드 패킷 시간 분할 평가 (04-11 ↔ 04-16, k=5·10·20·30).

라벨은 캡처 단위 **잠정** 규칙(공격자 → 피해자:443 = 공격)이다. 이 평가는 '잠정 라벨을 예측하는 탐색 평가'이며
실제 공격 정답 검증이나 일반화 증거가 아니다. 사전 합의(Slack, 11/2):
- 범위 2종: all(공격 전 피해자) / only_242104(공격이 .242.104인 플로우만; 피해자·날짜 교락 통제)
- payload 0 플로우는 두 범위 모두 평가에서 제외하고 개수를 따로 보고한다
- 입력 2종: seq(앞 k개 시퀀스) / counts(앞 k개에서 계산한 4개 요약)
- RF·LightGBM × 시드 0·1·2 → 2방향 × 2범위 × 4k × 2입력 × 2모델 × 3시드 = 192회
- 부분집합(같은 저장 시험 예측에서 계산, 추가 학습 없음): CSV 일치 정상은 FPR·specificity·분모만 읽는다.
  공격 recall·전체 macro-F1의 대체 검증이 아니다.

Run: uv run python -m src.track_b.https_eval --flows-run-id 2026-11-02_v1 --audit-run-id 2026-11-02_v2 --run-id <ID>
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
from sklearn.tree import DecisionTreeClassifier

from src.common.experiment import run_classification
from src.common.results import sha256_file, summarize_runs
from src.track_b.early_eval import MODELS, SEEDS, FEATURE_KINDS, latency_table, prefix_features, subset_scores
from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.pcap_flows import K_MAX, PADDING_POLICY

KS = (5, 10, 20, 30)
LABELS = [BENIGN, ATTACK]
DIRECTIONS = (("2021-04-11", "2021-04-16"), ("2021-04-16", "2021-04-11"))
SCOPES = ("all", "only_242104")
SOURCE_ONLY = "128.199.242.104"
CONFIRMED = ("single_agree", "multi_agree")


def partitions_for(flows, train_day, test_day, scope):
    """payload 0 플로우와, only_242104에서 다른 피해자로 가는 공격 플로우는 excluded."""
    eligible = flows.payload_packets > 0
    if scope == "only_242104":
        eligible &= ~((flows.label == ATTACK) & (flows.server_ip != SOURCE_ONLY))
    elif scope != "all":
        raise ValueError("알 수 없는 범위")
    day = np.where(flows.day == train_day, "train", np.where(flows.day == test_day, "test", "excluded"))
    return pd.Series(np.where(eligible, day, "excluded"), index=flows.index)


def subset_masks(test, k):
    """시험 행(플로우 정보 포함)을 나누는 부분집합. 이름 -> 불리언 마스크."""
    attack, benign = test.label == ATTACK, test.label == BENIGN
    confirmed = test.status_key_payload.isin(CONFIRMED)
    return {"all": pd.Series(True, index=test.index),
            "reached_k": test.payload_packets >= k, "short": test.payload_packets < k,
            "not_midstream": test.midstream == 0,
            "benign_csv_confirmed": benign & confirmed, "benign_csv_unmatched": benign & ~confirmed,
            "attack_242104": attack & (test.server_ip == SOURCE_ONLY),
            "attack_88_81": attack & (test.server_ip != SOURCE_ONLY)}


def scored(actual, predicted):
    """한 클래스가 없는 부분집합에서는 macro-F1을 비운다(읽지 않도록)."""
    scores = subset_scores(actual, predicted, labels=LABELS)
    if min(scores[f"support_{label}"] for label in LABELS) == 0:
        scores["macro_f1"] = np.nan
    return scores


def diagnostics(flows):
    """다수 클래스·첫 패킷 길이 하나·받은 패킷 수 하나만 쓴 깊이 2 트리의 시험 정확도(기준선)."""
    rows = []
    for train_day, test_day in DIRECTIONS:
        for scope in SCOPES:
            part = partitions_for(flows, train_day, test_day, scope)
            train, test = part.eq("train").to_numpy(), part.eq("test").to_numpy()
            direction = f"{train_day[5:].replace('-', '')}to{test_day[5:].replace('-', '')}"
            majority = flows.label[train].value_counts().idxmax()
            rows.append(dict(kind="majority", scope=scope, direction=direction, k=0, name=majority,
                             accuracy=float((flows.label[test] == majority).mean())))
            for k in KS:
                x = prefix_features(flows, k, "counts").assign(seq_0=flows["seq_0"].to_numpy())
                for name in ("seq_0", "n_seen"):
                    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x.loc[train, [name]], flows.label[train])
                    rows.append(dict(kind="single_feature", scope=scope, direction=direction, k=k, name=name,
                                     accuracy=float(tree.score(x.loc[test, [name]], flows.label[test]))))
    return pd.DataFrame(rows)


def run_all(flows, workers=2):
    from src.common.config import MACHINE

    flows = flows.reset_index(drop=True)
    today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    note = (f"{MACHINE} 로컬 실행. HIKARI Bruteforce-HTTPS 캡처 단위 잠정 라벨을 예측하는 탐색 평가이며 "
            "실제 공격 정답 검증·일반화 증거가 아님")
    if set(flows.label) != set(LABELS):
        raise ValueError("정상·공격 라벨 2종이 필요합니다")
    metrics, subsets, predictions = [], [], []
    for train_day, test_day in DIRECTIONS:
        for scope in SCOPES:
            split = f"time_{train_day[5:].replace('-', '')}to{test_day[5:].replace('-', '')}_{scope}"
            partitions = partitions_for(flows, train_day, test_day, scope)
            for k in KS:
                for kind in FEATURE_KINDS:
                    x = prefix_features(flows, k, kind)
                    for model in MODELS:
                        for seed in SEEDS:
                            result = run_classification(x, flows.label, partitions, known_labels=LABELS,
                                                        model_name=model, seed=seed, split=split, workers=workers)
                            identity = dict(date=today, track="B", experiment="https_early_k", split=split,
                                            feature_group=f"{kind}_k{k}", model=model, machine=MACHINE,
                                            sklearn=version("scikit-learn"), lightgbm=version("lightgbm"),
                                            torch="", seed=seed, split_seed=0, k=k, feature_kind=kind,
                                            scope=scope, note=note)
                            metrics.append({**identity, **result.metrics})
                            test = result.predictions[result.predictions.partition == "test"].copy()
                            context = flows.loc[test.row_id.to_numpy(), ["server_ip", "payload_packets", "midstream",
                                                                          "status_key_payload", "label"]]
                            context.index = test.index
                            test = test.join(context.drop(columns="label"))
                            predictions.append(test.assign(**identity))
                            full = test.assign(label=test.actual)
                            for name, mask in subset_masks(full, k).items():
                                part = full[mask]
                                subsets.append({**identity, "subset": name, **scored(part.actual, part.predicted)})
    return pd.DataFrame(metrics), pd.DataFrame(subsets), pd.concat(predictions, ignore_index=True)


def check_inputs(root, flows_run_id, audit_run_id, data_dir):
    """flows.csv·flow_status.csv의 SHA-256이 추출·감사 metadata와 같은지 학습 전에 검사한다."""
    flows_path = data_dir / "processed" / f"track_b_https_{flows_run_id}" / "flows.csv"
    status_path = data_dir / "processed" / f"track_b_https_audit_{audit_run_id}" / "flow_status.csv"
    extract = json.loads((root / "outputs/results" / f"track_b_https_extract_{flows_run_id}" / "metadata.json")
                         .read_text(encoding="utf-8"))
    audit = json.loads((root / "outputs/results" / f"track_b_https_audit_{audit_run_id}" / "metadata.json")
                       .read_text(encoding="utf-8"))
    if sha256_file(flows_path) != extract["flows_sha256"] or sha256_file(status_path) != audit["flow_status_sha256"]:
        raise ValueError("flows.csv 또는 flow_status.csv SHA-256이 metadata와 다릅니다")
    if audit["flows_sha256"] != extract["flows_sha256"]:
        raise ValueError("감사가 본 flows.csv와 추출 flows.csv가 다릅니다")
    return flows_path, status_path


def load_flows(flows_path, status_path):
    flows = pd.read_csv(flows_path, keep_default_na=False)
    flows = flows.assign(**{c: pd.to_numeric(flows[c], errors="coerce") for c in [f"t_{i}" for i in range(K_MAX)]})
    status = pd.read_csv(status_path, usecols=["pcap", "flow_id", "status_key_payload"])
    merged = flows.merge(status, on=["pcap", "flow_id"], how="left", validate="one_to_one")
    if merged.status_key_payload.isna().any() or len(merged) != len(flows):
        raise ValueError("flow_status와 flows가 1:1로 맞지 않습니다")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_https_eval_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_https_eval_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows_path, status_path = check_inputs(root, args.flows_run_id, args.audit_run_id, DATA_DIR)
    flows = load_flows(flows_path, status_path)
    metrics, subsets, predictions = run_all(flows, workers=args.workers)
    summary = summarize_runs(metrics)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    subsets.to_csv(out / "subsets.csv", index=False)
    latency_table(flows, KS).to_csv(out / "latency.csv", index=False)
    diagnostics(flows).to_csv(out / "diagnostics.csv", index=False)
    predictions.to_csv(data_out / "predictions.csv", index=False)
    excluded = (flows.payload_packets == 0).groupby([flows.day, flows.label]).sum().rename("payload0_excluded")
    excluded.reset_index().to_csv(out / "payload0_excluded.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, audit_run_id=args.audit_run_id,
        flows_sha256=sha256_file(flows_path), flow_status_sha256=sha256_file(status_path),
        predictions_sha256=sha256_file(data_out / "predictions.csv"),
        code_sha256={name: sha256_file(root / name) for name in (
            "src/track_b/https_eval.py", "src/track_b/early_eval.py", "src/track_b/https_flows.py",
            "src/track_b/pcap_flows.py", "src/common/experiment.py", "src/common/models.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        versions=dict(sklearn=version("scikit-learn"), lightgbm=version("lightgbm"), numpy=version("numpy"),
                      pandas=version("pandas")), ks=list(KS), models=list(MODELS), seeds=list(SEEDS),
        directions=[list(d) for d in DIRECTIONS], scopes=list(SCOPES), feature_kinds=list(FEATURE_KINDS),
        labels=LABELS, padding_policy=PADDING_POLICY, runs=len(metrics),
        note="캡처 단위 잠정 라벨을 예측하는 탐색 평가. 실제 공격 정답 검증·일반화 증거·실시간 지연 근거가 아니다. "
             "CSV 일치 정상 부분집합은 FPR·specificity·분모만 읽는다."), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(summary[["split", "feature_group", "model", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]]
          .to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
