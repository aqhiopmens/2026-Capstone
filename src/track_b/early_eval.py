"""트랙 B: 앞 k개 페이로드 패킷으로 정상/공격을 분류하는 시간 분할 평가 (k=5·10).

04-12 학습 → 04-17 시험, 그리고 역방향. 모델은 RF·LightGBM 고정 설정(src/common/models.py),
시드 0·1·2는 모델 시드만 바꾼다(분할은 날짜라 고정). 같은 입력에서 **패킷 수만 쓰는 기준선**
(`counts`)을 함께 돌려, 모델이 "공격은 짧고 일정하다"보다 나은지 확인한다.
결과는 '초기 페이로드 패킷 수에 따른 오프라인 분류 성능'이다. 실시간 지연·운영 오탐의 근거가 아니다.

Run: uv run python -m src.track_b.early_eval --flows-run-id 2026-10-31 --run-id <ID>
"""

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common.experiment import run_classification
from src.common.results import summarize_runs
from src.track_b.pcap_flows import K_MAX, PADDING_POLICY

KS = (5, 10)
LABELS = ["benign", "bruteforce_xml"]
MODELS = ("rf", "lgbm")
SEEDS = (0, 1, 2)
FEATURE_KINDS = ("seq", "counts")
DIRECTIONS = (("2021-04-12", "2021-04-17"), ("2021-04-17", "2021-04-12"))
SEQ_COLUMNS = [f"seq_{i}" for i in range(K_MAX)]


def prefix_features(flows, k, kind):
    """앞 k개만 남긴 입력. seq는 30칸 구조 유지, counts는 앞 k개에서 계산한 4개 요약값만 쓴다."""
    seq = np.where(np.arange(K_MAX)[None, :] < k, flows[SEQ_COLUMNS].to_numpy(), 0)
    if kind == "seq":
        return pd.DataFrame(seq, columns=SEQ_COLUMNS, index=flows.index)
    if kind == "counts":
        return pd.DataFrame(dict(n_seen=(seq != 0).sum(axis=1), total_abs=np.abs(seq).sum(axis=1),
                                 total_up=np.where(seq > 0, seq, 0).sum(axis=1),
                                 total_down=np.where(seq < 0, -seq, 0).sum(axis=1)), index=flows.index)
    raise ValueError("알 수 없는 입력 종류")


def subset_scores(actual, predicted):
    """고정 2클래스 기준. 부분집합에 없는 클래스의 recall은 NA, macro-F1은 zero_division=0."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    scores = dict(rows=len(actual))
    for label in LABELS:
        support = int((actual == label).sum())
        scores[f"support_{label}"] = support
        scores[f"recall_{label}"] = (float(((predicted == label) & (actual == label)).sum() / support)
                                     if support else np.nan)
    scores["accuracy"] = float((actual == predicted).mean()) if len(actual) else np.nan
    scores["macro_f1"] = (float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0))
                          if len(actual) else np.nan)
    return scores


def latency_table(flows, ks=KS):
    """k번째 페이로드 패킷에 도달한 플로우의 도달 시각(첫 패킷 대비 초). 판정 시점의 하한이다."""
    rows = []
    for (day, label), group in flows.groupby(["day", "label"]):
        for k in ks:
            reached = group[group.payload_packets >= k]
            times = pd.to_numeric(reached[f"t_{k - 1}"], errors="coerce").dropna()
            rows.append(dict(day=day, label=label, k=k, rows=len(group), reached_k_rows=len(reached),
                             short_rows=len(group) - len(reached),
                             short_fraction=round((len(group) - len(reached)) / len(group), 6),
                             padding_policy=PADDING_POLICY,
                             latency_median_s=times.median() if len(times) else np.nan,
                             latency_p90_s=times.quantile(0.9) if len(times) else np.nan))
    return pd.DataFrame(rows)


def run_all(flows, workers=2):
    from src.common.config import MACHINE

    flows = flows.reset_index(drop=True)
    labels = flows["label"]
    if set(labels) != set(LABELS):
        raise ValueError("정상·공격 라벨 2종이 필요합니다")
    metrics, subsets, predictions = [], [], []
    for train_day, test_day in DIRECTIONS:
        split = f"time_{train_day[5:].replace('-', '')}to{test_day[5:].replace('-', '')}"
        partitions = pd.Series(np.where(flows.day == train_day, "train",
                                        np.where(flows.day == test_day, "test", "excluded")), index=flows.index)
        for k in KS:
            reached = (flows.payload_packets >= k).to_numpy()
            for kind in FEATURE_KINDS:
                x = prefix_features(flows, k, kind)
                for model in MODELS:
                    for seed in SEEDS:
                        result = run_classification(x, labels, partitions, known_labels=LABELS,
                                                    model_name=model, seed=seed, split=split, workers=workers)
                        identity = dict(track="B", experiment="early_k", split=split,
                                        feature_group=f"{kind}_k{k}", model=model, machine=MACHINE,
                                        sklearn=version("scikit-learn"), lightgbm=version("lightgbm"),
                                        torch="", seed=seed, split_seed=0, k=k, feature_kind=kind)
                        metrics.append({**identity, **result.metrics})
                        test = result.predictions[result.predictions.partition == "test"].copy()
                        test["reached_k"] = reached[test.row_id.to_numpy()]
                        predictions.append(test.assign(**identity))
                        for name, part in (("all", test), ("reached_k", test[test.reached_k]),
                                           ("short", test[~test.reached_k])):
                            subsets.append({**identity, "subset": name,
                                            **subset_scores(part.actual, part.predicted)})
    return pd.DataFrame(metrics), pd.DataFrame(subsets), pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    flows_path = DATA_DIR / "processed" / f"track_b_{args.flows_run_id}" / "flows.csv"
    out = root / "outputs/results" / f"track_b_early_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_early_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows = pd.read_csv(flows_path, keep_default_na=False)
    flows = flows.assign(**{c: pd.to_numeric(flows[c], errors="coerce") for c in
                            [f"t_{i}" for i in range(K_MAX)]})
    metrics, subsets, predictions = run_all(flows, workers=args.workers)
    summary = summarize_runs(metrics)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    subsets.to_csv(out / "subsets.csv", index=False)
    latency_table(flows).to_csv(out / "latency.csv", index=False)
    predictions.to_csv(data_out / "predictions.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, ks=list(KS), models=list(MODELS), seeds=list(SEEDS),
        directions=[list(d) for d in DIRECTIONS], feature_kinds=list(FEATURE_KINDS), labels=LABELS,
        padding_policy=PADDING_POLICY, runs=len(metrics),
        note="시드는 모델 시드만 바꾼다. counts는 패킷 수·길이 합만 쓰는 기준선. 오프라인 분류 성능이며 "
             "실시간 지연·운영 오탐의 근거가 아니다. 공격 플로우가 모두 12개 이하라 k=20·30은 평가하지 않는다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[["split", "feature_group", "model", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]]
          .to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
