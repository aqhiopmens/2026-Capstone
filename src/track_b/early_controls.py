"""트랙 B 통제 실험: 앞 k개 평가(#22)의 높은 점수가 어디서 나오는지 단독 변형으로 확인한다.

변형(각각 따로 적용, train·test에 같은 규칙):
  mask_first  seq_0만 0으로 둔다(열을 당기지 않음)
  shuffle     앞 k개 안의 비영 값을 행 ID+시드로 독립 순열한다(0 패딩 위치 고정)
  abs         방향 부호를 없애 절댓값만 쓴다
원본 앞 k개를 먼저 자른 뒤 변형한다. reached/short는 변형 전 payload_packets로 정한다.
원본 기준선은 #22 저장 예측을 SHA·분할·설정 확인 후 재사용해 같은 시험 행에서 비교한다.
결과는 '이 설정에서 해당 요소의 추가 이득'까지만 말하며 조기 탐지 능력의 근거가 아니다.

Run: uv run python -m src.track_b.early_controls --flows-run-id 2026-10-31_v3 --base-run-id 2026-10-31_v3 --run-id <ID>
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

from src.common.experiment import run_classification
from src.common.results import sha256_file, summarize_runs
from src.track_b.early_eval import (DIRECTIONS, KS, LABELS, MODELS, SEEDS, SEQ_COLUMNS, prefix_features,
                                    read_extract_meta, subset_scores)
from src.track_b.pcap_flows import K_MAX, PADDING_POLICY

CONTROLS = ("mask_first", "shuffle", "abs")
ALL_CONTROLS = CONTROLS + ("mask3",)  # mask3: seq_0~2 마스킹(안건 8). 기본 실행에는 포함하지 않는다(--controls로 지정)


def control_features(flows, k, control, seed):
    """원본 앞 k개 seq를 자른 뒤 변형한다. 30칸 구조와 0 패딩 위치는 유지한다."""
    x = prefix_features(flows, k, "seq")
    values = x.to_numpy().copy()
    if control == "mask_first":
        values[:, 0] = 0
    elif control == "mask3":
        values[:, :3] = 0
    elif control == "abs":
        values = np.abs(values)
    elif control == "shuffle":
        for i, row_id in enumerate(x.index):
            n = int((values[i] != 0).sum())  # 0은 뒤쪽에만 있으므로 비영 값은 앞 n칸
            if n > 1:
                values[i, :n] = values[i, :n][np.random.default_rng([seed, int(row_id)]).permutation(n)]
    else:
        raise ValueError("알 수 없는 통제")
    return pd.DataFrame(values, columns=SEQ_COLUMNS, index=x.index)


def compare_to_baseline(control_test, base_test):
    """같은 시험 행의 원본 대비 변화. 행 ID·정답이 어긋나면 오류."""
    merged = control_test.merge(base_test, on="row_id", suffixes=("", "_base"), validate="one_to_one")
    if len(merged) != len(control_test) or len(merged) != len(base_test) or (merged.actual != merged.actual_base).any():
        raise ValueError("통제와 기준선의 시험 행 또는 정답이 일치하지 않습니다")
    rows = []
    for name, part in (("all", merged), ("reached_k", merged[merged.reached_k]), ("short", merged[~merged.reached_k])):
        now, before = subset_scores(part.actual, part.predicted), subset_scores(part.actual, part.predicted_base)
        rows.append(dict(subset=name, rows=now["rows"], support_benign=now["support_benign"],
                         support_bruteforce_xml=now["support_bruteforce_xml"],
                         macro_f1=now["macro_f1"], macro_f1_base=before["macro_f1"],
                         delta_macro_f1=now["macro_f1"] - before["macro_f1"],
                         accuracy=now["accuracy"], accuracy_base=before["accuracy"],
                         delta_accuracy=now["accuracy"] - before["accuracy"],
                         recall_benign=now["recall_benign"], recall_benign_base=before["recall_benign"],
                         recall_bruteforce_xml=now["recall_bruteforce_xml"],
                         recall_bruteforce_xml_base=before["recall_bruteforce_xml"],
                         newly_wrong=int(((part.predicted != part.actual) & (part.predicted_base == part.actual)).sum()),
                         newly_right=int(((part.predicted == part.actual) & (part.predicted_base != part.actual)).sum())))
    return rows


def load_baseline(root, base_run_id, flows):
    """#22 저장 예측을 SHA·행 라벨 확인 후 읽는다."""
    from src.common.config import DATA_DIR

    meta = json.loads((root / "outputs/results" / f"track_b_early_{base_run_id}" / "metadata.json").read_text(encoding="utf-8"))
    path = DATA_DIR / "predictions" / f"track_b_early_{base_run_id}" / "predictions.csv"
    if sha256_file(path) != meta["predictions_sha256"]:
        raise ValueError("기준선 predictions SHA-256이 metadata와 다릅니다")
    base = pd.read_csv(path, keep_default_na=False)
    base = base[base.feature_kind == "seq"]
    if base.reached_k.dtype != bool:
        base["reached_k"] = base.reached_k.astype(str) == "True"
    if (flows.loc[base.row_id.to_numpy(), "label"].to_numpy() != base.actual.to_numpy()).any():
        raise ValueError("기준선 예측의 정답이 flows 라벨과 다릅니다")
    return base, meta


def run_all(flows, base, workers=2, controls=CONTROLS):
    from src.common.config import MACHINE

    flows = flows.reset_index(drop=True)
    today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    note = (f"{MACHINE} 로컬 통제 실험(최종 수치 아님). 단독 변형의 추가 이득만 본다. 단일 공격 데이터라 "
            "조기 탐지 능력의 근거가 아님")
    labels = flows["label"]
    metrics, deltas, predictions = [], [], []
    for train_day, test_day in DIRECTIONS:
        split = f"time_{train_day[5:].replace('-', '')}to{test_day[5:].replace('-', '')}"
        partitions = pd.Series(np.where(flows.day == train_day, "train",
                                        np.where(flows.day == test_day, "test", "excluded")), index=flows.index)
        for k in KS:
            reached = (flows.payload_packets >= k).to_numpy()
            for control in controls:
                for model in MODELS:
                    for seed in SEEDS:
                        x = control_features(flows, k, control, seed)
                        result = run_classification(x, labels, partitions, known_labels=LABELS,
                                                    model_name=model, seed=seed, split=split, workers=workers)
                        identity = dict(date=today, track="B", experiment="early_control", split=split,
                                        feature_group=f"{control}_k{k}", model=model, machine=MACHINE,
                                        sklearn=version("scikit-learn"), lightgbm=version("lightgbm"),
                                        torch="", seed=seed, split_seed=0, k=k, feature_kind="seq",
                                        control=control, note=note)
                        metrics.append({**identity, **result.metrics})
                        test = result.predictions[result.predictions.partition == "test"].copy()
                        test["reached_k"] = reached[test.row_id.to_numpy()]
                        predictions.append(test.assign(**identity))
                        base_test = base[(base.split == split) & (base.k == k) & (base.model == model)
                                         & (base.seed == seed)][["row_id", "actual", "predicted", "reached_k"]]
                        for row in compare_to_baseline(test[["row_id", "actual", "predicted", "reached_k"]],
                                                       base_test.drop(columns="reached_k")):
                            deltas.append({**identity, **row})
    return pd.DataFrame(metrics), pd.DataFrame(deltas), pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--controls", nargs="+", default=list(CONTROLS), choices=ALL_CONTROLS)
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    flows_path = DATA_DIR / "processed" / f"track_b_{args.flows_run_id}" / "flows.csv"
    out = root / "outputs/results" / f"track_b_control_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_control_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    extract_meta = read_extract_meta(root, args.flows_run_id, flows_path)
    flows = pd.read_csv(flows_path, keep_default_na=False)
    base, base_meta = load_baseline(root, args.base_run_id, flows)
    if base_meta["flows_sha256"] != extract_meta["flows_sha256"]:
        raise ValueError("기준선과 통제 실험의 flows가 다릅니다")
    metrics, deltas, predictions = run_all(flows, base, workers=args.workers, controls=tuple(args.controls))
    summary = summarize_runs(metrics)
    delta_summary = (deltas.groupby(["split", "k", "model", "control", "subset"])
                     [["macro_f1", "macro_f1_base", "delta_macro_f1", "delta_accuracy", "newly_wrong", "newly_right"]]
                     .mean().reset_index())
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    deltas.to_csv(out / "deltas.csv", index=False)
    delta_summary.to_csv(out / "delta_summary.csv", index=False)
    predictions.to_csv(data_out / "predictions.csv", index=False)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, base_run_id=args.base_run_id,
        flows_sha256=sha256_file(flows_path), base_predictions_sha256=base_meta["predictions_sha256"],
        predictions_sha256=sha256_file(data_out / "predictions.csv"),
        code_sha256={name: sha256_file(root / name) for name in ("src/track_b/early_controls.py", "src/track_b/early_eval.py",
                                                                  "src/common/experiment.py", "src/common/models.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        versions=dict(sklearn=version("scikit-learn"), lightgbm=version("lightgbm"), numpy=version("numpy"),
                      pandas=version("pandas")), ks=list(KS), models=list(MODELS), seeds=list(SEEDS),
        directions=[list(d) for d in DIRECTIONS], controls=list(args.controls), padding_policy=PADDING_POLICY,
        runs=len(metrics),
        note="변형은 원본 앞 k개를 자른 뒤 적용. shuffle은 행 ID+모델 시드의 독립 순열. reached/short는 변형 전 "
             "payload_packets 기준. 단독 효과만 해석하며 조기 탐지 능력의 근거가 아니다"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(delta_summary.to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
