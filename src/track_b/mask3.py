"""앞 3칸 마스킹 통제(안건 8 확정안): seq_0~2만 0으로 두고(열을 당기지 않음, 0 패딩 위치 유지) 같은 분할·행·시드·앞 k개에서 원본과 짝 비교한다.

TLS 레코드 매핑 확인 전이라 '앞 3칸 마스킹'이라고만 부른다. 큰 하락(|Δmacro-F1| >= 0.05, 서술용 표기)은 '해당 3칸을 가린 조건에 민감했다'는 관측이고,
작은 변화는 남은 특징으로도 해당 시험 성능이 유지됐다는 범위다. 인과·식별자 대리 변수 단정 금지. 시험 결과를 보고 마스킹 범위·k·모델을 바꾸지 않는다.

하위 명령
  https  : #35 설계(HTTPS 시간 분할 04-11<->04-16, all/only_242104, RF·LightGBM, k=5·10·20·30, 시드 3, seq) 96회. 원본 기준선은 Colab 최종 저장 결과.
  compare: #39(.88.81 적응)·#40(2022 적응)의 마스킹 결과를 저장된 원본 예측과 같은 시험 행에서 짝 비교한다(재학습 없음).
XML(#22/#25 설계)은 `python -m src.track_b.early_controls --controls mask3`로 실행한다.
적응 평가(#39·#40)는 `https_unseen b1 --kinds seq_mask3`, `https_2022 --kinds seq_mask3`로 실행하며, 정상 지원 행(IF 선택)·oracle 공격 순서·공통 시험 ID는 원본과 같게 고정된다.

Run: uv run python -m src.track_b.mask3 https --flows-run-id 2026-11-02_v1 --audit-run-id 2026-11-02_v2 --run-id <ID>
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
from src.track_b.early_eval import MODELS, SEEDS, mask3, prefix_features
from src.track_b.https_2022 import metrics_2022
from src.track_b.https_eval import DIRECTIONS, KS, SCOPES, check_inputs, load_flows, run_all
from src.track_b.https_robust import paired_tables
from src.track_b.https_unseen import b1_metrics
from src.track_b.pcap_flows import PADDING_POLICY

KIND = "seq_mask3"
BIG_DELTA = 0.05  # 서술용 표기 임계(판정 규칙 아님)


def masked_features(flows, k, kind):
    if kind != KIND:
        raise ValueError("알 수 없는 입력 종류")
    return mask3(prefix_features(flows, k, "seq"))


def https_main(args):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_mask3_https_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_mask3_https_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    from src.track_b.https_robust import BASELINE

    flows_path, status_path = check_inputs(root, args.flows_run_id, args.audit_run_id, DATA_DIR)
    flows = load_flows(flows_path, status_path)
    metrics, subsets, predictions = run_all(flows, workers=args.workers, kinds=(KIND,), features=masked_features,
                                            experiment="https_mask3_k")
    summary = summarize_runs(metrics)
    delta, dsum = paired_tables(subsets, summary, root)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    for name, frame in (("metrics", metrics), ("summary", summary), ("subsets", subsets), ("paired_subsets", delta), ("paired_summary", dsum)):
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
            "src/track_b/mask3.py", "src/track_b/https_robust.py", "src/track_b/https_eval.py", "src/track_b/early_eval.py",
            "src/common/experiment.py", "src/common/models.py")},
        git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")),
        versions=dict(sklearn=version("scikit-learn"), lightgbm=version("lightgbm"), numpy=version("numpy"), pandas=version("pandas")),
        ks=list(KS), models=list(MODELS), seeds=list(SEEDS), directions=[list(d) for d in DIRECTIONS], scopes=list(SCOPES),
        kinds=[KIND], padding_policy=PADDING_POLICY, runs=len(metrics), big_delta=BIG_DELTA,
        note="앞 3칸 마스킹. 같은 분할·행·시드·k에서 원본(저장된 Colab 최종)과 짝 비교. 관측 범위이며 인과·식별자 단정 금지. "
             "C1(같은 분포)과 새 피해자(.88.81)는 서브셋에서 구분"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(dsum.groupby(["split", "feature_kind_new"]).d_macro_f1.agg(["mean", "min", "max"]).round(4).to_string())
    print("DONE", out, flush=True)


def _same_rows(a, b, keys, what):
    ka = a[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    kb = b[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if not ka.equals(kb):
        raise ValueError(f"{what}: 원본과 마스킹의 시험 행이 일치하지 않습니다")


def pair_metrics(base, new, group):
    """같은 (k, seed, n)의 원본·마스킹 혼동 지표 짝. 반환: 조건별 행과 (k, n) 시드 평균."""
    keys = ["k", "seed", "n"]
    both = new.merge(base, on=keys, suffixes=("_mask3", "_seq"), validate="one_to_one")
    for m in ("tp", "tn", "fp", "fn", "accuracy", "macro_f1"):
        both[f"d_{m}"] = both[f"{m}_mask3"] - both[f"{m}_seq"]
    both["recall_seq"] = both.tp_seq / (both.tp_seq + both.fn_seq)
    both["recall_mask3"] = both.tp_mask3 / (both.tp_mask3 + both.fn_mask3)
    both["d_recall"] = both.recall_mask3 - both.recall_seq
    cols = ["macro_f1_seq", "macro_f1_mask3", "d_macro_f1", "recall_seq", "recall_mask3", "d_recall", "fp_seq", "fp_mask3", "d_fp"]
    both["d_fp"] = both.fp_mask3 - both.fp_seq
    summ = both.groupby(["k", "n"])[cols].mean().reset_index()
    summ["big_change"] = summ.d_macro_f1.abs() >= BIG_DELTA
    return both.assign(experiment=group), summ.assign(experiment=group)


def role_recall(base, new, tags=("seq", "mask3")):
    """피해자(role)별 공격 recall 짝: (k, n)별 시드 평균과 시험 행 수. role: attack=.88.81, a242=.242.104.
    pair_metrics의 recall은 두 피해자를 합친 전체 공격(all_attack) 값이라 따로 낸다."""
    def one(frame, tag):
        g = frame[frame.role != "benign"].groupby(["k", "n", "seed", "role"]).pred_attack.agg(["mean", "size"])
        g = g.rename(columns={"mean": f"recall_{tag}", "size": "rows"}).reset_index()
        return g.groupby(["k", "n", "role"]).agg(**{f"recall_{tag}": (f"recall_{tag}", "mean"), "rows": ("rows", "first")}).reset_index()
    out = one(base, tags[0]).merge(one(new, tags[1]), on=["k", "n", "role"], validate="one_to_one")
    if not (out.rows_x == out.rows_y).all():
        raise ValueError("원본과 마스킹의 피해자별 시험 행 수가 다릅니다")
    out = out.drop(columns="rows_y").rename(columns={"rows_x": "rows"})
    out["d_recall"] = out[f"recall_{tags[1]}"] - out[f"recall_{tags[0]}"]
    return out.replace({"role": {"attack": ".88.81", "a242": ".242.104"}})


def compare_main(args):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_mask3_adapt_{args.run_id}"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    pdir = DATA_DIR / "predictions"
    files = dict(b1_base=pdir / f"track_b_https_b1_{args.b1_base}" / "b1_test_predictions.csv.gz",
                 b1_new=pdir / f"track_b_https_b1_{args.b1_new}" / "b1_test_predictions.csv.gz",
                 e2022_base=pdir / f"track_b_https2022_eval_{args.e2022_base}" / "eval2022_predictions.csv.gz",
                 e2022_new=pdir / f"track_b_https2022_eval_{args.e2022_new}" / "eval2022_predictions.csv.gz")
    d = {k: pd.read_csv(v) for k, v in files.items()}
    b1b, b1n = d["b1_base"][d["b1_base"].kind == "seq"], d["b1_new"][d["b1_new"].kind == KIND]
    _same_rows(b1b, b1n, ["pcap", "flow_id", "role"], "#39 적응")
    eb, en = d["e2022_base"][d["e2022_base"].kind == "seq"], d["e2022_new"][d["e2022_new"].kind == KIND]
    _same_rows(eb[eb.in_test == 1], en[en.in_test == 1], ["pcap", "flow_id", "actual"], "#40 2022")
    p39 = pair_metrics(b1_metrics(b1b).drop(columns="kind"), b1_metrics(b1n).drop(columns="kind"), "39_88_81")
    p40 = pair_metrics(metrics_2022(eb).drop(columns="kind"), metrics_2022(en).drop(columns="kind"), "40_2022")
    rows = pd.concat([p39[0], p40[0]], ignore_index=True)
    summ = pd.concat([p39[1], p40[1]], ignore_index=True)
    rows["recall_population"] = summ["recall_population"] = "all_attack(.88.81+.242.104)"  # 2022는 시험 공격 전체
    rows.loc[rows.experiment == "40_2022", "recall_population"] = "all_attack(2022)"
    summ.loc[summ.experiment == "40_2022", "recall_population"] = "all_attack(2022)"
    by_role = role_recall(b1b, b1n)
    out.mkdir(parents=True)
    rows.to_csv(out / "paired_conditions.csv", index=False)
    summ.to_csv(out / "paired_summary.csv", index=False)
    by_role.to_csv(out / "paired_recall_by_victim_39.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, **{k: dict(path=str(v.relative_to(DATA_DIR)), sha256=sha256_file(v)) for k, v in files.items()},
        code_sha256={"src/track_b/mask3.py": sha256_file(Path(__file__))}, big_delta=BIG_DELTA,
        note="저장 예측에서 계산(재학습 없음). 같은 시험 행임을 행 키로 확인. 양성=공격. 원본 N=0 등 같은 (k,seed,n) 짝"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))
    print("DONE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("https")
    h.add_argument("--flows-run-id", required=True)
    h.add_argument("--audit-run-id", required=True)
    h.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    h.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("compare")
    for name in ("b1-base", "b1-new", "e2022-base", "e2022-new"):
        c.add_argument(f"--{name}", required=True)
    c.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    https_main(args) if args.cmd == "https" else compare_main(args)


if __name__ == "__main__":
    main()
