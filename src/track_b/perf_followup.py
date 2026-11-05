"""RF 단건 판정 지연 미달 원인 분해와 경량화 후속 측정(안건 11 확정안). #42 주 수치(모델 구성 고정)와 분리한 진단이다.

변형(사전 고정): RF 트리 100(기준)·50·25 × 입력 DataFrame·numpy(각각 같은 데이터·시드로 따로 학습) + LightGBM 고정 기준. k=5·10.
플로우(#42와 같은 3개 10MiB 시료의 k 도달 전수)마다 변형별로 (a) 벡터 생성만, (b) 미리 만든 입력에 predict만, (d) 벡터 생성+predict를 직접 잰다.
(a)+(b)를 더해 전체로 해석하지 않고 전체 지연은 직접 잰 (d)만 쓴다. 변형마다 따로 워밍업하고 3패스를 돌리며 패스마다 변형 순서를 순환한다.
통과 기준(사전 고정): 3패스 모두 전체 지연 p95 <= 10ms. RF 예측 불일치는 같은 트리 수의 DataFrame/numpy 모델 사이에서만 센다.
C1 정확도: 2021 04-11 학습 -> 04-16 payload>0, 조건 A(정상 + .242.104 공격)와 B(정상 + 새 피해자 .88.81 공격)를 나눠 혼동 건수·macro-F1(변형 선택에 쓰지 않음).

Run: uv run python -m src.track_b.perf_followup --flows-run-id 2026-11-02_v1 --run-id <ID>
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.results import sha256_file
from src.track_b.https_flows import ATTACK, BENIGN
from src.track_b.https_unseen import NEW_VICTIM, run_environment
from src.track_b.pcap_flows import K_MAX, assemble_flows, prefix_input
from src.track_b.perf_measure import KS, LIMIT_MS, SAMPLES, SEQ_COLUMNS, WARMUP_RUNS, cut_pcap, pctl

PASSES, WARMUP_FLOWS = 3, 10
TREES = (100, 50, 25)
SEED = 0


def variants():
    """(이름, 종류, 트리 수, 입력형). 종류 rf|lgbm, 입력형 df|np. LightGBM 기준 패스는 DataFrame 입력 하나."""
    out = [(f"rf{t}_{kind}", "rf", t, kind) for t in TREES for kind in ("df", "np")]
    return out + [("lgbm_df", "lgbm", 100, "df")]


def build_input(flow, k, kind):
    row = prefix_input(flow, k)
    return pd.DataFrame([row], columns=SEQ_COLUMNS) if kind == "df" else np.asarray([row], dtype=np.int64)


def time_variant(model, kind, flows, k):
    """플로우별 (벡터 생성 ms, 미리 만든 입력 predict ms, 직접 잰 전체 ms). 변형 워밍업을 앞에서 따로 한다."""
    for f in flows[:WARMUP_FLOWS]:
        model.predict(build_input(f, k, kind))
    rows = []
    for f in flows:
        t0 = time.perf_counter_ns()
        x = build_input(f, k, kind)
        t1 = time.perf_counter_ns()
        model.predict(x)
        t2 = time.perf_counter_ns()
        model.predict(build_input(f, k, kind))
        t3 = time.perf_counter_ns()
        rows.append(((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6))
    return np.asarray(rows)


def summarize(ms):
    return dict(median_ms=pctl(ms, 50), p95_ms=pctl(ms, 95), p99_ms=pctl(ms, 99), max_ms=float(ms.max()), over_10ms=int((ms > LIMIT_MS).sum()))


def fit_models(train, k):
    from src.common.models import make_model

    x = pd.DataFrame(np.where(np.arange(K_MAX)[None, :] < k, train[SEQ_COLUMNS].to_numpy(), 0), columns=SEQ_COLUMNS)
    models = {}
    for name, kind, trees, inp in variants():
        m = make_model(kind, seed=SEED, workers=2)
        m.set_params(n_estimators=trees)
        models[name] = m.fit(x if inp == "df" else x.to_numpy(), train.label.to_numpy())
    return models, x


def accuracy_rows(models, test, k):
    """조건 A(정상+.242.104 공격), B(정상+.88.81 공격)별 혼동 건수·macro-F1. DataFrame 모델만 쓴다(numpy 모델과 예측 일치는 따로 확인)."""
    x = pd.DataFrame(np.where(np.arange(K_MAX)[None, :] < k, test[SEQ_COLUMNS].to_numpy(), 0), columns=SEQ_COLUMNS)
    rows = []
    for name, kind, trees, inp in variants():
        if inp != "df":
            continue
        pred = models[name].predict(x) == ATTACK
        for cond, mask in (("A_242_104", (test.label == BENIGN) | ((test.label == ATTACK) & (test.server_ip != NEW_VICTIM))),
                           ("B_88_81", (test.label == BENIGN) | ((test.label == ATTACK) & (test.server_ip == NEW_VICTIM)))):
            m = mask.to_numpy()
            truth, p = (test.label == ATTACK).to_numpy()[m], pred[m]
            tp, tn, fp, fn = int((truth & p).sum()), int((~truth & ~p).sum()), int((~truth & p).sum()), int((truth & ~p).sum())
            f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
            rows.append(dict(variant=name, k=k, condition=cond, rows=int(m.sum()), tp=tp, tn=tn, fp=fp, fn=fn,
                             macro_f1=(f1(tp, fp, fn) + f1(tn, fn, fp)) / 2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_perf_followup_{args.run_id}"
    rows_dir = DATA_DIR / "perf_runs" / f"track_b_perf_followup_{args.run_id}"
    if out.exists() or rows_dir.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows_csv = DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"
    f = pd.read_csv(flows_csv, usecols=["day", "label", "server_ip", "payload_packets"] + SEQ_COLUMNS, keep_default_na=False)
    f = f[f.payload_packets > 0]
    train, test = f[f.day == "2021-04-11"], f[f.day == "2021-04-16"]
    sample_dir = DATA_DIR / "perf_samples"
    pool = []
    for name in SAMPLES:
        cut = sample_dir / f"{Path(name).stem}_10MiB.pcap"
        if not cut.exists():
            cut_pcap(DATA_DIR / "raw" / "hikari" / name, cut)
        flows, _, _ = assemble_flows(cut)
        pool.extend((name, fl) for fl in flows if fl.payload_packets > 0)
    timing, summary, mismatch, accuracy = [], [], [], []
    for k in KS:
        models, _ = fit_models(train, k)
        pick = [(s, fl) for s, fl in pool if fl.payload_packets >= k]
        flows_k = [fl for _, fl in pick]
        order = [v[0] for v in variants()]
        for p in range(PASSES):
            rot = order[p % len(order):] + order[:p % len(order)]  # 패스마다 변형 실행 순서 순환
            for name in rot:
                inp = name.rsplit("_", 1)[1]
                ms = time_variant(models[name], inp, flows_k, k)
                timing.extend(dict(k=k, variant=name, run_pass=p, sample=s, client_ip=fl.client_ip, client_port=fl.client_port,
                                   server_ip=fl.server_ip, start_ts=round(fl.start_ts, 6), vector_ms=a, predict_ms=b, total_ms=c)
                              for (s, fl), (a, b, c) in zip(pick, ms))
                for stage, col in (("vector", 0), ("predict_only", 1), ("total", 2)):
                    summary.append(dict(k=k, variant=name, run_pass=p, stage=stage, n=len(flows_k), **summarize(ms[:, col])))
        # RF 예측 불일치: 같은 트리 수의 DataFrame/numpy 모델 사이(플로우 전체에서 일괄 predict)
        xs = pd.DataFrame(np.where(np.arange(K_MAX)[None, :] < k, test[SEQ_COLUMNS].to_numpy(), 0), columns=SEQ_COLUMNS)
        for t in TREES:
            a = models[f"rf{t}_df"].predict(xs)
            b = models[f"rf{t}_np"].predict(xs.to_numpy())
            mismatch.append(dict(k=k, trees=t, rows=len(xs), mismatches=int((a != b).sum())))
        accuracy.extend(accuracy_rows(models, test, k))
    timing_df, summary_df = pd.DataFrame(timing), pd.DataFrame(summary)
    tot = summary_df[summary_df.stage == "total"]
    verdict = (tot.groupby(["k", "variant"]).p95_ms.agg(max_p95_ms="max", passes_over_10ms=lambda s: int((s > LIMIT_MS).sum())).reset_index()
               .assign(pass_all_3=lambda d: d.passes_over_10ms == 0))
    out.mkdir(parents=True)
    rows_dir.mkdir(parents=True)
    timing_df.to_csv(rows_dir / "timing_rows.csv.gz", index=False, float_format="%.6g")
    summary_df.to_csv(out / "summary_by_pass.csv", index=False)
    verdict.to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(mismatch).to_csv(out / "rf_prediction_mismatch.csv", index=False)
    pd.DataFrame(accuracy).to_csv(out / "c1_accuracy.csv", index=False)
    env = run_environment(root)
    env.update(cpu=platform.processor(), logical_cpu_count=__import__("os").cpu_count(), single_predict_n_jobs_setting=1)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, passes=PASSES, warmup_flows=WARMUP_FLOWS, trees=list(TREES), ks=list(KS), seed=SEED,
        variants=[v[0] for v in variants()], limit_ms=LIMIT_MS, pass_rule="3패스 모두 전체(직접 측정) 지연 p95 <= 10ms",
        flows_per_k={str(k): int(sum(1 for _, fl in pool if fl.payload_packets >= k)) for k in KS}, environment=env,
        timing_rows_file=dict(path=f"perf_runs/track_b_perf_followup_{args.run_id}/timing_rows.csv.gz", sha256=sha256_file(rows_dir / "timing_rows.csv.gz")),
        flows_sha256=sha256_file(flows_csv), code_sha256={"src/track_b/perf_followup.py": sha256_file(Path(__file__))},
        note="#42 주 수치와 분리한 진단. 변형 실행 순서는 패스마다 순환, 벡터·추론 p95를 더해 전체로 해석하지 않음. C1 정확도는 같은 분포 쉬운 구분이며 "
             "변형 선택에 쓰지 않는다. 트리 수 축소는 앙상블 성질을 바꾸고 시드 1개 결과다"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(verdict.round(2).to_string(index=False))
    print(pd.DataFrame(mismatch).to_string(index=False))
    print(pd.DataFrame(accuracy).round(4).to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
