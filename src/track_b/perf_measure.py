"""계획서 성능 목표 측정(안건 7 확정안): 10MiB pcap 처리 시간과 플로우 1건 판정 지연. 정확도는 재지 않는다.

목표 1: 기존 pcap을 패킷 경계에서 10MiB 이상이 되는 지점까지 잘라 쓰고, 조합마다 [파일 읽기 시작→조립→payload 필터→특징 생성→predict 종료] 연속 타이머의
벽시계 시간을 워밍업 1회 뒤 5회 잰다. 판정은 시료별 5회 중 최댓값 <= 30초(중앙값은 참고).
목표 2: 앞 k개 페이로드 패킷이 도달한 플로우 하나의 [특징 벡터 생성 + 단건 predict](n_jobs=1) 시간. 도달 대기·pcap 파싱은 제외하고 별도 보고.
플로우 1,000개를 고정 시드로 뽑고(도달 플로우가 적으면 전수) 워밍업 10회 뒤 중앙값·p95·p99·최댓값·10ms 초과 건수를 낸다. 판정은 p95 <= 10ms.
모델 로드·학습(2021 04-11 HTTPS, 시드 0)은 측정 구간 밖이다. 시간은 환경 의존이라 실행 환경을 같이 기록한다.

Run: uv run python -m src.track_b.perf_measure --flows-run-id 2026-11-02_v1 --run-id <ID>
"""

import argparse
import json
import platform
import struct
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.results import sha256_file
from src.track_b.https_flows import ATTACK
from src.track_b.https_unseen import run_environment
from src.track_b.pcap_flows import K_MAX, assemble_flows, prefix_input

MIB10 = 10 * 1024 * 1024
KS = (5, 10)
MODELS = ("rf", "lgbm")
SAMPLES = ("Sunday_2021-04-11_2154.pcap", "Friday_2021-04-16_2304.pcap", "Sunday_2022-04-10_2335_BRUTEFORCE_HTTPS_150s.pcap")
RUNS, WARMUP_RUNS, SINGLE_N, SINGLE_WARMUP, SINGLE_SEED = 5, 1, 1000, 10, 0
LIMIT_S, LIMIT_MS = 30.0, 10.0
SEQ_COLUMNS = [f"seq_{i}" for i in range(K_MAX)]


def cut_pcap(src, dst, min_bytes=MIB10):
    """classic libpcap을 패킷 경계에서 자른다. 파일 크기가 min_bytes 이상이 되는 첫 패킷까지 포함. (바이트, 패킷 수)를 반환."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        header = fin.read(24)
        fout.write(header)
        size, packets = 24, 0
        while size < min_bytes:
            rec = fin.read(16)
            if len(rec) < 16:
                break
            incl = struct.unpack_from("<I", rec, 8)[0]
            data = fin.read(incl)
            if len(data) < incl:
                break
            fout.write(rec + data)
            size += 16 + incl
            packets += 1
    return size, packets


def pctl(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def vector_frame(flows, k):
    return pd.DataFrame([prefix_input(f, k) for f in flows], columns=SEQ_COLUMNS)


def single_latency(model, flows, k, warmup=SINGLE_WARMUP):
    """플로우별 [특징 벡터 생성 + 단건 predict] 시간(ms) 배열."""
    for f in flows[:warmup]:
        model.predict(vector_frame([f], k))
    out = []
    for f in flows:
        t0 = time.perf_counter_ns()
        model.predict(vector_frame([f], k))
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return np.asarray(out)


def train_models(flows_csv, seed=0):
    from src.common.models import make_model

    f = pd.read_csv(flows_csv, usecols=["day", "label", "payload_packets"] + SEQ_COLUMNS, keep_default_na=False)
    f = f[(f.day == "2021-04-11") & (f.payload_packets > 0)]
    models = {}
    for name in MODELS:
        for k in KS:
            x = pd.DataFrame(np.where(np.arange(K_MAX)[None, :] < k, f[SEQ_COLUMNS].to_numpy(), 0), columns=SEQ_COLUMNS)
            models[(name, k)] = make_model(name, seed=seed, workers=2).fit(x, f.label.to_numpy())
    return models, len(f), f.label.value_counts().to_dict()


def measure_sample(path, models, runs=RUNS, warmup=WARMUP_RUNS):
    """조합(모델·k·n_jobs)마다 [파일 읽기 시작 -> 플로우 조립 -> payload 필터 -> 특징 생성 -> predict 종료]를 하나의 연속 타이머로 잰다(t_total_s).

    단계별 시간은 같은 실행 안의 체크포인트 차이이며 진단용이다. 조립을 조합마다 새로 하므로 단계 사이 비용이 총 시간에 포함된다.
    """
    rows = []
    for run in range(warmup + runs):
        for (name, k), model in models.items():
            for jobs in (1, -1):
                model.set_params(n_jobs=jobs)
                t0 = time.perf_counter()
                flows, skipped, tcp_packets = assemble_flows(path)
                t1 = time.perf_counter()
                flows = [f for f in flows if f.payload_packets > 0]
                x = pd.DataFrame([prefix_input(f, k) for f in flows], columns=SEQ_COLUMNS)
                t2 = time.perf_counter()
                model.predict(x)
                t3 = time.perf_counter()
                rows.append(dict(run=run, warmup=int(run < warmup), model=name, k=k, n_jobs=jobs, flows=len(flows),
                                 t_assemble_s=t1 - t0, t_filter_features_s=t2 - t1, t_predict_s=t3 - t2, t_total_s=t3 - t0))
    return pd.DataFrame(rows), flows, tcp_packets


def count_records(path):
    """classic libpcap 파일의 패킷 레코드 수(TCP 외 포함)."""
    n = 0
    with open(path, "rb") as f:
        f.read(24)
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                return n
            f.seek(struct.unpack_from("<I", rec, 8)[0], 1)
            n += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_perf_{args.run_id}"
    sample_dir = DATA_DIR / "perf_samples"
    if out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    flows_csv = DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv"
    models, train_rows, train_labels = train_models(flows_csv)
    sample_dir.mkdir(exist_ok=True)
    stats, runs_all, pool, wait = [], [], {k: [] for k in KS}, []
    for name in SAMPLES:
        cut = sample_dir / f"{Path(name).stem}_10MiB.pcap"
        if not cut.exists():
            cut_pcap(DATA_DIR / "raw" / "hikari" / name, cut)
        runs, flows, tcp_packets = measure_sample(cut, models)
        all_flows, skipped, _ = assemble_flows(cut)
        stats.append(dict(sample=name, bytes=cut.stat().st_size, sha256=sha256_file(cut), packet_records=count_records(cut),
                          tcp_packets_processed=tcp_packets, flows_payload_gt0=len(flows), flows_all_443=len(all_flows),
                          first_observed_not_client_syn=sum(f.midstream for f in all_flows),
                          no_fin_rst_in_sample=sum((not f.closed) for f in all_flows), skipped=json.dumps(skipped)))
        runs_all.append(runs.assign(sample=name))
        for k in KS:
            reached = [f for f in flows if f.payload_packets >= k]
            pool[k].extend((name, f) for f in reached)
            stats[-1][f"flows_reached_k{k}"] = len(reached)
            stats[-1][f"flows_short_k{k}"] = len(flows) - len(reached)
    runs_df = pd.concat(runs_all, ignore_index=True)
    meas = runs_df[runs_df.warmup == 0]
    total = (meas.groupby(["sample", "model", "k", "n_jobs"]).t_total_s.agg(median="median", max="max").reset_index()
             .assign(pass_30s_max=lambda d: d["max"] <= LIMIT_S))
    rng = np.random.default_rng(SINGLE_SEED)
    single, single_rows = [], []
    for k in KS:
        flows_k = pool[k]
        pick_pairs = [flows_k[i] for i in (rng.permutation(len(flows_k))[:SINGLE_N] if len(flows_k) > SINGLE_N else range(len(flows_k)))]
        pick = [f for _, f in pick_pairs]
        wait = np.array([f.times[k - 1] for f in pick])
        for (name, kk), model in models.items():
            if kk != k:
                continue
            model.set_params(n_jobs=1)
            ms = single_latency(model, pick, k)
            single_rows.extend(dict(model=name, k=k, sample=sm, client_ip=f.client_ip, client_port=f.client_port, server_ip=f.server_ip,
                                    start_ts=round(f.start_ts, 6), single_ms=float(m), wait_until_kth_packet_s=float(w))
                               for (sm, f), m, w in zip(pick_pairs, ms, wait))
            single.append(dict(model=name, k=k, n=len(pick), median_ms=pctl(ms, 50), p95_ms=pctl(ms, 95), p99_ms=pctl(ms, 99),
                               max_ms=float(ms.max()), over_10ms=int((ms > LIMIT_MS).sum()), pass_p95=pctl(ms, 95) <= LIMIT_MS,
                               wait_until_kth_packet_median_s=float(np.median(wait)), wait_until_kth_packet_p95_s=pctl(wait, 95)))
    out.mkdir(parents=True)
    pd.DataFrame(stats).to_csv(out / "samples.csv", index=False)
    runs_df.to_csv(out / "runs.csv", index=False)
    total.to_csv(out / "total_time.csv", index=False)
    pd.DataFrame(single).to_csv(out / "single_flow.csv", index=False)
    rows_dir = DATA_DIR / "perf_runs" / f"track_b_perf_{args.run_id}"  # 행별 측정 자료는 Git 밖(DATA_DIR/Drive)
    rows_dir.mkdir(parents=True)
    pd.DataFrame(single_rows).to_csv(rows_dir / "single_flow_rows.csv.gz", index=False, float_format="%.9g")
    env = run_environment(root)
    env.update(cpu=platform.processor(), logical_cpu_count=__import__("os").cpu_count(), single_predict_n_jobs_setting=1, batch_predict_n_jobs_settings=[1, -1],
               train_rows=train_rows, train_label_counts=train_labels)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, flows_run_id=args.flows_run_id, runs=RUNS, warmup_runs=WARMUP_RUNS, single_n=SINGLE_N, single_warmup=SINGLE_WARMUP,
        single_seed=SINGLE_SEED, limit_total_s=LIMIT_S, limit_single_p95_ms=LIMIT_MS, ks=list(KS), models=list(MODELS), environment=env,
        single_rows_file=dict(path=f"perf_runs/track_b_perf_{args.run_id}/single_flow_rows.csv.gz", sha256=sha256_file(rows_dir / "single_flow_rows.csv.gz")),
        flows_sha256=sha256_file(flows_csv), code_sha256={"src/track_b/perf_measure.py": sha256_file(Path(__file__)),
                                                         "src/track_b/pcap_flows.py": sha256_file(root / "src/track_b/pcap_flows.py")},
        note="총 시간 = 조합마다 [파일 읽기 시작→조립→payload 필터→특징 생성→predict 종료] 연속 타이머. 모델 로드·학습은 측정 구간 밖. 판정 지연 = 앞 k개 도달 후 [특징 생성+단건 predict]; 도달 대기·pcap 파싱은 별도. "
             "시료는 3개 pcap 앞부분 10MiB라 대표성이 제한적이며 서비스 지연(대시보드·네트워크)은 측정하지 않는다. 환경 의존 값"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(stats)[["sample", "bytes", "packet_records", "tcp_packets_processed", "flows_payload_gt0", "first_observed_not_client_syn", "no_fin_rst_in_sample"]].to_string(index=False))
    print(total.round(3).to_string(index=False))
    print(pd.DataFrame(single).round(3).to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
