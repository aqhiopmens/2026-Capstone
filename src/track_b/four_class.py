"""4클래스(정상 / Probing / Bruteforce-XML / Bruteforce-HTTPS) PoC 평가와 '공격 하나 빼기' 경보 비율(안건 9 확정안).

분할(성능 확인 전 고정): 학습 = HTTPS 2021-04-11, XML 2021-04-12, Probing 캡처 앞 70%(시간), 시험 = HTTPS 04-16, XML 04-17, Probing 캡처 뒤 30%.
Probing은 짝 캡처가 없어 캡처 안 시간 블록으로 나누고(경계 T = 시작 + 0.7 × 구간), 학습은 종료 시각 <= T, 시험은 시작 >= T + 600초이며
경계를 가로지르거나 간격 안에 있는 흐름은 양쪽에서 제외한다. 정상은 각 캡처의 비피해자 443 플로우를 같은 규칙으로 나눈다(정상 출처 = 캡처).
입력은 원본 seq와 앞 3칸 마스킹(seq_mask3) 둘 다, k=5·10, RF·LightGBM, 시드 3. 모델별·클래스별 점수와 혼동 행렬, 시험 출처별 분모·recall을 저장한다.
공격 하나 빼기: 한 공격 유형을 학습에서 빼고(정상 + 나머지 두 유형으로 학습) 그 유형 시험 행을 '정상이 아니라고 판정한 비율(경보율)'로 본다. 4분류 재현율이 아니다.
PoC이며 캡처 단위 잠정 라벨, 쉬운 구분 단서를 유지한다. 시험 결과로 분할·k·모델을 바꾸지 않는다.

Run: uv run python -m src.track_b.four_class --flows-run-id 2026-11-02_v1 --xml-flows-run-id 2026-10-31_v3 --probing-run-id 2026-11-08_v1 --run-id <ID>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.common.results import sha256_file
from src.track_b.early_eval import mask3, prefix_features
from src.track_b.https_flows import ATTACK as HTTPS, BENIGN
from src.track_b.https_unseen import NEW_VICTIM, run_environment

XML, PROBING = "bruteforce_xml", "probing"
CLASSES = (BENIGN, PROBING, XML, HTTPS)
ATTACKS = (PROBING, XML, HTTPS)
KS, KINDS, MODELS, SEEDS = (5, 10), ("seq", "seq_mask3"), ("rf", "lgbm"), (0, 1, 2)
TRAIN_FRACTION, GAP_S = 0.7, 600.0
SEQ = [f"seq_{i}" for i in range(30)]
COLS = ["pcap", "day", "flow_id", "label", "client_ip", "server_ip", "start_ts", "duration", "payload_packets"] + SEQ


def probing_parts(frame, fraction=TRAIN_FRACTION, gap=GAP_S):
    """Probing 캡처 시간 블록 분할. 반환: part 열(train/test/excluded) Series와 경계 시각 T."""
    t0, t1 = float(frame.start_ts.min()), float((frame.start_ts + frame.duration).max())
    t = t0 + fraction * (t1 - t0)
    end = frame.start_ts + frame.duration
    part = np.where(end <= t, "train", np.where(frame.start_ts >= t + gap, "test", "excluded"))
    return pd.Series(part, index=frame.index), t


def assign_parts(https, xml, probing):
    """세 캡처 묶음을 하나의 DataFrame으로 합치고 part(train/test/excluded)·source 열을 붙인다. 모든 입력은 payload>0."""
    parts = []
    h = https.assign(part=np.select([https.day == "2021-04-11", https.day == "2021-04-16"], ["train", "test"], "excluded"),
                     source=np.where(https.day == "2021-04-11", "https_0411", "https_0416"))
    h["source"] = np.where((h.label == HTTPS) & (h.server_ip == NEW_VICTIM), "https_0416_new_victim", h.source)
    x = xml.assign(part=np.select([xml.day == "2021-04-12", xml.day == "2021-04-17"], ["train", "test"], "excluded"),
                   source=np.where(xml.day == "2021-04-12", "xml_0412", "xml_0417"))
    pp, t = probing_parts(probing)
    p = probing.assign(part=pp, source=np.where(pp == "train", "probing_train", "probing_test"))
    for f in (h, x, p):
        parts.append(f)
    return pd.concat(parts, ignore_index=True), t


def features(flows, k, kind):
    x = prefix_features(flows, k, "seq")
    return mask3(x) if kind == "seq_mask3" else x


def class_scores(actual, predicted, labels=CLASSES):
    p, r, f, s = precision_recall_fscore_support(actual, predicted, labels=list(labels), zero_division=0)
    return pd.DataFrame(dict(label=list(labels), precision=p, recall=r, f1=f, support=s)), float(np.mean(f[s > 0])) if (s > 0).any() else 0.0


def fit_predict(model_name, seed, x_train, y_train, x_test):
    from src.common.models import make_model

    return make_model(model_name, seed=seed, workers=2).fit(x_train, y_train).predict(x_test)


def run(data, report_rows, detail_rows, confusion_rows, alert_rows, holdout_rows=None):
    train, test = data[data.part == "train"], data[data.part == "test"]
    for kind in KINDS:
        for k in KS:
            xtr, xte = features(train, k, kind), features(test, k, kind)
            for model in MODELS:
                for seed in SEEDS:
                    pred = fit_predict(model, seed, xtr, train.label.to_numpy(), xte)
                    scores, macro = class_scores(test.label, pred)
                    cond = dict(kind=kind, k=k, model=model, seed=seed)
                    report_rows.append(dict(**cond, macro_f1=macro, rows=len(test), **{f"f1_{r.label}": r.f1 for r in scores.itertuples()},
                                            **{f"recall_{r.label}": r.recall for r in scores.itertuples()}))
                    cm = confusion_matrix(test.label, pred, labels=list(CLASSES))
                    confusion_rows.extend(dict(**cond, actual=a, predicted=b, count=int(cm[i, j])) for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES))
                    detail_rows.append(pd.DataFrame(dict(pcap=test.pcap.to_numpy(), flow_id=test.flow_id.to_numpy(), source=test.source.to_numpy(),
                                                          actual=test.label.to_numpy(), predicted=pred, reached_k=(test.payload_packets >= k).to_numpy())).assign(**cond))
                    for held in ATTACKS:  # 공격 하나 빼기: 학습에서 해당 유형을 빼고 시험 행의 경보율(정상이 아니라고 판정한 비율)을 본다
                        keep = train.label != held
                        pred_h = fit_predict(model, seed, xtr[keep.to_numpy()], train.label[keep].to_numpy(), xte)
                        for label, m in ((held, test.label == held), (BENIGN, test.label == BENIGN)):
                            alert_rows.append(dict(**cond, held_out=held, test_label=label, rows=int(m.sum()), alerts=int((pred_h[m.to_numpy()] != BENIGN).sum()),
                                                   alert_rate=float((pred_h[m.to_numpy()] != BENIGN).mean())))
                        if holdout_rows is not None:  # 경보율 독립 재계산용: 빼낸 유형 행과 정상 행의 행별 예측
                            sel = ((test.label == held) | (test.label == BENIGN)).to_numpy()
                            holdout_rows.append(pd.DataFrame(dict(pcap=test.pcap.to_numpy()[sel], flow_id=test.flow_id.to_numpy()[sel], actual=test.label.to_numpy()[sel],
                                                                  predicted=pred_h[sel])).assign(**cond, held_out=held))


def by_source_metrics(pred):
    """저장 예측에서 시험 출처별·실제 클래스별 recall과 분모, 그리고 새 피해자(.88.81) 행을 뺀 4분류 macro-F1(C1 같은 분포 조건)을 계산한다(재학습 없음)."""
    rows, c1 = [], []
    for (kind, k, model, seed), g in pred.groupby(["kind", "k", "model", "seed"]):
        for (source, actual), h in g.groupby(["source", "actual"]):
            rows.append(dict(kind=kind, k=k, model=model, seed=seed, source=source, actual=actual, rows=len(h), recall=float((h.predicted == h.actual).mean())))
        sub = g[g.source != "https_0416_new_victim"]
        c1.append(dict(kind=kind, k=k, model=model, seed=seed, rows=len(sub), macro_f1_without_new_victim=class_scores(sub.actual, sub.predicted)[1]))
    return pd.DataFrame(rows), pd.DataFrame(c1)


def posthoc_main(run_id):
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_four_class_{run_id}"
    path = DATA_DIR / "predictions" / f"track_b_four_class_{run_id}" / "predictions.csv.gz"
    if (out / "by_source.csv").exists():
        raise FileExistsError("이미 후속 집계가 있습니다")
    src, c1 = by_source_metrics(pd.read_csv(path, keep_default_na=False))
    src.groupby(["kind", "k", "model", "source", "actual"]).agg(rows=("rows", "first"), recall=("recall", "mean")).reset_index().to_csv(out / "by_source.csv", index=False)
    c1.groupby(["kind", "k", "model"]).agg(rows=("rows", "first"), macro_f1_without_new_victim=("macro_f1_without_new_victim", "mean")).reset_index().to_csv(out / "macro_f1_without_new_victim.csv", index=False)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    meta["posthoc"] = dict(files=["by_source.csv", "macro_f1_without_new_victim.csv"], predictions_sha256=sha256_file(path), code_sha256=sha256_file(Path(__file__)),
                           note="재학습 없이 저장 예측에서 계산. macro_f1_without_new_victim은 새 피해자 .88.81 공격 행을 뺀 시험(C1 같은 분포)")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posthoc", help="기존 run-id의 저장 예측에서 출처별 recall·새 피해자 제외 macro-F1을 계산(재학습 없음)")
    parser.add_argument("--flows-run-id")
    parser.add_argument("--xml-flows-run-id")
    parser.add_argument("--probing-run-id")
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    if args.posthoc:
        return posthoc_main(args.posthoc)
    if not (args.flows_run_id and args.xml_flows_run_id and args.probing_run_id):
        parser.error("--flows-run-id, --xml-flows-run-id, --probing-run-id가 필요합니다")
    from src.common.config import DATA_DIR

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/results" / f"track_b_four_class_{args.run_id}"
    data_out = DATA_DIR / "predictions" / f"track_b_four_class_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    paths = dict(https=DATA_DIR / "processed" / f"track_b_https_{args.flows_run_id}" / "flows.csv",
                 xml=DATA_DIR / "processed" / f"track_b_{args.xml_flows_run_id}" / "flows.csv",
                 probing=DATA_DIR / "processed" / f"track_b_https_probing_{args.probing_run_id}" / "flows.csv")
    load = lambda p: pd.read_csv(p, usecols=COLS, keep_default_na=False).query("payload_packets > 0")
    https, xml, probing = (load(paths[n]) for n in ("https", "xml", "probing"))
    xml = xml.assign(label=xml.label.replace({"bruteforce_xml": XML}))
    data, boundary = assign_parts(https, xml, probing)
    data = data.reset_index(drop=True)
    support = data.groupby(["part", "source", "label"]).size().rename("rows").reset_index()
    reach = data[data.part != "excluded"].groupby(["part", "label"]).apply(
        lambda g: pd.Series({f"reached_k{k}": float((g.payload_packets >= k).mean()) for k in KS} | {"rows": len(g)}), include_groups=False).reset_index()
    report, detail, confusion, alert, holdout = [], [], [], [], []
    run(data, report, detail, confusion, alert, holdout)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    report_df, alert_df = pd.DataFrame(report), pd.DataFrame(alert)
    support.to_csv(out / "support.csv", index=False)
    reach.to_csv(out / "reach_by_class.csv", index=False)
    report_df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(out / "confusion.csv", index=False)
    alert_df.to_csv(out / "hold_out_alert.csv", index=False)
    pd.concat(detail, ignore_index=True).to_csv(data_out / "predictions.csv.gz", index=False)
    pd.concat(holdout, ignore_index=True).to_csv(data_out / "holdout_predictions.csv.gz", index=False)
    summary = report_df.groupby(["kind", "k", "model"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    summary.to_csv(out / "summary.csv", index=False)
    alert_df.groupby(["kind", "k", "model", "held_out", "test_label"]).agg(rows=("rows", "first"), alert_rate=("alert_rate", "mean")).reset_index().to_csv(out / "hold_out_summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(dict(
        run_id=args.run_id, classes=list(CLASSES), ks=list(KS), kinds=list(KINDS), models=list(MODELS), seeds=list(SEEDS), train_fraction=TRAIN_FRACTION, gap_s=GAP_S,
        probing_boundary_ts=boundary, inputs={n: dict(path=str(p.relative_to(DATA_DIR)), sha256=sha256_file(p)) for n, p in paths.items()},
        predictions_sha256=sha256_file(data_out / "predictions.csv.gz"), holdout_predictions_sha256=sha256_file(data_out / "holdout_predictions.csv.gz"), code_sha256={"src/track_b/four_class.py": sha256_file(Path(__file__))},
        environment=run_environment(root),
        note="PoC. 캡처 단위 잠정 라벨(Probing은 CSV로 확인되지 않음, docs 참조), 정상 출처=캡처, 쉬운 구분 단서. 공격 하나 빼기는 4분류 재현율이 아니라 경보율"),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(support.to_string(index=False))
    print(summary[["kind", "k", "model", "macro_f1"] + [f"recall_{c}" for c in CLASSES]].round(4).to_string(index=False))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
