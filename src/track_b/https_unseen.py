"""처음 보는 공격(.88.81)을 저장 예측 점수로 진단한다 (단계 0, 학습 없음).

저장 예측(max_probability, predicted)에서 P(attack)을 복원해 .88.81 공격 대 04-16 정상을 AUC·점수 분포·동률 비율·
시험 점수 상한 recall@정상 오탐으로 본다. 상한은 시험 점수로 문턱을 골랐을 때의 값이라 **진단 전용**이며 탐지 성능이 아니다.
판정 기준(사전 고정, 안건 5): AUC>=0.9 이고 상한(5%)>=0.5 -> 문턱 문제(표현은 충분), AUC<=0.6 -> 표현 문제, 그 사이 보류.

Run: uv run python -m src.track_b.https_unseen diag --predictions <csv> [--predictions <csv> ...] --run-id <ID>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.results import sha256_file
from src.track_b.https_flows import ATTACK, BENIGN

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag")
    d.add_argument("--predictions", action="append", required=True)
    d.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
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
