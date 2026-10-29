"""원본 TLS 형식 변환·품질 격리·분할 배정. 학습/스케일링은 수행하지 않는다.

Run: uv run python -m src.track_a.preprocess
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


EXCLUDED = frozenset({
    "Shifu.A_None_TLS_CC", "Banker_None_TLS_CC", "Drixed_None_TLS_CC",
    "Vawtrak.C_None_TLS_CC", "golistmero_golistmero_TLS_scan",
})
HELD_OUT = frozenset({
    "nessus_nessus_TLS_scan", "arachni_arachni_TLS_scan", "Totbrick_None_TLS_CC",
    "Tiggre_None_TLS_CC", "TrickBotCC_None_TLS_CC",
})
HEX = ["tls_version", "tls_chosen_cipher_suit"] + [f"tls_cipher_suites_{i}" for i in range(5)]
STAT_NAMES = """flow_duration total_fwd_packets total_backward_packets total_packets
total_fwd_bytes total_backward_bytes total_bytes
fwd_packet_length_Max fwd_packet_length_Min fwd_packet_length_Mean fwd_packet_length_Std
fwd_packet_length_Var fwd_packet_length_Median fwd_packet_length_Most
backward_packet_length_Max backward_packet_length_Min backward_packet_length_Mean
backward_packet_length_Std backward_packet_length_Var backward_packet_length_Median backward_packet_length_Most
total_packets_sent_rate fwd_packets_sent_rate backward_packets_sent_rate
total_bytes_sent_rate fwd_bytes_sent_rate backward_bytes_sent_rate
fwd_arrtime_diff_Max fwd_arrtime_diff_Min fwd_arrtime_diff_Mean fwd_arrtime_diff_Std fwd_arrtime_diff_Var fwd_arrtime_diff_Median
backward_arrtime_diff_Max backward_arrtime_diff_Min backward_arrtime_diff_Mean backward_arrtime_diff_Std backward_arrtime_diff_Var backward_arrtime_diff_Median""".split()


def feature_groups():
    fingerprint = ["tls_version", "tls_cipher_suites_length", "tls_chosen_cipher_suit",
                   "tls_extensions_length"] + [f"tls_cipher_suites_{i}" for i in range(5)]
    stats = [f"{prefix}_{name}" for prefix in ("IP", "payload") for name in STAT_NAMES]
    sequence = [str(i) for i in range(30)]
    return dict(all=fingerprint + stats + sequence, fingerprint=fingerprint, stats=stats, sequence=sequence)


def tls_integer(value):
    if isinstance(value, str):
        token = value.strip()
        return int(token, 16) if token.lower().startswith("0x") else int(token)
    if not np.isfinite(value) or value != int(value):
        raise ValueError("TLS 식별자는 정수여야 합니다")
    return int(value)


def encode_frame(raw):
    """값의 표기만 변환한다. 행·중복·극단값은 삭제하지 않고 잘못된 형식은 중단한다."""
    cols = feature_groups()
    if raw.columns.duplicated().any() or set(raw.columns) != set(cols["all"] + ["Label"]):
        raise ValueError("원본은 특징 117개와 Label, 총 118개 컬럼이어야 합니다")
    f = raw.loc[:, cols["all"] + ["Label"]].copy().reset_index(drop=True)
    if f.isna().any().any() or not f.Label.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("결측값 또는 빈 라벨: 자동 대체하지 않고 원인을 확인해야 합니다")
    for col in cols["all"]:
        try:
            if col in HEX:
                f[col] = f[col].map(tls_integer)
                if not f[col].between(0, 65535).all():
                    raise ValueError("TLS 식별자 범위 초과")
            else:
                f[col] = pd.to_numeric(f[col], errors="raise")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"숫자 변환 실패: {col}") from exc
    values = f[cols["all"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("유한하지 않은 특징값: 시험 데이터까지 포함해 자동 대체하지 않습니다")
    seq = f[cols["sequence"]].to_numpy(dtype=np.float64)
    if not np.equal(seq, np.trunc(seq)).all() or np.abs(seq).max(initial=0) >= 2**53:
        raise ValueError("패킷 길이 시퀀스는 정확히 표현 가능한 정수여야 합니다")
    f[cols["sequence"]] = f[cols["sequence"]].astype(np.int64)
    return f


def row_roles(frame):
    """분포에 적합하지 않는 고정 범위 검사. 부호 있는 시퀀스는 검사에서 제외."""
    roles = pd.Series("known", index=frame.index)
    roles.loc[frame.Label.isin(HELD_OUT)] = "unknown"
    roles.loc[frame.Label.isin(EXCLUDED)] = "excluded"
    roles.loc[(frame[feature_groups()["stats"]] < 0).any(axis=1)] = "invalid"
    return roles


def pattern_ids(frame):
    """부호 있는 30개 길이의 little-endian int64 바이트를 SHA-256으로 식별."""
    sequences = frame[feature_groups()["sequence"]].to_numpy(dtype="<i8")
    ids = pd.Series([hashlib.sha256(row.tobytes()).hexdigest() for row in sequences], index=frame.index)
    if ids.nunique() != len(frame[feature_groups()["sequence"]].drop_duplicates()):
        raise ValueError("서로 다른 시퀀스의 패턴 ID 충돌")
    return ids


def group_split(labels, patterns, seed, minimum=30):
    groups, _ = pd.factorize(patterns, sort=True)
    ordered = np.random.default_rng(seed).permutation(groups.max() + 1)[groups]
    candidates = []
    for fold, (train, test) in enumerate(StratifiedGroupKFold(5, shuffle=False).split(
            np.zeros(len(labels)), labels, ordered)):
        if all(min(int((labels[train] == c).sum()), int((labels[test] == c).sum())) >= minimum
               for c in set(labels)):
            candidates.append((abs(len(test) / len(labels) - .2), fold, train, test))
    if not candidates:
        raise ValueError("클래스별 학습·시험 30행 조건을 만족하는 그룹 폴드가 없습니다")
    _, fold, train, test = min(candidates, key=lambda c: c[:2])
    return train, test, fold


def validate_assignment(frame, assignment, split, minimum=30):
    if split not in ("random", "pattern_group"):
        raise ValueError("알 수 없는 분할 종류")
    if not assignment.source_row.tolist() == list(range(len(frame))):
        raise ValueError("원본 행 번호가 중복·누락·재배열되었습니다")
    if not assignment.Label.equals(frame.Label) or not assignment.pattern_id.equals(pattern_ids(frame)):
        raise ValueError("라벨 또는 패턴 ID가 원본과 다릅니다")
    roles = row_roles(frame)
    for role in ("excluded", "unknown", "invalid"):
        if not assignment.loc[roles.eq(role), "partition"].eq(role).all():
            raise ValueError(f"격리 대상이 잘못 배정되었습니다: {role}")
    known = assignment.loc[roles.eq("known")]
    if not known.partition.isin(["train", "test"]).all():
        raise ValueError("알려진 클래스의 분할 누락")
    counts = pd.crosstab(known.Label, known.partition).reindex(columns=["train", "test"], fill_value=0)
    if counts.empty or (counts < minimum).any().any():
        raise ValueError("알려진 클래스의 학습·시험이 30행 미만입니다")
    if split == "pattern_group":
        train = set(assignment.loc[assignment.partition.eq("train"), "pattern_id"])
        evaluation = set(assignment.loc[assignment.partition.isin(["test", "unknown"]), "pattern_id"])
        if train & evaluation:
            raise ValueError("학습·평가의 시퀀스 패턴이 겹칩니다")


def build_assignment(frame, patterns, split, seed):
    roles = row_roles(frame)
    known_rows = np.flatnonzero(roles.eq("known"))
    if split == "pattern_group":
        if set(patterns[roles.eq("known")]) & set(patterns[roles.eq("unknown")]):
            raise ValueError("보류·알려진 클래스 패턴이 겹칩니다. 보류 후보 또는 격리 설계 검토 필요")
        train, test, fold = group_split(frame.Label.iloc[known_rows].to_numpy(), patterns.iloc[known_rows], seed)
    elif split == "random":
        train, test = train_test_split(np.arange(len(known_rows)), test_size=.2,
                                      stratify=frame.Label.iloc[known_rows], random_state=seed)
        fold = None
    else:
        raise ValueError("알 수 없는 분할 종류")
    assignment = pd.DataFrame(dict(source_row=np.arange(len(frame)), Label=frame.Label,
                                   pattern_id=patterns, partition=roles.copy()))
    assignment.loc[known_rows[train], "partition"] = "train"
    assignment.loc[known_rows[test], "partition"] = "test"
    validate_assignment(frame, assignment, split)
    return assignment, fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id는 영문·숫자·밑줄·하이픈만 허용합니다")
    from src.common.config import DATA_DIR, MACHINE

    source = DATA_DIR / "raw/malicious_tls/malicious_TLS.csv"
    raw = pd.read_csv(source, low_memory=False)
    frame = encode_frame(raw)
    roles, patterns = row_roles(frame), pattern_ids(frame)
    if frame.Label.nunique() != 23 or not (EXCLUDED | HELD_OUT) <= set(frame.Label):
        raise ValueError("예상한 23종 클래스 구성이 아닙니다")
    known_labels = sorted(frame.loc[roles.eq("known"), "Label"].unique())
    if len(known_labels) != 13 or "benign" not in known_labels:
        raise ValueError("알려진 13종 구성이 아닙니다")

    # Candidate selection audit only: hypothetical split of all 18 eligible classes.
    # These indices are never used for actual known-class training or preprocessing fit.
    eligible = np.flatnonzero(roles.isin(["known", "unknown"]))
    candidate_counts = []
    for seed in (0, 1, 2):
        tr, te, fold = group_split(frame.Label.iloc[eligible].to_numpy(), patterns.iloc[eligible], seed)
        for label in sorted(HELD_OUT):
            ntrain = int(frame.Label.iloc[eligible[tr]].eq(label).sum())
            ntest = int(frame.Label.iloc[eligible[te]].eq(label).sum())
            if min(ntrain, ntest) < 30:
                raise ValueError(f"보류 후보 분할 부적합: {label}")
            candidate_counts.append(dict(seed=seed, fold=fold, label=label, train_rows=ntrain, test_rows=ntest))

    manifests, counts, split_info = {}, [], []
    for split in ("random", "pattern_group"):
        for seed in (0, 1, 2):
            assignment, fold = build_assignment(frame, patterns, split, seed)
            manifests[f"{split}_{seed}"] = assignment.partition
            for (label, partition), group in assignment.groupby(["Label", "partition"]):
                counts.append(dict(split=split, seed=seed, label=label, partition=partition,
                                   rows=len(group), patterns=group.pattern_id.nunique()))
            overlap = len(set(patterns[assignment.partition.eq("train")]) &
                          set(patterns[assignment.partition.eq("test")]))
            split_info.append(dict(split=split, seed=seed, fold=fold,
                                   train_rows=int(assignment.partition.eq("train").sum()),
                                   test_rows=int(assignment.partition.eq("test").sum()), overlap_patterns=overlap))
            print(split_info[-1], flush=True)

    root = Path(__file__).resolve().parents[2]
    data_out = DATA_DIR / "processed" / f"track_a_{args.run_id}"
    report_out = root / "outputs/results" / f"preprocess_{args.run_id}"
    if data_out.exists() or report_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    data_out.mkdir(parents=True)
    report_out.mkdir(parents=True)
    frame.to_csv(data_out / "features.csv", index_label="source_row")
    registry = pd.DataFrame(dict(source_row=np.arange(len(frame)), Label=frame.Label, pattern_id=patterns, role=roles))
    registry.to_csv(data_out / "rows.csv", index=False)
    pd.DataFrame(dict(source_row=registry.source_row, **manifests)).to_csv(data_out / "splits.csv", index=False)
    invalid = registry.loc[roles.eq("invalid")].copy()
    invalid["reason"] = "negative_statistic"
    invalid.to_csv(data_out / "invalid_rows.csv", index=False)
    class_audit = registry.groupby(["Label", "role"]).agg(rows=("source_row", "size"), patterns=("pattern_id", "nunique")).reset_index()
    class_audit.to_csv(report_out / "class_audit.csv", index=False)
    pd.DataFrame(counts).to_csv(report_out / "split_class_counts.csv", index=False)
    pd.DataFrame(candidate_counts).to_csv(report_out / "held_out_candidate_check.csv", index=False)
    pd.DataFrame(split_info).to_csv(report_out / "split_summary.csv", index=False)
    metadata = dict(run_id=args.run_id, machine=MACHINE, source="raw/malicious_tls/malicious_TLS.csv",
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_rows=len(frame),
                    roles=roles.value_counts().to_dict(), full_duplicates=int(frame.duplicated().sum()),
                    zero_payload_duration=int(frame.payload_flow_duration.eq(0).sum()),
                    feature_groups=feature_groups(), known_labels=known_labels,
                    excluded=sorted(EXCLUDED), held_out=sorted(HELD_OUT),
                    unknown_known_shared_patterns=len(set(patterns[roles.eq("known")]) & set(patterns[roles.eq("unknown")])),
                    policy="fixed negative-statistics quarantine; retain duplicates, zero durations and extreme values; no fitted transforms",
                    group_split="seed-permuted global pattern IDs; SGKF(5,shuffle=False); valid >=30 rows/class/partition; closest overall test fraction to .2, ties by fold index; model scores unused",
                    validation="all six assignments pass; no holdout in train/test; group train patterns disjoint from known test and unknown; fixed 13 known classes",
                    data_files="features.csv (all raw rows; select via rows.csv and splits.csv), rows.csv, splits.csv, invalid_rows.csv; under DATA_DIR only",
                    pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    (report_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(class_audit.to_string(index=False), flush=True)
    print("PASS; data:", data_out, "reports:", report_out, flush=True)


if __name__ == "__main__":
    main()
