"""학습과 분리한 최대확률 문턱·확률 계약·미확인 판정 평가."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

UNKNOWN = "__unknown__"
PERCENTS = (1, 5, 10)
METRICS = ("macro_f1", "accuracy", "known_reject_rate", "unknown_reject_rate",
           "unknown_class_macro_reject_rate", "benign_reject_rate", "benign_wrong_known_rate",
           "accepted_known_accuracy", "known_coverage", "unknown_pattern_macro_reject_rate")


def probability_columns(labels):
    return [f"prob_{i:02d}" for i in range(len(labels))]


def validate_saved(predictions, assignments, labels, partitions, known):
    columns = probability_columns(known)
    required = {"row_id", "partition", "actual", "predicted", "max_probability", *columns}
    if (set(predictions.columns) != required or set(assignments.columns) != {"row_id", "partition"}
            or UNKNOWN in set(labels) or len(known) < 2 or len(set(known)) != len(known)):
        raise ValueError("확률 저장 계약/예약 라벨 오류")
    if (assignments.row_id.tolist() != labels.index.tolist()
            or assignments.partition.tolist() != partitions.tolist()):
        raise ValueError("원본 배정 불일치")
    expected = partitions.index[partitions.isin(["validation", "test", "unknown"])]
    if predictions.row_id.duplicated().any() or set(predictions.row_id) != set(expected):
        raise ValueError("예측 행 누락·중복·평가 외 행")
    ordered = predictions.set_index("row_id").loc[expected]
    if (ordered.actual.tolist() != labels.loc[expected].tolist()
            or ordered.partition.tolist() != partitions.loc[expected].tolist()):
        raise ValueError("라벨 또는 분할 불일치")
    probabilities = ordered[columns].to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("확률 벡터 범위/합 오류")
    if (not np.isfinite(ordered.max_probability).all()
            or not np.array_equal(probabilities.max(1), ordered.max_probability.to_numpy())
            or not np.array_equal(np.asarray(known)[probabilities.argmax(1)], ordered.predicted.to_numpy())):
        raise ValueError("확률 max/argmax 불일치")
    return ordered.rename_axis("row_id").reset_index()


def select_threshold(validation, percent):
    """validation만 입력. strict <, 정수 거절 예산을 넘지 않는 최대 관측 문턱."""
    if (type(percent) is not int or percent not in PERCENTS or validation.empty
            or not validation.partition.eq("validation").all()):
        raise ValueError("검증셋과 사전 고정 거절률이 필요합니다")
    values = validation.max_probability.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("검증 확률 오류")
    unique, counts = np.unique(values, return_counts=True)
    before = np.r_[0, np.cumsum(counts)[:-1]]
    budget = len(values) * percent // 100
    index = np.flatnonzero(before <= budget)[-1]
    tau = float(unique[index])
    rejected = int((values < tau).sum())
    return dict(policy_id=f"reject_{percent:02d}", target_reject_rate=percent / 100,
                threshold=tau, validation_rows=len(values), allowed_reject_rows=budget,
                validation_reject_rows=rejected, validation_reject_rate=rejected / len(values),
                boundary_tie_rows=int(counts[index]), unique_scores=len(unique))


def policies_for(predictions):
    validation = predictions.loc[predictions.partition.eq("validation")]
    selected = [select_threshold(validation, percent) for percent in PERCENTS]
    none = dict(policy_id="none", target_reject_rate=np.nan, threshold=0.,
                validation_rows=len(validation), allowed_reject_rows=np.nan,
                validation_reject_rows=0, validation_reject_rate=0.,
                boundary_tie_rows=int(validation.max_probability.eq(0).sum()),
                unique_scores=validation.max_probability.nunique())
    return selected, [none, *selected]


def partition_audit(labels, partitions, patterns, identity):
    train = set(patterns.loc[partitions.eq("train")])
    records = []
    for part in ("train", "validation", "test", "unknown"):
        selected = partitions.eq(part)
        for label in ("__all__", *sorted(labels.loc[selected].unique())):
            mask = selected if label == "__all__" else selected & labels.eq(label)
            local = patterns.loc[mask]
            n = len(local)
            seen = int(local.isin(train).sum())
            records.append(dict(**identity, partition=part, label=label, rows=n,
                unique_patterns=local.nunique(), train_patterns=len(train),
                shared_patterns=len(set(local) & train), train_pattern_seen_rows=seen,
                train_pattern_seen_fraction=seen / n if n else np.nan))
    return pd.DataFrame(records)


def score_audit(predictions, identity):
    records = []
    for part, group in predictions.groupby("partition", sort=False):
        values = group.max_probability.to_numpy()
        quantiles = np.quantile(values, [.01, .05, .5, .95, .99], method="linear")
        records.append(dict(**identity, partition=part, rows=len(group), unique_scores=len(np.unique(values)),
            minimum=float(values.min()), maximum=float(values.max()), fraction_one=float((values == 1).mean()),
            **dict(zip(("q01", "q05", "q50", "q95", "q99"), quantiles)), quantile_method="linear; audit only"))
    return pd.DataFrame(records)


def evaluate_policy(predictions, known, patterns, policy, identity):
    evaluation = predictions.loc[predictions.partition.isin(["test", "unknown"])].copy()
    if evaluation.empty or not {"test", "unknown"} <= set(evaluation.partition):
        raise ValueError("알려진 시험과 보류가 모두 필요합니다")
    rejected = evaluation.max_probability.to_numpy() < policy["threshold"]
    is_known = evaluation.partition.eq("test").to_numpy()
    actual = np.where(is_known, evaluation.actual, UNKNOWN)
    predicted = np.where(rejected, UNKNOWN, evaluation.predicted)
    open_labels = [*known, UNKNOWN]
    matrix = confusion_matrix(actual, predicted, labels=open_labels)
    support, predicted_support, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    f1 = np.divide(2 * tp, support + predicted_support, out=np.zeros(len(open_labels)),
                   where=(support + predicted_support) > 0)
    key = dict(**identity, policy_id=policy["policy_id"], target_reject_rate=policy["target_reject_rate"],
               threshold=policy["threshold"])
    class_rows = []
    for part in ("test", "unknown"):
        part_mask = evaluation.partition.eq(part).to_numpy()
        for label in sorted(evaluation.loc[part_mask, "actual"].unique()):
            mask = part_mask & evaluation.actual.eq(label).to_numpy()
            n, count = int(mask.sum()), int(rejected[mask].sum())
            class_rows.append(dict(**key, partition=part, label=label, rows=n, reject_rows=count,
                                   reject_rate=count / n if n else np.nan, low_support=0 < n < 30))
    classes = pd.DataFrame(class_rows)
    unknown = evaluation.loc[~is_known, ["row_id", "actual"]].copy()
    unknown["pattern_id"] = patterns.loc[unknown.row_id].to_numpy()
    unknown["rejected"] = rejected[~is_known].astype(int)
    pattern_rows = (unknown.groupby(["actual", "pattern_id"], sort=True).rejected.agg(["size", "sum"])
                    .reset_index().rename(columns={"actual": "label", "size": "rows", "sum": "reject_rows"}))
    pattern_rows["reject_rate"] = pattern_rows.reject_rows / pattern_rows.rows
    per_class = pattern_rows.groupby("label", sort=True).agg(patterns=("pattern_id", "size"),
        rows=("rows", "sum"), pattern_equal_weight_reject_rate=("reject_rate", "mean")).reset_index()
    def with_key(table):
        return table.assign(**key)
    per_class = with_key(per_class)
    pattern_rows = with_key(pattern_rows)
    accepted = is_known & ~rejected
    benign = evaluation.actual.eq("benign").to_numpy() & is_known
    unknown_rates = classes.loc[classes.partition.eq("unknown"), "reject_rate"]
    record = dict(**key, rows=len(evaluation), known_rows=int(is_known.sum()), unknown_rows=int((~is_known).sum()),
        macro_f1=float(f1.mean()), accuracy=float(tp.sum() / matrix.sum()),
        known_reject_rows=int(rejected[is_known].sum()), known_reject_rate=float(rejected[is_known].mean()),
        unknown_reject_rows=int(rejected[~is_known].sum()), unknown_reject_rate=float(rejected[~is_known].mean()),
        unknown_class_macro_reject_rate=float(unknown_rates.mean()),
        unknown_pattern_macro_reject_rate=float(per_class.pattern_equal_weight_reject_rate.mean()),
        benign_rows=int(benign.sum()), benign_reject_rows=int(rejected[benign].sum()),
        benign_reject_rate=float(rejected[benign].mean()) if benign.any() else np.nan,
        benign_wrong_known_rows=int((benign & ~rejected & (predicted != "benign")).sum()),
        benign_wrong_known_rate=float((~rejected[benign] & (predicted[benign] != "benign")).mean()) if benign.any() else np.nan,
        accepted_known_rows=int(accepted.sum()), known_coverage=float(accepted.sum() / is_known.sum()),
        accepted_known_accuracy=float((predicted[accepted] == actual[accepted]).mean()) if accepted.any() else np.nan)
    confusion = pd.DataFrame([(a, b, int(matrix[i, j])) for i, a in enumerate(open_labels)
                             for j, b in enumerate(open_labels)], columns=["actual", "predicted", "count"]).assign(**key)
    return record, confusion, classes, pattern_rows, per_class


def auc_report(predictions, identity):
    p = predictions.loc[predictions.partition.isin(["test", "unknown"])]
    return dict(**identity, known_rows=int(p.partition.eq("test").sum()),
                unknown_rows=int(p.partition.eq("unknown").sum()),
                auroc=float(roc_auc_score(p.partition.eq("unknown").astype(int), 1 - p.max_probability)))


def summarize_policies(metrics):
    keys = ["machine", "feature_group", "split", "model", "policy_id"]
    if metrics.duplicated(keys + ["seed"]).any():
        raise ValueError("정책 조건·시드 중복")
    records = []
    for key, group in metrics.groupby(keys, sort=False):
        if set(group.seed) != {0, 1, 2}:
            raise ValueError("세 시드가 모두 필요합니다")
        record = dict(zip(keys, key), runs=3, known_rows_min=int(group.known_rows.min()),
                      known_rows_max=int(group.known_rows.max()), unknown_rows=int(group.unknown_rows.iloc[0]))
        for metric in METRICS:
            record.update({f"{metric}_mean": group[metric].mean(), f"{metric}_std": group[metric].std(ddof=1),
                           f"{metric}_valid_runs": int(group[metric].notna().sum())})
        records.append(record)
    return pd.DataFrame(records)
