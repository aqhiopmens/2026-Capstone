"""학습 없이 저장 행별 예측에서 prefix 배치·부분집합 집계를 독립 재계산한다."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.common.results import sha256_file, summarize_runs
from src.track_a.feature_group_experiment import ROOT, load_inputs
from src.track_a.result_metadata import read_archive
from src.track_a.sequence_controls import load_baseline
from src.track_a.sequence_prefix_experiment import code_hashes, conditions, load_baseline_predictions
from src.track_a.sequence_prefix_analysis import validate_predictions


def close(actual, expected):
    if not (pd.isna(actual) and pd.isna(expected)) and not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"수치 불일치: {actual} != {expected}")


def verify(run_id, data_dir, machine):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("배치 ID 오류")
    out = ROOT / "outputs/results" / run_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest["complete"] or manifest["completed_runs"] != 36 or manifest["expected_runs"] != 36
            or manifest["machine"] != machine or manifest["code_sha256"] != code_hashes()):
        raise ValueError("완료·환경·코드 검증 실패")
    frame, registry, splits, known, paths, hashes = load_inputs(data_dir, manifest["preprocess_run_id"])
    baseline = manifest["baseline_run_id"]
    original, baseline_paths = load_baseline(baseline, machine=machine, hashes=hashes, workers=2)
    saved, prediction_paths = load_baseline_predictions(data_dir, baseline, original, frame, splits, known)
    paths.update(baseline_paths)
    paths.update(prediction_paths)
    if {name: sha256_file(path) for name, path in paths.items()} != manifest["source_sha256"]:
        raise ValueError("입력·기준선 파일 SHA 불일치")
    archived = read_archive(out)
    if archived is None or set(archived) != set(manifest["run_ids"]) or len(archived) != 36:
        raise ValueError("36개 메타데이터 보존 오류")
    read = lambda name: pd.read_csv(out / f"{name}.csv", float_precision="round_trip", keep_default_na=name not in {
        "metrics", "confusion", "comparison_curve", "curve_summary", "summary", "paired"})
    metrics, matrices = read("metrics"), read("confusion")
    subsets, class_scores, subgroup_matrices = read("subset_scores"), read("subset_class_scores"), read("subset_confusion")
    audits = read("prefix_audit")
    if (len(metrics) != 36 or len(matrices) != 6084 or len(subsets) != 240
            or len(class_scores) != 3120 or len(subgroup_matrices) != 40560):
        raise ValueError("배치·부분집합 건수 불일치")
    expected = set(conditions())
    if set(map(tuple, metrics[["split", "seed", "model", "k"]].to_numpy())) != expected:
        raise ValueError("조건 누락·중복")
    summary = summarize_runs(metrics)
    pd.testing.assert_frame_equal(read("summary"), summary, check_dtype=False, rtol=0, atol=1e-12)
    curve = read("comparison_curve")
    if len(curve) != 48 or len(read("curve_summary")) != 16 or len(read("paired")) != 36:
        raise ValueError("기준선 짝 비교 건수")
    curve_summary = summarize_runs(curve)
    observed_curve_summary = read("curve_summary").drop(columns="k")
    pd.testing.assert_frame_equal(observed_curve_summary, curve_summary, check_dtype=False, rtol=0, atol=1e-12)
    lengths = frame[[str(i) for i in range(30)]].ne(0).sum(axis=1)
    checked = 0
    for row in curve.to_dict("records"):
        split, seed, model, k = row["split"], row["seed"], row["model"], int(row["k"])
        parts = splits[f"{split}_{seed}"]
        child = row["source_run_id"]
        if k == 30:
            pred = saved[split, seed, model]
            original_row = original.loc[original.split.eq(split) & original.seed.eq(seed) & original.model.eq(model)].iloc[0]
        else:
            folder = data_dir / "predictions" / child
            pred = validate_predictions(pd.read_csv(folder / "predictions.csv", float_precision="round_trip"),
                pd.read_csv(folder / "assignments.csv", keep_default_na=False), frame.Label, parts, known)
            original_row = metrics.loc[metrics.split.eq(split) & metrics.seed.eq(seed) & metrics.model.eq(model) & metrics.k.eq(k)].iloc[0]
            meta = archived[child]
            if (meta["k"] != k or meta["feature_columns"] != [str(i) for i in range(30)]
                    or meta["identity"]["machine"] != machine or meta["extra"]["code_sha256"] != manifest["code_sha256"]):
                raise ValueError("실행 메타데이터 오류")
            pair = read("paired").query("split == @split and seed == @seed and model == @model and k == @k").iloc[0]
            baseline_row = original.loc[original.split.eq(split) & original.seed.eq(seed) & original.model.eq(model)].iloc[0]
            for score in ("macro_f1", "accuracy"):
                close(pair[f"{score}_prefix"], original_row[score])
                close(pair[f"{score}_k30"], baseline_row[score])
                close(pair[f"delta_{score}"], original_row[score] - baseline_row[score])
        test = pred.loc[pred.partition.eq("test")].set_index("row_id")
        values = frame[[str(i) for i in range(30)]].to_numpy(dtype="<i8", copy=True)
        values[:, k:] = 0
        patterns = pd.Series([v.tobytes() for v in values], index=frame.index)
        train_patterns = set(patterns.loc[parts.eq("train")])
        seen = patterns.isin(train_patterns)
        reached = lengths.ge(k)
        mask_map = dict(all=pd.Series(True, index=frame.index), reached_k=reached, short=~reached,
                        prefix_seen=seen, prefix_unseen=~seen)
        key_mask = subsets.split.eq(split) & subsets.seed.eq(seed) & subsets.model.eq(model) & subsets.k.eq(k)
        for subset, mask in mask_map.items():
            selected = test.loc[mask.loc[test.index]]
            n = len(selected)
            chosen_score = subsets.loc[key_mask & subsets.subset.eq(subset)]
            if len(chosen_score) != 1:
                raise ValueError("부분집합 키 중복·누락")
            score_row = chosen_score.iloc[0]
            close(score_row.rows, n)
            close(score_row.fraction, n / len(test))
            for score, value in (("macro_f1", f1_score(selected.actual, selected.predicted, labels=known,
                    average="macro", zero_division=0) if n else np.nan),
                    ("accuracy", accuracy_score(selected.actual, selected.predicted) if n else np.nan)):
                close(score_row[score], value)
                if subset == "all":
                    close(original_row[score], value)
                    close(row[score], value)
            matrix = confusion_matrix(selected.actual, selected.predicted, labels=known) if n else np.zeros((13, 13), dtype=int)
            selected_matrix = subgroup_matrices.loc[subgroup_matrices.source_run_id.eq(child) & subgroup_matrices.subset.eq(subset)]
            pd.testing.assert_frame_equal(selected_matrix[["actual", "predicted"]].reset_index(drop=True),
                pd.DataFrame([(a, p) for a in known for p in known], columns=["actual", "predicted"]))
            np.testing.assert_array_equal(selected_matrix["count"].to_numpy(), matrix.ravel())
            classes = class_scores.loc[class_scores.source_run_id.eq(child) & class_scores.subset.eq(subset)]
            assert classes.label.tolist() == known
            support, predicted_support, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
            np.testing.assert_array_equal(classes.support, support)
            for i, cls in enumerate(classes.itertuples(index=False)):
                close(cls.recall, tp[i] / support[i] if support[i] else np.nan)
                close(cls.f1, 2 * tp[i] / (support[i] + predicted_support[i]) if n and support[i] + predicted_support[i] else (0 if n else np.nan))
                assert cls.low_support == bool(0 < support[i] < 30)
            if k != 30 and subset == "all":
                selected_cm = matrices.loc[matrices.split.eq(split) & matrices.seed.eq(seed) & matrices.model.eq(model) & matrices.k.eq(k)]
                np.testing.assert_array_equal(selected_cm["count"], matrix.ravel())
            checked += 1
        for record in audits.loc[audits.split.eq(split) & audits.seed.eq(seed) & audits.k.eq(k)].to_dict("records"):
            selected = parts.eq(record["partition"])
            if record["label"] != "__all__":
                selected &= frame.Label.eq(record["label"])
            n = int(selected.sum())
            local = set(patterns.loc[selected])
            for name, value in dict(rows=n, reached_k_rows=int((selected & reached).sum()),
                short_rows=int((selected & ~reached).sum()), prefix_seen_rows=int((selected & seen).sum()),
                prefix_patterns=len(local), shared_prefix_patterns=len(local & train_patterns)).items():
                close(record[name], value)
    verification = dict(complete=True, run_id=run_id, machine=machine, new_runs=36, baseline_runs=12,
        checked_subsets=checked, new_confusion_cells=6084, subset_confusion_cells=40560,
        low_support_policy="scores retained; support 1..29 flagged", input_sha256=manifest["source_sha256"],
        verifier_sha256=sha256_file(Path(__file__)))
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in verification.items() if key != "input_sha256"}))
    return verification


def main():
    from src.common.config import DATA_DIR, MACHINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    verify(parser.parse_args().run_id, DATA_DIR, MACHINE)


if __name__ == "__main__":
    main()
