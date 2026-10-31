"""시퀀스 앞 5/10/20개 비교 36회와 기존 30개 기준선의 부분집합 감사."""

import argparse
from itertools import product
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from src.common.evaluation import evaluate_classification
from src.common.experiment import run_classification
from src.common.results import save_run, sha256_file, summarize_runs
from src.track_a.feature_group_experiment import ROOT, code_hashes as baseline_code_hashes, load_inputs
from src.track_a.preprocess import feature_groups
from src.track_a.result_metadata import compact_batch
from src.track_a.sequence_controls import load_baseline
from src.track_a.sequence_prefix_analysis import (
    KS, audit_prefix, prefix_input, score_subsets, summarize_subsets, validate_predictions,
)


def conditions():
    return list(product(("random", "pattern_group"), (0, 1, 2), ("rf", "lgbm"), KS))


def code_hashes():
    names = [*baseline_code_hashes(), "src/track_a/sequence_controls.py", "src/track_a/result_metadata.py",
             "src/track_a/sequence_prefix_analysis.py", "src/track_a/sequence_prefix_experiment.py"]
    return {name: sha256_file(ROOT / name) for name in names}


def load_baseline_predictions(data_dir, run_id, original, frame, splits, known):
    """기준선 저장본의 원본 배정·점수·전체 혼동 행렬을 학습 전에 대조한다."""
    paths, saved = {}, {}
    matrix = pd.read_csv(ROOT / "outputs/results" / run_id / "confusion.csv", keep_default_na=False)
    for row in original.itertuples(index=False):
        child = f"{run_id}_{row.split}_{row.model}_sequence_s{row.seed}"
        folder = data_dir / "predictions" / child
        for name in ("predictions.csv", "assignments.csv"):
            paths[f"{child}_{name}"] = folder / name
        predictions = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        assignments = pd.read_csv(folder / "assignments.csv", keep_default_na=False)
        predictions = validate_predictions(predictions, assignments, frame.Label,
                                             splits[f"{row.split}_{row.seed}"], known)
        test = predictions.loc[predictions.partition.eq("test")]
        scores, confusion = evaluate_classification(test.actual, test.predicted, labels=known)
        if any(not np.isclose(scores[key], getattr(row, key), rtol=0, atol=1e-12)
               for key in ("macro_f1", "accuracy")):
            raise ValueError(f"기준선 예측 재계산 점수 불일치: {child}")
        selected = matrix.loc[matrix.split.eq(row.split) & matrix.seed.eq(row.seed)
                              & matrix.model.eq(row.model) & matrix.feature_group.eq("sequence"),
                              ["actual", "predicted", "count"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, confusion, check_dtype=False, check_exact=True)
        saved[row.split, row.seed, row.model] = predictions
    return saved, paths


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_batch(*, run_id, baseline_run_id, preprocess_run_id="2026-10-29", workers=2):
    from src.common.config import DATA_DIR, MACHINE

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or type(workers) is not int or workers < 1:
        raise ValueError("run-id 또는 workers 오류")
    data_dir = DATA_DIR.expanduser().resolve()
    if data_dir.is_relative_to(ROOT):
        raise ValueError("DATA_DIR은 저장소 밖이어야 합니다")
    jobs = conditions()
    child_ids = [f"{run_id}_{split}_{model}_k{k}_s{seed}" for split, seed, model, k in jobs]
    out = ROOT / "outputs/results" / run_id
    if out.exists() or any((out.parent / child).exists() or (data_dir / "predictions" / child).exists()
                           for child in child_ids):
        raise FileExistsError("기존 run-id를 덮어쓰지 않습니다")
    frame, registry, splits, known, paths, hashes = load_inputs(data_dir, preprocess_run_id)
    original, baseline_paths = load_baseline(baseline_run_id, machine=MACHINE, hashes=hashes, workers=workers)
    paths.update(baseline_paths)
    saved, prediction_paths = load_baseline_predictions(data_dir, baseline_run_id, original,
                                                        frame, splits, known)
    paths.update(prediction_paths)
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_code = code_hashes()
    sequence = frame[feature_groups()["sequence"]]
    audit, masks = [], {}
    for split, seed, k in product(("random", "pattern_group"), (0, 1, 2), (*KS, 30)):
        selected_masks, record = audit_prefix(sequence, frame.Label, splits[f"{split}_{seed}"],
                                               split=split, seed=seed, k=k)
        masks[split, seed, k] = selected_masks
        audit.append(record)
        if split == "pattern_group" and k == 30 and record.loc[
                record.partition.isin(["test", "unknown"]), "prefix_seen_rows"].sum():
            raise ValueError("원본 30개 그룹이 학습과 겹칩니다")
    audit = pd.concat(audit, ignore_index=True)
    manifest = dict(run_id=run_id, baseline_run_id=baseline_run_id, preprocess_run_id=preprocess_run_id,
        machine=MACHINE, preliminary=MACHINE != "colab-cpu", complete=False, completed_runs=0,
        expected_runs=36, run_ids=child_ids, source_sha256=source_hashes, code_sha256=source_code,
        known_labels=known, input_policy="30 signed lengths; keep first k, zero suffix; existing padding retained",
        ks=list(KS), reused_k=30, validation="none; fit entire outer train; no tuning",
        seed_policy="outer split_seed=model_seed=0/1/2; no reselection",
        grouping="original 30-value patterns; prefix overlap reported, not regrouped",
        subset_policy="fixed known labels; zero_division=0; support=0 recall NA; empty scores NA; 1..29 low_support",
        interpretation="offline payload-packet budget classification; not wall-clock early detection",
        git_commit_at_start=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip(),
        git_dirty_at_start=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8").strip()))
    out.mkdir(parents=True)
    def checkpoint():
        _write_json(out / "manifest.json", manifest)
    checkpoint()
    audit.to_csv(out / "prefix_audit.csv", index=False)
    subset_scores, subset_classes, subset_confusion = [], [], []
    curve, metrics, matrices = [], [], []
    for row in original.to_dict("records"):
        split, seed, model = row["split"], row["seed"], row["model"]
        child = f"{baseline_run_id}_{split}_{model}_sequence_s{seed}"
        identity = dict(machine=MACHINE, split=split, seed=seed, model=model, k=30, source_run_id=child)
        reports = score_subsets(saved[split, seed, model], known_labels=known,
                                masks=masks[split, seed, 30], identity=identity)
        for table, report in zip((subset_scores, subset_classes, subset_confusion), reports):
            table.append(report)
        curve.append(pd.DataFrame([dict(row, k=30, source_run_id=child)]))
    del saved
    note = "맥 예비; 최종은 Colab CPU" if manifest["preliminary"] else "Colab CPU; 교차 검토 후 보고서 사용"
    for index, ((split, seed, model, k), child) in enumerate(zip(jobs, child_ids), start=1):
        partition = splits[f"{split}_{seed}"]
        result = run_classification(prefix_input(sequence, k), frame.Label, partition,
            known_labels=known, model_name=model, seed=seed, split=split, groups=registry.pattern_id, workers=workers)
        result.metrics.update(k=k, observed_feature_slots=k)
        result.metadata.update(k=k, original_feature_slots=30, observed_feature_slots=k,
                               prefix_policy=manifest["input_policy"])
        predictions = validate_predictions(result.predictions, result.assignments, frame.Label, partition, known)
        reports = score_subsets(predictions, known_labels=known, masks=masks[split, seed, k],
            identity=dict(machine=MACHINE, split=split, seed=seed, model=model, k=k, source_run_id=child))
        all_score = reports[0].loc[reports[0].subset.eq("all")].iloc[0]
        if any(not np.isclose(all_score[key], result.metrics[key], rtol=0, atol=1e-12)
               for key in ("macro_f1", "accuracy")):
            raise ValueError("부분집합 평가의 전체 점수가 공통 평가기와 다릅니다")
        report, _ = save_run(result, run_id=child, track="A", experiment="sequence_prefix",
            feature_group=f"sequence_prefix_k{k}", split_seed=seed, source_hashes=source_hashes, note=note,
            extra_metadata=dict(batch_run_id=run_id, code_sha256=source_code,
                                preliminary=manifest["preliminary"], baseline_run_id=baseline_run_id))
        row = pd.read_csv(report / "metrics.csv", keep_default_na=False, float_precision="round_trip")
        matrix = pd.read_csv(report / "confusion.csv", keep_default_na=False)
        metrics.append(row)
        matrices.append(matrix)
        curve.append(row.assign(source_run_id=child))
        for table, partial in zip((subset_scores, subset_classes, subset_confusion), reports):
            table.append(partial)
        for name, tables in (("metrics", metrics), ("confusion", matrices), ("subset_scores", subset_scores),
                             ("subset_class_scores", subset_classes), ("subset_confusion", subset_confusion)):
            pd.concat(tables, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
        manifest["completed_runs"] = index
        checkpoint()
        print(f"{index}/36 {split} s{seed} {model} k={k}: macro_f1={result.metrics['macro_f1']:.6f} "
              f"fit={result.metrics['train_seconds']:.1f}s", flush=True)
        del result
    if (code_hashes() != source_code or any(sha256_file(path) != source_hashes[name] for name, path in paths.items())):
        raise ValueError("실행 중 입력·기준선·코드 변경")
    combined = pd.concat(metrics, ignore_index=True)
    summary = summarize_runs(combined)
    curve = pd.concat(curve, ignore_index=True)
    curve_summary = summarize_runs(curve)
    curve_summary["k"] = curve_summary.feature_group.map({"sequence": 30, **{f"sequence_prefix_k{k}": k for k in KS}})
    paired = combined.merge(original[["split", "seed", "model", "macro_f1", "accuracy"]],
        on=["split", "seed", "model"], suffixes=("_prefix", "_k30"), validate="many_to_one")
    for score in ("macro_f1", "accuracy"):
        paired[f"delta_{score}"] = paired[f"{score}_prefix"] - paired[f"{score}_k30"]
    if (len(combined) != 36 or len(summary) != 12 or len(pd.concat(matrices)) != 6084
            or len(curve) != 48 or len(curve_summary) != 16 or len(paired) != 36
            or not combined.validation_rows.eq(0).all()):
        raise ValueError("실행·집계 건수 오류")
    for name, table in (("summary", summary), ("comparison_curve", curve), ("curve_summary", curve_summary),
                        ("paired", paired), ("subset_summary", summarize_subsets(pd.concat(subset_scores)))):
        table.to_csv(out / f"{name}.csv", index=False)
    manifest["complete"] = True
    checkpoint()
    try:
        compact_batch(out)
    except Exception:
        manifest["complete"] = False
        checkpoint()
        raise
    print("COMPLETE:", out, flush=True)
    print(curve_summary.to_string(index=False), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--preprocess-run-id", default="2026-10-29")
    parser.add_argument("--workers", type=int, default=2)
    run_batch(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
