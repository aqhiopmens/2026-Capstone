"""A: train 내부 검증으로 고정하는 RF 미확인 거절 평가 12회/환경."""

import argparse
from itertools import product
import json
import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from src.common.experiment import run_classification
from src.common.evaluation import evaluate_classification
from src.common.results import save_run, sha256_file, summarize_runs
from src.common.splits import split_validation
from src.track_a.feature_group_experiment import ROOT, load_inputs, code_hashes as base_hashes
from src.track_a.preprocess import feature_groups
from src.track_a.result_metadata import compact_batch
from src.track_a.ood_threshold_analysis import (probability_columns, validate_saved, policies_for,
    partition_audit, score_audit, evaluate_policy, auc_report, summarize_policies, METRICS)


def conditions():
    return list(product(("random", "pattern_group"), (0, 1, 2), ("all", "sequence")))


def code_hashes():
    names = [*base_hashes(), "src/track_a/result_metadata.py", "src/track_a/ood_threshold_analysis.py",
             "src/track_a/ood_threshold_experiment.py"]
    return {name: sha256_file(ROOT / name) for name in names}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_record(path):
    raw = path.read_bytes()
    return dict(sha256=sha256_file(path), bytes=len(raw),
                lf_sha256=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                normalization="CRLF->LF; text only; original bytes preserved")


def paired_policies(metrics):
    keys = ["machine", "feature_group", "split", "model", "seed", "source_run_id"]
    paired = metrics.loc[metrics.policy_id.ne("none")].merge(
        metrics.loc[metrics.policy_id.eq("none"), [*keys, *METRICS]],
        on=keys, suffixes=("", "_none"), validate="many_to_one")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_none"]
    return paired


def run_batch(*, run_id, preprocess_run_id="2026-10-29", workers=2):
    from src.common.config import DATA_DIR, MACHINE

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or type(workers) is not int or workers != 2:
        raise ValueError("run-id/workers 오류")
    data_dir = DATA_DIR.expanduser().resolve()
    if data_dir.is_relative_to(ROOT):
        raise ValueError("DATA_DIR은 저장소 밖이어야 합니다")
    jobs = conditions()
    children = [f"{run_id}_{split}_rf_{features}_s{seed}" for split, seed, features in jobs]
    out = ROOT / "outputs/results" / run_id
    if out.exists() or any((out.parent / child).exists() or (data_dir / "predictions" / child).exists() for child in children):
        raise FileExistsError("기존 실행을 덮어쓰지 않습니다")
    frame, registry, outer_splits, known, paths, hashes = load_inputs(data_dir, preprocess_run_id)
    partitions, audits = {}, []
    for split, seed in product(("random", "pattern_group"), range(3)):
        outer = outer_splits[f"{split}_{seed}"]
        part = split_validation(frame.Label, outer, split=split, seed=seed,
                               groups=registry.pattern_id if split == "pattern_group" else None)
        if not part.loc[outer.ne("train")].equals(outer.loc[outer.ne("train")]):
            raise ValueError("외부 배정 변경")
        partitions[split, seed] = part
        audits.append(partition_audit(frame.Label, part, registry.pattern_id, dict(split=split, seed=seed)))
    source_code = code_hashes()
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    manifest = dict(run_id=run_id, preprocess_run_id=preprocess_run_id, machine=MACHINE,
        preliminary=MACHINE != "colab-cpu", complete=False, completed_runs=0, expected_runs=12,
        completed_policies=0, expected_policies=48, run_ids=children, source_sha256=hashes,
        code_sha256=source_code, known_labels=known, unknown_labels=sorted(frame.Label[registry.role.eq("unknown")].unique()),
        policies=["none", "reject_01", "reject_05", "reject_10"], main_policy="reject_05",
        calibration="validation only; strict score<threshold; max observed threshold under integer rejection budget; ties accepted",
        fit_policy="final train only; no validation refit; previous models/predictions not reused",
        probability_columns=dict(zip(probability_columns(known), known)),
        git_commit_at_start=git("rev-parse", "HEAD"), git_dirty_at_start=bool(git("status", "--porcelain")),
        git_dirty_source_at_start=bool(git("status", "--porcelain", "--", "src", "tests", "pyproject.toml", "uv.lock")))
    out.mkdir(parents=True)
    write_json(out / "manifest.json", manifest)
    pd.concat(audits, ignore_index=True).to_csv(out / "partition_audit.csv", index=False)
    tables = {name: [] for name in ("metrics", "confusion", "thresholds", "policy_metrics", "open_confusion",
               "class_rejection", "pattern_rejection", "pattern_summary", "score_audit", "auc")}
    note = "맥 예비; 최종은 Colab CPU" if manifest["preliminary"] else "Colab CPU; 독립 교차 검증 후 최종 보고"
    for index, ((split, seed, features), child) in enumerate(zip(jobs, children), start=1):
        part = partitions[split, seed]
        x = frame[feature_groups()[features]]
        result = run_classification(x, frame.Label, part, known_labels=known, model_name="rf", seed=seed,
                                    split=split, groups=registry.pattern_id, workers=workers)
        if result.metrics["validation_rows"] == 0 or result.model.classes_.tolist() != known:
            raise ValueError("검증 없는 모델 또는 클래스 순서 불일치")
        pred = result.predictions.copy()
        probabilities = result.model.predict_proba(x.loc[pred.row_id])
        pred[probability_columns(known)] = probabilities
        # This full-vector inference is the authoritative saved prediction. RF's
        # parallel tree accumulation can differ at floating point round-off.
        pred["max_probability"] = probabilities.max(1)
        pred["predicted"] = result.model.classes_[probabilities.argmax(1)]
        test = pred.loc[pred.partition.eq("test")]
        scores, result.confusion = evaluate_classification(test.actual, test.predicted, labels=known)
        result.metrics.update(scores)
        pred = validate_saved(pred, result.assignments, frame.Label, part, known)
        result.predictions = pred
        selected, policies = policies_for(pred)  # Only validation rows are used here.
        result.metadata.update(probability_classes=known, probability_columns=probability_columns(known),
            unknown="held-out evaluation only; never used for threshold selection", validation_refit=False,
            rejection="max probability; validation-selected strict threshold; ties accepted")
        report, row_out = save_run(result, run_id=child, track="A", experiment="ood_threshold",
            feature_group=features, split_seed=seed, source_hashes=hashes, note=note,
            extra_metadata=dict(batch_run_id=run_id, code_sha256=source_code, preliminary=manifest["preliminary"],
                                calibration=selected))
        reread = pd.read_csv(row_out / "predictions.csv", float_precision="round_trip")
        assignments = pd.read_csv(row_out / "assignments.csv", keep_default_na=False)
        checked = validate_saved(reread, assignments, frame.Label, part, known)
        np.testing.assert_array_equal(pred[["max_probability", *probability_columns(known)]],
                                      checked[["max_probability", *probability_columns(known)]])
        meta_path = report / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["extra"]["row_files"] = {name: file_record(row_out / name) for name in ("predictions.csv", "assignments.csv")}
        write_json(meta_path, metadata)
        closed = pd.read_csv(report / "metrics.csv", keep_default_na=False, float_precision="round_trip")
        matrices = pd.read_csv(report / "confusion.csv", keep_default_na=False)
        identity = closed.iloc[0][["date", "track", "experiment", "split", "feature_group", "model", "seed", "split_seed",
                                  "machine", "sklearn", "lightgbm", "torch", "note"]].to_dict()
        identity["source_run_id"] = child
        tables["metrics"].append(closed)
        tables["confusion"].append(matrices)
        tables["thresholds"].append(pd.DataFrame([dict(**identity, **row) for row in selected]))
        for policy in policies:
            values = evaluate_policy(checked, known, registry.pattern_id, policy, identity)
            tables["policy_metrics"].append(pd.DataFrame([values[0]]))
            for name, table in zip(("open_confusion", "class_rejection", "pattern_rejection", "pattern_summary"), values[1:]):
                tables[name].append(table)
        tables["score_audit"].append(score_audit(checked, identity))
        tables["auc"].append(pd.DataFrame([auc_report(checked, identity)]))
        for name, pieces in tables.items():
            pd.concat(pieces, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
        manifest.update(completed_runs=index, completed_policies=index * 4)
        write_json(out / "manifest.json", manifest)
        primary = tables["policy_metrics"][-2].iloc[0]  # none, 1%, 5%, 10%
        print(f"{index}/12 {split} s{seed} {features}: closed F1={result.metrics['macro_f1']:.6f}; "
              f"tau={primary.threshold:.8g}; known reject={primary.known_reject_rate:.4f}; "
              f"unknown reject={primary.unknown_reject_rate:.4f}", flush=True)
        del result
    if code_hashes() != source_code or any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("실행 중 입력 또는 코드 변경")
    combined = {name: pd.concat(value, ignore_index=True) for name, value in tables.items()}
    expected = dict(metrics=12, confusion=2028, thresholds=36, policy_metrics=48, open_confusion=9408,
                    class_rejection=864, pattern_summary=240, score_audit=36, auc=12)
    if any(len(combined[name]) != size for name, size in expected.items()):
        raise ValueError("완료 모델·정책·집계 건수 불일치")
    summarize_runs(combined["metrics"]).to_csv(out / "summary.csv", index=False)
    summarize_policies(combined["policy_metrics"]).to_csv(out / "policy_summary.csv", index=False)
    paired_policies(combined["policy_metrics"]).to_csv(out / "paired.csv", index=False)
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)
    try:
        compact_batch(out)
        from src.track_a.verify_ood_threshold import verify
        verify(run_id, data_dir, MACHINE)
    except Exception:
        manifest["complete"] = False
        write_json(out / "manifest.json", manifest)
        raise
    print("COMPLETE 12/12 + saved-prediction verification:", out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preprocess-run-id", default="2026-10-29")
    parser.add_argument("--workers", type=int, default=2)
    run_batch(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
