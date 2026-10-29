"""패딩을 유지한 행별 순서 섞기 12회와 학습 패턴 조회 기준선 6회."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd

from src.common.evaluation import evaluate_classification
from src.common.experiment import RunResult, run_classification
from src.common.models import make_model
from src.common.results import save_run, sha256_file, summarize_runs
from src.common.splits import validate_partitions
from src.track_a.feature_group_experiment import ROOT, code_hashes as baseline_code_hashes, load_inputs
from src.track_a.preprocess import feature_groups


def shuffle_packets(sequence, seed):
    """행 ID+시드로 비영 패킷 값을 독립 순열화한다. 0 패딩 위치는 고정한다."""
    if list(sequence.columns) != feature_groups()["sequence"] or not sequence.index.is_unique:
        raise ValueError("고유 행 ID와 순서대로 된 시퀀스 30개 컬럼이 필요합니다")
    values = sequence.to_numpy(copy=True)
    if not np.isfinite(values).all() or not np.equal(values, np.trunc(values)).all():
        raise ValueError("시퀀스는 유한한 정수여야 합니다")
    nonzero = values != 0
    if (np.maximum.accumulate(~nonzero, axis=1) & nonzero).any():
        raise ValueError("0이 아닌 값 앞에 0이 있습니다. 패딩 정책을 재검토하세요")
    for index, row_id in enumerate(sequence.index):
        if not isinstance(row_id, (int, np.integer)) or row_id < 0:
            raise ValueError("행 ID는 비음수 정수여야 합니다")
        length = int(nonzero[index].sum())
        generator = np.random.default_rng(np.random.SeedSequence([seed, int(row_id)]))
        values[index, :length] = generator.permutation(values[index, :length])
    return pd.DataFrame(values, index=sequence.index, columns=sequence.columns)


def hash_baseline(labels, partitions, patterns, *, known_labels, split, seed):
    """학습 패턴별 다수 라벨 조회. 미조회는 학습 최빈 라벨, 동률은 문자열순."""
    validate_partitions(labels, partitions, known_labels, patterns if split == "pattern_group" else None)
    if not patterns.index.equals(labels.index) or patterns.isna().any():
        raise ValueError("패턴 ID의 정렬·결측 오류")
    if partitions.eq("validation").any():
        raise ValueError("이 기준선은 외부 train 전체 배정을 사용합니다")
    train = partitions.eq("train")
    start = time.monotonic()
    votes = defaultdict(Counter)
    for pattern, label in zip(patterns[train], labels[train]):
        votes[pattern][label] += 1
    majority = lambda counts: min(counts, key=lambda label: (-counts[label], label))
    fallback = majority(Counter(labels[train]))
    lookup = {pattern: majority(counts) for pattern, counts in votes.items()}
    elapsed = time.monotonic() - start
    selected = partitions.isin(["test", "unknown"])
    predictions = pd.DataFrame(dict(row_id=labels.index[selected], partition=partitions[selected].to_numpy(),
        actual=labels[selected].to_numpy(), predicted=[lookup.get(p, fallback) for p in patterns[selected]],
        lookup_hit=[p in lookup for p in patterns[selected]]))
    test = predictions.loc[predictions.partition.eq("test")]
    metrics, confusion = evaluate_classification(test.actual, test.predicted, labels=known_labels)
    metrics.update(n_features=30, train_rows=int(train.sum()), validation_rows=0, test_rows=len(test),
        unknown_rows=int(partitions.eq("unknown").sum()), train_seconds=elapsed,
        lookup_hit_rows=int(test.lookup_hit.sum()), lookup_coverage=float(test.lookup_hit.mean()),
        lookup_hit_accuracy=float(test.loc[test.lookup_hit, "actual"].eq(
            test.loc[test.lookup_hit, "predicted"]).mean()) if test.lookup_hit.any() else None)
    metadata = dict(labels=list(known_labels), feature_columns=feature_groups()["sequence"],
        model_name="hash_rule", model_seed=seed, split=split, fit_partition="train",
        model_params=dict(key="original sequence SHA-256", conflict="train majority; lexical ties",
                          unseen="train class majority; lexical ties", fallback_label=fallback),
        train_patterns=len(lookup), ambiguous_train_patterns=sum(len(v) > 1 for v in votes.values()),
        preprocessing="none; original sequence patterns, not shuffled",
        unknown="lookup predictions only; not unknown detection; unseen is not automatically malicious",
        seed_policy="seed changes outer split only; lookup has no random model seed")
    assignments = pd.DataFrame(dict(row_id=labels.index, partition=partitions.to_numpy()))
    return RunResult(None, metrics, confusion, predictions, assignments, metadata)


def load_baseline(run_id, *, machine, hashes, workers):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("잘못된 baseline-run-id")
    folder = ROOT / "outputs/results" / run_id
    paths = dict(baseline_manifest=folder / "manifest.json", baseline_metrics=folder / "metrics.csv")
    manifest = json.loads(paths["baseline_manifest"].read_text(encoding="utf-8"))
    if (not manifest["complete"] or manifest["completed_runs"] != 48
            or manifest["machine"] != machine or manifest["source_sha256"] != hashes
            or manifest["code_sha256"] != baseline_code_hashes()):
        raise ValueError("원순서 기준선의 완료·환경·입력·코드가 현재 실험과 다릅니다")
    metrics = pd.read_csv(paths["baseline_metrics"], keep_default_na=False)
    selected = metrics.loc[metrics.feature_group.eq("sequence")].copy()
    keys = ["split", "seed", "model"]
    expected = {(s, i, m) for s in ("random", "pattern_group") for i in (0, 1, 2) for m in ("rf", "lgbm")}
    if len(selected) != 12 or set(map(tuple, selected[keys].to_numpy())) != expected:
        raise ValueError("원순서 기준선 12개 조건이 중복·누락되었습니다")
    for row in selected.itertuples(index=False):
        path = folder.parent / f"{run_id}_{row.split}_{row.model}_sequence_s{row.seed}" / "metadata.json"
        paths[f"baseline_{row.split}_{row.model}_{row.seed}"] = path
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (meta["model_params"] != make_model(row.model, seed=row.seed, workers=workers).get_params()
                or row.sklearn != version("scikit-learn") or row.lightgbm != version("lightgbm")
                or meta["numpy"] != version("numpy") or meta["pandas"] != version("pandas")
                or row.split_seed != row.seed or row.validation_rows != 0):
            raise ValueError("원순서와 섞기 비교의 모델 설정·버전·배정 정책이 다릅니다")
    return selected, paths


def run_controls(*, run_id, baseline_run_id, preprocess_run_id="2026-10-29", workers=2):
    from src.common.config import DATA_DIR, MACHINE

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or workers < 1:
        raise ValueError("run-id 또는 workers 오류")
    data_dir = DATA_DIR.expanduser().resolve()
    if data_dir.is_relative_to(ROOT):
        raise ValueError("DATA_DIR은 저장소 밖이어야 합니다")
    jobs = [(s, i, m) for s in ("random", "pattern_group") for i in (0, 1, 2)
            for m in ("hash_rule", "rf", "lgbm")]
    child_ids = [f"{run_id}_{s}_{m}_s{i}" for s, i, m in jobs]
    out = ROOT / "outputs/results" / run_id
    if out.exists() or any((out.parent / child).exists() or (data_dir / "predictions" / child).exists()
                           for child in child_ids):
        raise FileExistsError("기존 run-id를 덮어쓰지 않습니다")
    frame, registry, splits, known, paths, hashes = load_inputs(data_dir, preprocess_run_id)
    original, baseline_paths = load_baseline(baseline_run_id, machine=MACHINE, hashes=hashes, workers=workers)
    paths.update(baseline_paths)
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    code_paths = {**{name: ROOT / name for name in baseline_code_hashes()},
                  "src/track_a/sequence_controls.py": Path(__file__)}
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    manifest = dict(run_id=run_id, baseline_run_id=baseline_run_id, machine=MACHINE,
        preliminary=MACHINE != "colab-cpu", complete=False, completed_runs=0, expected_runs=18,
        run_ids=child_ids, source_sha256=input_hashes, code_sha256=code_hashes,
        git_commit_at_start=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_dirty_at_start=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        shuffle="independent row permutations of signed nonzero values; trailing zero positions unchanged; row ID+seed",
        grouping="original unshuffled patterns; outer assignments unchanged",
        known_labels=known, validation="none", seed_policy="split, model and shuffle share seed 0/1/2",
        interpretation="RF/LightGBM position-information control; not proof that temporal models are unnecessary")
    out.mkdir(parents=True)
    def checkpoint():
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint()
    metrics, matrices, coverage = [], [], []
    shuffled = {seed: shuffle_packets(frame[feature_groups()["sequence"]], seed) for seed in (0, 1, 2)}
    note = ("맥/로컬 예비; 최종 수치는 Colab CPU 재실행" if manifest["preliminary"]
            else "Colab CPU; 버전·교차 검증 후 보고서 사용")
    for index, ((split, seed, model), child) in enumerate(zip(jobs, child_ids), start=1):
        partition = splits[f"{split}_{seed}"]
        if model == "hash_rule":
            result = hash_baseline(frame.Label, partition, registry.pattern_id, known_labels=known,
                                   split=split, seed=seed)
            for label in known:
                p = result.predictions.query("partition == 'test'")
                p = p.loc[p.actual.eq(label)]
                coverage.append(dict(split=split, seed=seed, label=label, test_rows=len(p),
                                     hit_rows=int(p.lookup_hit.sum()), coverage=float(p.lookup_hit.mean())))
            experiment, group = "hash_rule", "sequence"
        else:
            result = run_classification(shuffled[seed], frame.Label, partition, known_labels=known,
                model_name=model, seed=seed, split=split, groups=registry.pattern_id, workers=workers)
            experiment, group = "sequence_shuffle", "sequence_nonzero_shuffled"
        report, _ = save_run(result, run_id=child, track="A", experiment=experiment,
            feature_group=group, split_seed=seed, source_hashes=input_hashes, note=note,
            extra_metadata=dict(batch_run_id=run_id, code_sha256=code_hashes, preliminary=manifest["preliminary"],
                                shuffle_seed=seed if model != "hash_rule" else None))
        metrics.append(pd.read_csv(report / "metrics.csv", keep_default_na=False))
        matrices.append(pd.read_csv(report / "confusion.csv", keep_default_na=False))
        pd.concat(metrics, ignore_index=True).to_csv(out / "metrics.csv", index=False)
        pd.concat(matrices, ignore_index=True).to_csv(out / "confusion.csv", index=False)
        pd.DataFrame(coverage).to_csv(out / "hash_coverage.csv", index=False)
        manifest["completed_runs"] = index
        checkpoint()
        print(f"{index}/18 {split} s{seed} {model}: macro_f1={result.metrics['macro_f1']:.6f} "
              f"accuracy={result.metrics['accuracy']:.6f}", flush=True)
        del result
    if any(sha256_file(path) != input_hashes[name] for name, path in paths.items()) or any(
            sha256_file(path) != code_hashes[name] for name, path in code_paths.items()):
        raise ValueError("실행 중 입력·기준선·코드가 바뀌었습니다")
    combined = pd.concat(metrics, ignore_index=True)
    summary = summarize_runs(combined)
    paired = combined.loc[combined.model.isin(["rf", "lgbm"])].merge(
        original[["split", "seed", "model", "macro_f1", "accuracy"]], on=["split", "seed", "model"],
        suffixes=("_shuffled", "_original"), validate="one_to_one")
    for score in ("macro_f1", "accuracy"):
        paired[f"delta_{score}"] = paired[f"{score}_shuffled"] - paired[f"{score}_original"]
    if len(combined) != 18 or len(summary) != 6 or len(paired) != 12:
        raise ValueError("조건 수 오류")
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    manifest["complete"] = True
    checkpoint()
    print(summary.to_string(index=False), flush=True)
    print("COMPLETE:", out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--preprocess-run-id", default="2026-10-29")
    parser.add_argument("--workers", type=int, default=2)
    run_controls(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
