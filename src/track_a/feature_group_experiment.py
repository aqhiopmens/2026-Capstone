"""원본 재전처리본의 특징군 4 × 분할 2 × 모델 2 × 시드 3 비교.

Run: uv run python -m src.track_a.feature_group_experiment --run-id <ID>
맥 결과는 예비, 보고서용 최종 수치는 같은 코드의 Colab CPU 실행에서 얻는다.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from src.common.experiment import run_classification
from src.common.results import save_run, sha256_file, summarize_runs
from src.track_a.preprocess import (
    EXCLUDED, HELD_OUT, build_assignment, encode_frame, feature_groups,
    pattern_ids, row_roles, validate_assignment,
)


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("random", "pattern_group")
SEEDS = (0, 1, 2)
MODELS = ("rf", "lgbm")


def conditions():
    return list(product(SPLITS, SEEDS, MODELS, feature_groups()))


def validate_tables(features, rows, splits):
    """파일 간 행 정렬·형식·고정 정책·시드별 배정을 모델 fit 전에 검증한다."""
    for name, table in (("features", features), ("rows", rows), ("splits", splits)):
        if "source_row" not in table or table.source_row.tolist() != list(range(len(features))):
            raise ValueError(f"{name}: 원본 행 ID 누락·중복·순서 오류")
    if set(rows.columns) != {"source_row", "Label", "pattern_id", "role"}:
        raise ValueError("행 대장 컬럼 오류")
    expected_columns = {f"{split}_{seed}" for split in SPLITS for seed in SEEDS}
    if set(splits.columns) != expected_columns | {"source_row"}:
        raise ValueError("외부 분할 6개가 필요합니다")
    frame = encode_frame(features.drop(columns="source_row"))
    if not frame.Label.equals(rows.Label) or not row_roles(frame).equals(rows.role):
        raise ValueError("행 대장의 라벨·용도가 특징 파일의 고정 정책과 다릅니다")
    patterns = pattern_ids(frame)
    if not patterns.equals(rows.pattern_id):
        raise ValueError("시퀀스 패턴이 행 대장과 다릅니다")
    for split in SPLITS:
        for seed in SEEDS:
            actual = rows[["source_row", "Label", "pattern_id"]].assign(
                partition=splits[f"{split}_{seed}"])
            validate_assignment(frame, actual, split)
            expected, _ = build_assignment(frame, patterns, split, seed)
            if not actual.partition.equals(expected.partition):
                raise ValueError(f"저장 배정이 명시된 분할·시드와 다릅니다: {split}/{seed}")
    return frame


def load_inputs(data_dir, preprocess_run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", preprocess_run_id):
        raise ValueError("잘못된 preprocess-run-id")
    folder = data_dir / "processed" / f"track_a_{preprocess_run_id}"
    paths = {name: folder / name for name in ("features.csv", "rows.csv", "splits.csv")}
    paths["raw"] = data_dir / "raw/malicious_tls/malicious_TLS.csv"
    paths["preprocess_metadata"] = ROOT / "outputs/results" / f"preprocess_{preprocess_run_id}" / "metadata.json"
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    audit = json.loads(paths["preprocess_metadata"].read_text(encoding="utf-8"))
    if hashes["raw"] != audit["source_sha256"]:
        raise ValueError("원본 체크섬이 재전처리 기록과 다릅니다")
    features = pd.read_csv(paths["features.csv"], float_precision="round_trip")
    rows = pd.read_csv(paths["rows.csv"])
    splits = pd.read_csv(paths["splits.csv"])
    frame = validate_tables(features, rows, splits)
    raw = encode_frame(pd.read_csv(paths["raw"], low_memory=False))
    pd.testing.assert_frame_equal(frame, raw, check_dtype=False, check_exact=True)
    if len(frame) != audit["source_rows"] or rows.role.value_counts().to_dict() != audit["roles"]:
        raise ValueError("행 수·용도별 행 수가 재전처리 기록과 다릅니다")
    known = sorted(rows.loc[rows.role.eq("known"), "Label"].unique())
    if (len(known) != 13 or "benign" not in known or set(frame.Label) != set(known) | HELD_OUT | EXCLUDED
            or known != audit["known_labels"]):
        raise ValueError("고정 23종·알려진 13종 구성이 아닙니다")
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("읽는 동안 입력 파일이 바뀌었습니다")
    return frame, rows, splits, known, paths, hashes


def code_hashes():
    files = ["src/track_a/feature_group_experiment.py", "src/track_a/preprocess.py",
             "src/common/config.py", "src/common/experiment.py", "src/common/evaluation.py",
             "src/common/models.py", "src/common/results.py", "src/common/splits.py", "uv.lock"]
    return {name: sha256_file(ROOT / name) for name in files}


def run_batch(*, run_id, preprocess_run_id="2026-10-29", workers=2):
    from src.common.config import DATA_DIR, MACHINE

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or workers < 1:
        raise ValueError("run-id 또는 workers 오류")
    data_dir = DATA_DIR.expanduser().resolve()
    if data_dir.is_relative_to(ROOT):
        raise ValueError("DATA_DIR은 저장소 밖이어야 합니다")
    jobs = conditions()
    child_ids = [f"{run_id}_{split}_{model}_{group}_s{seed}"
                 for split, seed, model, group in jobs]
    out = ROOT / "outputs/results" / run_id
    if out.exists() or any((ROOT / "outputs/results" / child).exists()
                           or (data_dir / "predictions" / child).exists() for child in child_ids):
        raise FileExistsError("기존 batch/개별 run-id를 덮어쓰지 않습니다. 새 ID를 사용하세요")
    frame, registry, splits, known, paths, source_hashes = load_inputs(data_dir, preprocess_run_id)
    source_code = code_hashes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    preliminary = MACHINE != "colab-cpu"
    note = ("맥/로컬 예비 결과; 최종 수치는 Colab CPU에서 재실행" if preliminary
            else "Colab CPU 실행; 보고서 사용 전 버전·교차 검증 필요")
    manifest = dict(run_id=run_id, preprocess_run_id=preprocess_run_id, machine=MACHINE,
                    complete=False, expected_runs=48, completed_runs=0,
                    preliminary=preliminary, source_sha256=source_hashes,
                    code_sha256=source_code, git_commit_at_start=commit,
                    git_dirty_at_start=bool(status), run_ids=child_ids,
                    known_labels=known, held_out=sorted(HELD_OUT), excluded=sorted(EXCLUDED),
                    validation="none; fit entire outer train; no tuning or threshold",
                    seed_policy="split_seed=model_seed=0/1/2; split and model variance not separated",
                    std="descriptive sample std ddof=1; overlapping test sets, not independent-resampling CI",
                    note=note)
    out.mkdir(parents=True)

    def checkpoint():
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")

    checkpoint()
    print(f"INPUT PASS: {len(frame)} rows; known={len(known)}; 6 outer assignments verified", flush=True)
    metrics, confusion = [], []
    for index, ((split, seed, model, group), child) in enumerate(zip(jobs, child_ids), start=1):
        # 외부 배정을 그대로 사용한다. 검증 분리·특징 선택·튜닝은 하지 않는다.
        partition = splits[f"{split}_{seed}"]
        result = run_classification(frame[feature_groups()[group]], frame.Label, partition,
                                    known_labels=known, model_name=model, seed=seed, split=split,
                                    groups=registry.pattern_id, workers=workers)
        report, _ = save_run(result, run_id=child, track="A", experiment="feature_group",
                             feature_group=group, split_seed=seed, source_hashes=source_hashes,
                             note=note, extra_metadata=dict(batch_run_id=run_id,
                                 code_sha256=source_code, preliminary=preliminary))
        metrics.append(pd.read_csv(report / "metrics.csv", keep_default_na=False))
        confusion.append(pd.read_csv(report / "confusion.csv", keep_default_na=False))
        pd.concat(metrics, ignore_index=True).to_csv(out / "metrics.csv", index=False)
        pd.concat(confusion, ignore_index=True).to_csv(out / "confusion.csv", index=False)
        manifest["completed_runs"] = index
        checkpoint()
        print(f"{index}/48 {split} s{seed} {model} {group}: "
              f"macro_f1={result.metrics['macro_f1']:.6f} accuracy={result.metrics['accuracy']:.6f} "
              f"fit={result.metrics['train_seconds']:.1f}s", flush=True)
        del result
    if code_hashes() != source_code or any(sha256_file(path) != source_hashes[name]
                                          for name, path in paths.items()):
        raise ValueError("실행 중 코드 또는 입력이 바뀌었습니다. 완성된 실험으로 취급하지 마세요")
    all_metrics = pd.concat(metrics, ignore_index=True)
    summary = summarize_runs(all_metrics)
    if len(all_metrics) != 48 or len(summary) != 16 or not all_metrics.validation_rows.eq(0).all():
        raise ValueError("실험 조합 또는 검증 미분리 계약 오류")
    summary.to_csv(out / "summary.csv", index=False)
    manifest["complete"] = True
    checkpoint()
    print(summary.to_string(index=False), flush=True)
    print("COMPLETE:", out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preprocess-run-id", default="2026-10-29")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_batch(run_id=args.run_id, preprocess_run_id=args.preprocess_run_id, workers=args.workers)


if __name__ == "__main__":
    main()
