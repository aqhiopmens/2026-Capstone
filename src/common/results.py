"""집계는 Git, 행별 자료는 DATA_DIR. 한 실행당 새 디렉터리로 저장한다."""

from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_runs(rows):
    """같은 조건 3회의 평균·표본 표준편차. 중복·미완료·환경 혼합을 거부한다."""
    keys = ["track", "experiment", "split", "feature_group", "model", "machine",
            "sklearn", "lightgbm", "torch"]
    required = keys + ["seed", "split_seed", "macro_f1", "accuracy"]
    if rows.empty or not set(required) <= set(rows.columns):
        raise ValueError("집계에 필요한 결과 컬럼이 없습니다")
    frame = rows.copy()
    frame["torch"] = frame["torch"].fillna("")
    if frame[required].isna().any().any():
        raise ValueError("집계 키·점수에 결측값이 있습니다")
    if not frame[["macro_f1", "accuracy"]].apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("점수는 0~1의 유한값이어야 합니다")
    if frame.duplicated(keys + ["seed", "split_seed"]).any():
        raise ValueError("같은 조건·시드의 중복 결과")
    grouped = frame.groupby(keys, sort=False, dropna=False)
    if not grouped.size().eq(3).all():
        raise ValueError("각 조건 3회가 모인 뒤 집계하세요")
    return grouped.agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                       accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                       runs=("seed", "size")).reset_index()


def save_run(result, *, run_id, track, experiment, feature_group, split_seed,
             source_hashes, note, extra_metadata=None):
    """config의 DATA_DIR/MACHINE을 사용하며 기존 run-id를 덮어쓰지 않는다.

    source_hashes는 특징·행 대장·분할 파일 등의 SHA-256 딕셔너리다.
    분할 시드와 모델 시드를 따로 남기며 모델 본체는 저장하지 않는다.
    """
    from src.common.config import DATA_DIR, MACHINE

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or track not in {"A", "B"}:
        raise ValueError("run-id 또는 track이 잘못되었습니다")
    if not source_hashes or not all(re.fullmatch(r"[0-9a-f]{64}", v) for v in source_hashes.values()):
        raise ValueError("입력 파일의 SHA-256이 필요합니다")
    root = Path(__file__).resolve().parents[2]
    data_root = DATA_DIR.expanduser().resolve()
    if data_root.is_relative_to(root):
        raise ValueError("DATA_DIR을 Git 작업 디렉터리 밖에 두세요")
    report_out = root / "outputs/results" / run_id
    data_out = data_root / "predictions" / run_id
    if report_out.exists() or data_out.exists():
        raise FileExistsError("기존 run-id를 덮어쓰지 않습니다")
    identity = dict(date=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
                    track=track, experiment=experiment, split=result.metadata["split"],
                    feature_group=feature_group, model=result.metadata["model_name"],
                    seed=result.metadata["model_seed"], split_seed=split_seed, machine=MACHINE,
                    sklearn=version("scikit-learn"), lightgbm=version("lightgbm"), torch="", note=note)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    metadata = dict(identity=identity, **result.metadata, source_sha256=source_hashes,
                    git_commit=revision, git_dirty=dirty,
                    numpy=version("numpy"), pandas=version("pandas"),
                    extra=extra_metadata or {},
                    std="three-run descriptive sample std (ddof=1), not an independent-resampling CI")
    # JSON으로 변환할 수 없는 메타데이터로 부분 출력을 만들지 않는다.
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    report_out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    pd.DataFrame([{**identity, **result.metrics}]).to_csv(report_out / "metrics.csv", index=False)
    result.confusion.assign(**identity).to_csv(report_out / "confusion.csv", index=False)
    (report_out / "metadata.json").write_text(encoded, encoding="utf-8")
    result.predictions.to_csv(data_out / "predictions.csv", index=False)
    result.assignments.to_csv(data_out / "assignments.csv", index=False)
    return report_out, data_out
