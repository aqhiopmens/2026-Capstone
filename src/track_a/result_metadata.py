"""완료 배치의 개별 메타데이터를 보존하고 중복 집계 파일만 Git에서 제외한다."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = ("metrics.csv", "confusion.csv", "metadata.json")
KEYS = ("split", "seed", "model", "feature_group")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_for(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["run_ids"]
    if (manifest["run_id"] != folder.name or not manifest["complete"]
            or manifest["completed_runs"] != manifest["expected_runs"]
            or len(ids) != manifest["expected_runs"] or len(set(ids)) != len(ids)
            or any(not re.fullmatch(re.escape(folder.name) + r"_[A-Za-z0-9_-]+", child) for child in ids)):
        raise ValueError("완료 배치·고유 run-id가 필요합니다")
    return manifest


def metadata_bytes(entry):
    text = json.dumps(entry["metadata"], ensure_ascii=False, indent=2) + "\n"
    if entry["format"] == "CRLF":
        text = text.replace("\n", "\r\n")
    elif entry["format"] == "raw":
        text = entry["raw_json"]
    elif entry["format"] != "LF":
        raise ValueError("메타데이터 형식 오류")
    raw = text.encode("utf-8")
    if digest(raw) != entry["sha256"] or json.loads(raw) != entry["metadata"]:
        raise ValueError("메타데이터 SHA-256 불일치")
    return raw


def read_archive(folder):
    """아카이브 우선. 없을 때만 호출자가 기존 개별 파일을 사용한다."""
    folder = Path(folder)
    path = folder / "run_metadata.json"
    if not path.exists():
        return None
    manifest = manifest_for(folder)
    archive = json.loads(path.read_text(encoding="utf-8"))
    if (archive["schema_version"] != 1 or archive["run_id"] != folder.name
            or set(archive["runs"]) != set(manifest["run_ids"])
            or set(archive["source_sha256"]) != {"manifest.json", "metrics.csv", "confusion.csv"}
            or any(digest((folder / name).read_bytes()) != sha
                   for name, sha in archive["source_sha256"].items())):
        raise ValueError("아카이브의 배치·집계 SHA-256 불일치")
    for entry in archive["runs"].values():
        metadata_bytes(entry)
    return {child: entry["metadata"] for child, entry in archive["runs"].items()}


def compact_batch(folder):
    folder = Path(folder)
    manifest = manifest_for(folder)
    if read_archive(folder) is not None:
        return folder / "run_metadata.json"
    aggregates = {name: pd.read_csv(folder / name, keep_default_na=False)
                  for name in FILES[:2]}
    runs = {}
    covered = {name: set() for name in aggregates}
    for child in manifest["run_ids"]:
        child_folder = folder.parent / child
        raw = (child_folder / "metadata.json").read_bytes()
        meta = json.loads(raw)
        canonical = (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        entry = dict(metadata=meta, sha256=digest(raw), format="LF")
        if raw != canonical:
            if raw == canonical.replace(b"\n", b"\r\n"):
                entry["format"] = "CRLF"
            else:
                entry.update(format="raw", raw_json=raw.decode("utf-8"))
        metadata_bytes(entry)
        for name, aggregate in aggregates.items():
            mask = pd.Series(True, index=aggregate.index)
            for key in KEYS:
                mask &= aggregate[key].eq(meta["identity"][key])
            selected = aggregate.loc[mask]
            individual = pd.read_csv(child_folder / name, keep_default_na=False)
            if (selected.empty or covered[name].intersection(selected.index)
                    or not set(individual.columns).issubset(aggregate.columns)):
                raise ValueError(f"중복·누락된 조건: {child}/{name}")
            # 해시 룰의 추가 지표는 모델 배치에서 빈 컬럼이 될 수 있다.
            extras = set(aggregate.columns) - set(individual.columns)
            if any(not selected[column].eq("").all() for column in extras):
                raise ValueError(f"개별 결과에 없는 집계 값: {child}/{name}")
            selected = selected[individual.columns].reset_index(drop=True)
            for column in individual.columns:
                try:
                    selected[column] = pd.to_numeric(selected[column])
                    individual[column] = pd.to_numeric(individual[column])
                except (ValueError, TypeError):
                    pass
            pd.testing.assert_frame_equal(selected, individual, check_dtype=False,
                                          check_exact=False, rtol=0, atol=1e-12)
            covered[name].update(aggregate.index[mask])
        runs[child] = entry
    if any(len(covered[name]) != len(frame) for name, frame in aggregates.items()):
        raise ValueError("배치 집계에 미확인 행이 있습니다")
    archive = dict(schema_version=1, run_id=folder.name,
        source_sha256={name: digest((folder / name).read_bytes())
                       for name in ("manifest.json", *FILES[:2])}, runs=runs)
    path = folder / "run_metadata.json"
    path.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    read_archive(folder)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--untrack", action="store_true", help="로컬 파일은 보존하고 Git 추적만 해제")
    args = parser.parse_args()
    folders = []
    for run_id in args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("잘못된 run-id")
        folders.append(ROOT / "outputs/results" / run_id)
    for folder in folders:
        print(compact_batch(folder))
    if args.untrack:
        tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
        paths = [str((folder.parent / child / name).relative_to(ROOT))
                 for folder in folders for child in manifest_for(folder)["run_ids"] for name in FILES]
        paths = [path for path in paths if path in tracked]
        if paths:
            subprocess.run(["git", "rm", "--cached", "--", *paths], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
        print(f"untracked {len(paths)} duplicate files; local files retained")


if __name__ == "__main__":
    main()
