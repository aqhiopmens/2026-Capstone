"""Colab 집계 ZIP·실행 환경·원본 코드 대응을 학습 없이 검증한다."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ["3.12.13", "2.0.2", "2.2.2", "1.6.1", "4.6.0", "2.11.0"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_export(archive_path, info_path, snapshot_path, source_root=None):
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    source_root = Path(source_root) if source_root else ROOT
    check(info["complete"] and info["machine"] == "colab-cpu", "완료된 Colab 실행이 아닙니다")
    check(info["versions"] == VERSIONS and "202607" in info["colab_release"], "Colab 고정 환경 불일치")
    check(info["upstream_commit"] == snapshot["upstream_commit"], "원본 코드 커밋 대응 오류")
    for name, expected in snapshot["files_sha256"].items():
        check(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == expected,
              f"실행 원본 파일과 현재 파일이 다릅니다: {name}")
    feature, control = info["feature_run_id"], info["control_run_id"]
    allowed_prefixes = (f"outputs/results/{feature}", f"outputs/results/{control}")
    with zipfile.ZipFile(archive_path) as archive:
        check(archive.testzip() is None, "ZIP 파일 CRC 오류")
        names = archive.namelist()
        check(len(names) == len(set(names)), "ZIP 내부 파일명 중복")
        check(len(names) == 212, "66회 집계·메타데이터·감사 파일 212개가 모두 필요합니다")
        for name in names:
            path = PurePosixPath(name)
            check(not path.is_absolute() and ".." not in path.parts and name.startswith(allowed_prefixes),
                  f"허용한 집계 경로가 아닙니다: {name}")
            check(path.suffix in {".csv", ".json"}, "집계 외 파일 형식")
        def read_json(name):
            return json.loads(archive.read(name))
        def read_csv(name):
            frame = pd.read_csv(io.BytesIO(archive.read(name)), keep_default_na=False)
            check(not {"source_row", "row_id", "pattern_id", "max_probability", "lookup_hit"} & set(frame),
                  "행별 원자료가 집계 ZIP에 포함됐습니다")
            return frame
        def digest(name):
            return hashlib.sha256(archive.read(name)).hexdigest()
        tables = {}
        manifests = {}
        class_counts = pd.read_csv(ROOT / "outputs/results/preprocess_2026-10-29/split_class_counts.csv")
        for run_id, expected_runs, expected_summary in ((feature, 48, 16), (control, 18, 6)):
            folder = f"outputs/results/{run_id}"
            manifest = read_json(folder + "/manifest.json")
            check(manifest["complete"] and manifest["completed_runs"] == expected_runs
                  and manifest["machine"] == "colab-cpu" and not manifest["preliminary"], "배치 완료 계약 오류")
            check(manifest["git_commit_at_start"] == info["runtime_snapshot_commit"], "snapshot Git 커밋 불일치")
            manifests[run_id] = manifest
            for name, expected in manifest["code_sha256"].items():
                check(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == expected, f"모델 코드 불일치: {name}")
            metrics, cm, summary = [read_csv(folder + f"/{name}.csv") for name in ("metrics", "confusion", "summary")]
            check(len(metrics) == expected_runs and len(cm) == expected_runs * 169
                  and len(summary) == expected_summary, "집계 파일 조건 수 오류")
            keys = ["split", "seed", "model", "feature_group"]
            check(not metrics.duplicated(keys).any(), "실행 조건 중복")
            check(metrics.validation_rows.eq(0).all() and metrics.seed.eq(metrics.split_seed).all(), "배정 정책 오류")
            check(metrics.machine.eq("colab-cpu").all() and metrics.sklearn.eq(VERSIONS[3]).all()
                  and metrics.lightgbm.eq(VERSIONS[4]).all(), "결과 행의 환경 오류")
            labels = manifest["known_labels"]
            for row in metrics.itertuples(index=False):
                mask = cm.split.eq(row.split) & cm.seed.eq(row.seed) & cm.model.eq(row.model) & cm.feature_group.eq(row.feature_group)
                part = cm.loc[mask]
                check(len(part) == 169 and set(part.actual) == set(labels) == set(part.predicted), "혼동 행렬 클래스 오류")
                matrix = part.pivot(index="actual", columns="predicted", values="count").loc[labels, labels].to_numpy()
                check((matrix >= 0).all() and np.equal(matrix, np.trunc(matrix)).all(), "혼동 행렬 건수 오류")
                check(matrix.sum() == row.test_rows, "시험 행 수 오류")
                expected_counts = class_counts.loc[class_counts.split.eq(row.split) & class_counts.seed.eq(row.seed)
                    & class_counts.partition.eq("test")].set_index("label").loc[labels, "rows"].to_numpy()
                check(np.array_equal(matrix.sum(1), expected_counts), "클래스별 시험 분모가 원본 배정과 다릅니다")
                f1 = (2 * np.diag(matrix) / (matrix.sum(0) + matrix.sum(1))).mean()
                accuracy = np.trace(matrix) / matrix.sum()
                check(np.isclose(f1, row.macro_f1, rtol=0, atol=1e-12)
                      and np.isclose(accuracy, row.accuracy, rtol=0, atol=1e-12), "혼동 행렬 재계산 점수 불일치")
                child = (f"{feature}_{row.split}_{row.model}_{row.feature_group}_s{row.seed}" if run_id == feature
                         else f"{control}_{row.split}_{row.model}_s{row.seed}")
                meta = read_json(f"outputs/results/{child}/metadata.json")
                per_metrics = read_csv(f"outputs/results/{child}/metrics.csv")
                per_cm = read_csv(f"outputs/results/{child}/confusion.csv")
                check(len(per_metrics) == 1 and np.isclose(per_metrics.iloc[0].macro_f1, row.macro_f1, rtol=0, atol=1e-12)
                      and np.isclose(per_metrics.iloc[0].accuracy, row.accuracy, rtol=0, atol=1e-12), "개별·통합 점수 불일치")
                check(np.array_equal(per_cm.pivot(index="actual", columns="predicted", values="count").loc[labels, labels].to_numpy(), matrix),
                      "개별·통합 혼동 행렬 불일치")
                check(meta["git_commit"] == info["runtime_snapshot_commit"] and meta["numpy"] == VERSIONS[1]
                      and meta["pandas"] == VERSIONS[2] and meta["labels"] == labels, "개별 메타데이터 불일치")
                check(meta["source_sha256"] == manifest["source_sha256"], "개별 실행 입력 해시 불일치")
                for key in ("split", "seed", "model", "feature_group"):
                    check(meta["identity"][key] == getattr(row, key), "개별 실행 조건 식별자 불일치")
                if row.model in {"rf", "lgbm"}:
                    params = meta["model_params"]
                    check(params["n_estimators"] == 100 and params["class_weight"] == "balanced"
                          and params["random_state"] == row.seed and params["n_jobs"] == 2, "모델 설정 불일치")
                elif row.split == "pattern_group":
                    check(float(row.lookup_coverage) == 0 and matrix[:, labels.index("benign")].sum() == row.test_rows,
                          "그룹 해시 미조회 fallback 계약 오류")
            for row in summary.itertuples(index=False):
                selected = metrics.loc[metrics.split.eq(row.split) & metrics.model.eq(row.model)
                                       & metrics.feature_group.eq(row.feature_group)]
                check(len(selected) == 3 and set(selected.seed) == {0, 1, 2}, "3시드 집계 누락")
                for score in ("macro_f1", "accuracy"):
                    values = selected[score].to_numpy(dtype=float)
                    check(np.isclose(values.mean(), getattr(row, score + "_mean"), rtol=0, atol=1e-12)
                          and np.isclose(values.std(ddof=1), getattr(row, score + "_std"), rtol=0, atol=1e-12), "평균·편차 불일치")
            tables[run_id] = metrics
        original_hashes = json.loads((ROOT / "outputs/results/feature_group_2026-10-29_mac/manifest.json").read_text())["source_sha256"]
        check(manifests[feature]["source_sha256"] == original_hashes, "맥과 Colab 입력 해시가 다릅니다")
        for name, expected in manifests[control]["source_sha256"].items():
            if name in original_hashes:
                check(expected == original_hashes[name], "대조 실험 입력 해시 불일치")
            else:
                if name == "baseline_manifest": path = f"outputs/results/{feature}/manifest.json"
                elif name == "baseline_metrics": path = f"outputs/results/{feature}/metrics.csv"
                else:
                    split, model, seed = name.removeprefix("baseline_").rsplit("_", 2)
                    path = f"outputs/results/{feature}_{split}_{model}_sequence_s{seed}/metadata.json"
                check(digest(path) == expected, "대조 실험의 Colab 기준선 참조 해시 오류")
        paired = read_csv(f"outputs/results/{control}/paired.csv")
        check(len(paired) == 12, "짝 비교 건수 오류")
        original = tables[feature]
        for row in paired.itertuples(index=False):
            before = original.loc[original.split.eq(row.split) & original.model.eq(row.model)
                                  & original.seed.eq(row.seed) & original.feature_group.eq("sequence")].iloc[0]
            for score in ("macro_f1", "accuracy"):
                check(np.isclose(before[score], getattr(row, score + "_original"), rtol=0, atol=1e-12), "원순서 점수 참조 오류")
                check(np.isclose(getattr(row, score + "_shuffled") - before[score], getattr(row, "delta_" + score), rtol=0, atol=1e-12), "짝 차이 계산 오류")
        mac_hash = pd.read_csv(ROOT / "outputs/results/sequence_controls_2026-10-29_mac/metrics.csv")
        coverage = read_csv(f"outputs/results/{control}/hash_coverage.csv")
        check(len(coverage) == 78 and not coverage.duplicated(["split", "seed", "label"]).any(), "해시 클래스 조회율 건수 오류")
        mac_coverage = pd.read_csv(ROOT / "outputs/results/sequence_controls_2026-10-29_mac/hash_coverage.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(coverage, mac_coverage, check_exact=True)
        for row in tables[control].loc[tables[control].model.eq("hash_rule")].itertuples(index=False):
            before = mac_hash.loc[mac_hash.split.eq(row.split) & mac_hash.seed.eq(row.seed) & mac_hash.model.eq("hash_rule")].iloc[0]
            check(np.isclose(before.macro_f1, row.macro_f1, rtol=0, atol=1e-12)
                  and np.isclose(before.accuracy, row.accuracy, rtol=0, atol=1e-12), "맥·Colab 해시 기준선 불일치")
            selected = coverage.loc[coverage.split.eq(row.split) & coverage.seed.eq(row.seed)]
            check(selected.hit_rows.sum() == float(row.lookup_hit_rows)
                  and np.isclose(selected.hit_rows.sum() / selected.test_rows.sum(), float(row.lookup_coverage), atol=1e-12, rtol=0),
                  "클래스별·전체 해시 조회율 불일치")
        for filename in ("post_shuffle_overlap.csv", "bag_coverage.csv"):
            current = read_csv(f"outputs/results/{control}_audit/{filename}")
            old = pd.read_csv(ROOT / "outputs/results/sequence_controls_2026-10-29_mac_audit" / filename, keep_default_na=False)
            pd.testing.assert_frame_equal(current, old, check_exact=True)
        audited = read_csv(f"outputs/results/{control}_audit/paired_class_scores.csv")
        check(len(audited) == 156 and not audited.duplicated(["split", "seed", "model", "label"]).any(),
              "클래스별 짝 비교 건수 오류")
        for row in audited.itertuples(index=False):
            values = {}
            for kind, run_id in (("original", feature), ("shuffled", control)):
                cm = read_csv(f"outputs/results/{run_id}/confusion.csv")
                part = cm.loc[cm.split.eq(row.split) & cm.seed.eq(row.seed) & cm.model.eq(row.model)]
                if kind == "original": part = part.loc[part.feature_group.eq("sequence")]
                matrix = part.pivot(index="actual", columns="predicted", values="count")
                tp = matrix.loc[row.label, row.label]
                support = matrix.loc[row.label].sum()
                f1 = 2 * tp / (support + matrix[row.label].sum())
                recall = tp / support
                check(support == row.test_rows and np.isclose(f1, getattr(row, kind + "_f1"), atol=1e-12, rtol=0)
                      and np.isclose(recall, getattr(row, kind + "_recall"), atol=1e-12, rtol=0), "클래스별 점수 재계산 불일치")
                values[kind] = f1
            check(np.isclose(values["shuffled"] - values["original"], row.delta_f1, atol=1e-12, rtol=0), "클래스별 차이 오류")
        audit_meta = read_json(f"outputs/results/{control}_audit/metadata.json")
        check(audit_meta["complete"] and not audit_meta["model_training"] and audit_meta["machine"] == "colab-cpu",
              "추가 감사 메타데이터 오류")
        audit_paths = {"confusion": f"outputs/results/{control}/confusion.csv",
                       "original_confusion": f"outputs/results/{feature}/confusion.csv",
                       "manifest": f"outputs/results/{control}/manifest.json",
                       "original_manifest": f"outputs/results/{feature}/manifest.json"}
        for name, expected in audit_meta["source_sha256"].items():
            check((original_hashes[name] if name in original_hashes else digest(audit_paths[name])) == expected,
                  "추가 감사 입력 해시 불일치")
        check(hashlib.sha256((source_root / "src/track_a/audit_sequence_controls.py").read_bytes()).hexdigest()
              == audit_meta["audit_code_sha256"], "추가 감사 코드 불일치")
    return dict(verified=True, runs=66, confusion_cells=11154, summary_conditions=22,
                feature_run_id=feature, control_run_id=control, artifact_files=len(names),
                upstream_commit=info["upstream_commit"], runtime_snapshot_commit=info["runtime_snapshot_commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--run-info", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source-root", help="해당 실행 코드가 보존된 snapshot 폴더. 생략하면 현재 저장소")
    args = parser.parse_args()
    print(json.dumps(verify_export(args.archive, args.run_info, args.snapshot, args.source_root), indent=2))


if __name__ == "__main__":
    main()
