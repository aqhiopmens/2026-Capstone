"""통제 수집(안건 12) LOSO 평가: 폴드 분할·특징 선택·부트스트랩 집계 (사전 고정 개정 5e).

분할·선택·집계는 공개 함수로 검증하고, 모델 학습·저장은 `run` 쪽에서 기존 `run_classification`을 쓴다.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

CONDITIONS = ("none", "label", "select", "both")
WINDOW_FIRST_ROUND = {1: 1, 2: 10, 3: 19}  # 지원 창(학습 라운드 1~9, 10~18, 19~27)의 첫 라운드
KEY = ["server", "phase", "episode"]


def split_fold(frame, held_out, condition, *, n, window):
    """보류 서버 하나에 대한 (train, support, test) 행 인덱스. n = 클래스당 지원 수(구현당 n/2).

    무지원 원 모델은 보류 서버를 학습에서 완전히 뺀다. 지원은 보류 서버 학습 단계의 창 첫 라운드에서
    start_ts 순으로 클래스×구현마다 앞 n/2개를 고르고, 학습에는 label·both 조건에서만 더한다.
    시험은 보류 서버의 시험 단계 전체이며 모든 조건에서 같다.
    """
    if condition not in CONDITIONS or window not in WINDOW_FIRST_ROUND:
        raise ValueError("알 수 없는 조건 또는 창")
    if frame.duplicated(KEY).any():
        raise ValueError("행 키 (server, phase, episode)가 중복됩니다")
    held = frame.server.eq(held_out)
    train = frame.index[~held & frame.phase.eq("train")]
    test = frame.index[held & frame.phase.eq("test")]
    support = frame.index[:0]
    if condition != "none":
        first = frame[held & frame.phase.eq("train") & frame["round"].eq(WINDOW_FIRST_ROUND[window])]
        support = first.sort_values("start_ts").groupby(["cls", "impl"]).head(n // 2).index
        support = frame.index[frame.index.isin(support)]
        if not frame.loc[support].groupby(["cls", "impl"]).size().eq(n // 2).all() or len(support) != 2 * n:
            raise ValueError("지원 칸(클래스×구현)에 n/2개가 모자랍니다. 그 창을 제외하고 보고하세요")
    if condition in ("label", "both"):
        train = train.append(support)
    return dict(train=train, support=support, test=test)


def split_design(frame, design, *, seed):
    """서버 하나를 빼는 LOSO가 아닌 학습 설계의 (train, test). `mixed`·`confounded`는 정상 600·공격 600행으로 크기를 맞추고
    시험은 세 서버 시험 단계 전체(같은 행·순서). `impl:A>B`는 구현 A의 학습 단계로 학습해 구현 B의 시험 단계로 시험한다."""
    ph = frame.phase
    if design.startswith("impl:") and ">" in design:
        a, b = design[5:].split(">")
        return dict(train=frame.index[ph.eq("train") & frame.impl.eq(a)], test=frame.index[ph.eq("test") & frame.impl.eq(b)])
    if design not in ("mixed", "confounded"):
        raise ValueError("알 수 없는 학습 설계")
    rng = np.random.default_rng(seed)
    train = frame[ph.eq("train")]
    if design == "mixed":
        want = {(sv, im, c): 100 for sv in sorted(train.server.unique()) for im in sorted(train.impl.unique()) for c in ("attack", "normal")}
    else:
        want = {("S1", im, "attack"): 300 for im in ("curl", "requests")} | {(sv, im, "normal"): 150 for sv in ("S2", "S3") for im in ("curl", "requests")}
    pick = [rng.choice(g.index.to_numpy(), want[k], replace=False) for k, g in sorted(train.groupby(["server", "impl", "cls"])) if k in want]
    return dict(train=frame.index[frame.index.isin(np.concatenate(pick))], test=frame.index[ph.eq("test")])


def select_columns(x_src, y_src, x_sup, y_sup, *, r):
    """원 서버 학습 행으로 칸마다 깊이 1 스텀프를 학습하고, 지원 균형 정확도로 상위 r칸을 고른다.

    후보 = x_src에서 분산이 0이 아닌 칸. 동률은 원학습 정확도 → 입력 열 순서. 후보가 r보다 적으면 전부 유지하고,
    0개면 "unavailable"로 표시한다(다른 칸으로 대체하거나 조용히 건너뛰지 않는다). 시험 데이터는 받지 않는다.
    """
    cand = [c for c in x_src.columns if x_src[c].var() > 0]
    ranked = []
    for pos, c in enumerate(cand):
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x_src[[c]], y_src)
        ranked.append((-balanced_accuracy_score(y_sup, stump.predict(x_sup[[c]])), -stump.score(x_src[[c]], y_src), pos, c))
    order = sorted(ranked)
    keep = {c for *_, c in order[:r]}
    tie = len(order) > r and order[r - 1][:2] == order[r][:2]            # r번째와 r+1번째 점수가 같아 열 번호로 갈렸는가
    return dict(status="ok" if cand else "unavailable", candidates=cand, selected=[c for c in cand if c in keep], tie_at_cutoff=tie,
                scores={c: (-sa, -ta) for sa, ta, _, c in ranked})


CLASSES = ("normal", "attack")


def check_complete(d, *, seeds, folds):
    """예측 행이 (시드 × 폴드) 셀마다 같은 수로 모두 있어야 한다(평가 불가 폴드가 합산에 조용히 섞이는 것을 막는다)."""
    size = d.groupby(["seed", "fold"]).size()
    if set(size.index) != {(s, f) for s in seeds for f in folds} or size.nunique() != 1:
        raise ValueError("시드×폴드 셀이 빠졌거나 행 수가 다릅니다(평가 불가 폴드 포함 여부 확인)")


def macro_f1(m):
    """(..., 2, 2) 혼동행렬(행 = 실제, 열 = 예측)의 클래스 평균 F1. 분모가 0이면 그 클래스 F1은 0."""
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    denom = m.sum(axis=-1) + m.sum(axis=-2)           # tp+fn + tp+fp
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(axis=-1)


def build_counts(pred, *, seeds, folds, rounds):
    """행 단위 예측(seed, fold, round, actual, predicted) → counts[seed, fold, round-1, actual, predicted]."""
    c = np.zeros((len(seeds), len(folds), rounds, 2, 2), dtype=int)
    for r in pred.itertuples():
        c[seeds.index(r.seed), folds.index(r.fold), r.round - 1, CLASSES.index(r.actual), CLASSES.index(r.predicted)] += 1
    return c


def boot_summary(counts, idx=None):
    """counts[seed, fold, round, actual, predicted]와 idx[반복, 폴드, 블록](라운드 번호-1)에서 반복별 지표를 낸다.

    폴드마다 idx의 블록을 더하고, 폴드를 합산한 혼동행렬에서 시드별 macro-F1을 계산한 뒤 시드를 평균한다.
    idx가 없으면 모든 라운드를 한 번씩 쓴 원 점추정 1개를 돌려준다.
    """
    seeds, folds, rounds = counts.shape[:3]
    if idx is None:
        idx = np.broadcast_to(np.arange(rounds), (1, folds, rounds))
    pooled = sum(counts[:, f][:, idx[:, f]].sum(axis=2) for f in range(folds))   # (seed, 반복, 2, 2)
    return macro_f1(pooled).mean(axis=0)


# ---- 실행부(단위 테스트 없음: 실제 데이터로 완주·행 키·시험 집합·manifest 일치를 점검한다) ----
import argparse
import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

from src.common.experiment import run_classification
from src.common.results import sha256_file
from src.track_b.early_eval import prefix_features
from src.track_b.mask3 import masked_features

FOLDS, SEEDS, KS = ("S1", "S2", "S3"), (0, 1, 2), (5, 10, 15, 20, 30)
R_BY_K = {5: 2, 10: 4, 15: 6}                     # 특징 선택 유지 칸 수(사전 고정)
BOOT_SEED, BOOT_N, TEST_ROUNDS, DESIGN_SEED = 20261109, 2000, 10, 20261110
FEATURES = {"seq_mask3": lambda f, k: masked_features(f, k, "seq_mask3"), "seq": lambda f, k: prefix_features(f, k, "seq")}
PRIMARY = dict(kind="seq_mask3", k=10, n=10, window=1)
PLAN = [dict(kind="seq_mask3", k=k, cond="none", n=10, window=1) for k in KS] + \
       [dict(PRIMARY, cond=c) for c in ("label", "select", "both")]


BASE = dict(kind="seq_mask3", k=10, n=10, window=1, model="rf")
VARIANTS = [dict(window=2), dict(window=3), dict(n=6), dict(k=5), dict(k=15), dict(kind="seq"), dict(model="lgbm")]  # 한 축씩(개정 5f)
PLAN2 = ([dict(BASE, k=k, cond="none") for k in (5, 10, 15)] + [dict(BASE, kind="seq", cond="none"), dict(BASE, model="lgbm", cond="none")]
         + [dict(BASE, **v, cond=c) for v in VARIANTS for c in ("label", "select", "both")]
         + [dict(BASE, k=k, cond="none", n=0, window=0, design=d) for d in ("mixed", "confounded", "impl:requests>curl", "impl:curl>requests") for k in (5, 10, 15)])


def key_sha(frame, index):
    return hashlib.sha256("\n".join(map(str, sorted(map(tuple, frame.loc[index, KEY].to_numpy())))).encode()).hexdigest()


def fit_predict(frame, x, s, seed, workers, model="rf"):
    idx = s["train"].append(s["test"])
    part = pd.Series("train", index=idx)
    part.loc[s["test"]] = "test"
    res = run_classification(x.loc[idx], frame.cls.loc[idx], part, known_labels=list(CLASSES), model_name=model,
                             seed=seed, split="server_loso", workers=workers)
    pred = res.predictions.set_index("row_id").loc[s["test"]]
    p = res.model.predict_proba(x.loc[s["test"]])[:, list(res.model.classes_).index("attack")]
    return pd.DataFrame(dict(row=s["test"], predicted=pred.predicted.to_numpy(), p_attack=p))


def evaluate(frame, plan, workers=2):
    """plan 항목마다 (LOSO는 폴드별로, 그 밖의 설계는 한 번) 시드별로 학습해 행 단위 예측과 manifest 항목을 돌려준다.
    후보 0열 폴드는 평가 불가로 기록한다. 항목 키: kind, k, cond, n, window, model(기본 rf), design(기본 loso)."""
    preds, manifest = [], []
    for e in plan:
        model, design = e.get("model", "rf"), e.get("design", "loso")
        x = FEATURES[e["kind"]](frame, e["k"])
        splits = ([(f, split_fold(frame, f, e["cond"], n=e["n"], window=e["window"])) for f in FOLDS] if design == "loso"
                  else [("all", split_design(frame, design, seed=DESIGN_SEED))])
        for fold, s in splits:
            support = s.get("support", frame.index[:0])
            gap = float(frame.loc[s["test"], "start_ts"].min() - frame.loc[support, "start_ts"].max()) if len(support) else None
            m = dict(e, model=model, design=design, fold=fold, train_rows=len(s["train"]), support_rows=len(support), test_rows=len(s["test"]),
                     train_sha=key_sha(frame, s["train"]), support_sha=key_sha(frame, support), test_sha=key_sha(frame, s["test"]),
                     support_keys=frame.loc[support, KEY].to_numpy().tolist(), support_last_start_to_test_first_start_s=gap)
            cols = list(x.columns)
            if e["cond"] in ("select", "both"):
                src = s["train"] if e["cond"] == "select" else s["train"].difference(support, sort=False)
                sel = select_columns(x.loc[src], frame.cls.loc[src], x.loc[support], frame.cls.loc[support], r=R_BY_K[e["k"]])
                m.update(status=sel["status"], candidates=sel["candidates"], selected=sel["selected"], stump_scores=sel["scores"])
                if sel["status"] == "unavailable":
                    manifest.append(m)
                    continue
                cols = sel["selected"]
            manifest.append(m)
            for seed in SEEDS:
                p = fit_predict(frame, x[cols], s, seed, workers, model)
                base = frame.loc[p.row, KEY + ["round", "cls", "ep_type", "impl"]].reset_index(drop=True)
                preds.append(pd.concat([base, p.drop(columns="row").reset_index(drop=True)], axis=1)
                             .assign(condition=e["cond"], kind=e["kind"], k=e["k"], n=e["n"], window=e["window"], seed=seed, model=model, design=design,
                                     fold=base.server)  # 폴드 = 시험 행의 서버(LOSO에서는 보류 서버와 같다)
                             .rename(columns={"cls": "actual"}))
    return pd.concat(preds, ignore_index=True), manifest


def wilson_upper(x, n, z=1.96):
    p = x / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


GROUP = ["design", "model", "kind", "k", "condition", "n", "window"]


def summarize(pred, idx):
    """설정(GROUP)별 합산 macro-F1·CI·정상 FP·공격 recall, 같은 (모델, 입력, k)의 무지원 대비 paired 차이(LOSO만), 폴드별 값,
    칸별 값, 부분집합 진단. 무지원은 지원 N·창이 바뀌어도 재사용한다."""
    counts = lambda d: build_counts(d, seeds=list(SEEDS), folds=list(FOLDS), rounds=TEST_ROUNDS)
    groups = {key: d for key, d in pred.groupby(GROUP)}
    none = {(m, kd, k): d for (des, m, kd, k, c, n, w), d in groups.items() if des == "loso" and c == "none"}
    mixed = {(m, kd, k): d for (des, m, kd, k, c, n, w), d in groups.items() if des == "mixed"}
    rows, by_fold = [], []
    for key, d in groups.items():
        design, model, kind, k, cond, n, window = key
        check_complete(d, seeds=SEEDS, folds=FOLDS)
        cd = counts(d)
        dist, point = boot_summary(cd, idx), boot_summary(cd)[0]
        fp = [int(((g.actual == "normal") & (g.predicted == "attack")).sum()) for _, g in d.groupby("seed")]
        nn = int((d[d.seed == SEEDS[0]].actual == "normal").sum())  # 폴드 합산 정상 행(시드 하나 기준)
        rec = [float(((g.actual == "attack") & (g.predicted == "attack")).sum() / (g.actual == "attack").sum()) for _, g in d.groupby("seed")]
        row = dict(zip(GROUP, key), macro_f1=point, ci_lo=np.percentile(dist, 2.5), ci_hi=np.percentile(dist, 97.5), normal_fp_by_seed=fp,
                   normal_rows_pooled=nn, fp_wilson_upper_max_pooled=max(wilson_upper(x, nn) for x in fp), attack_recall_by_seed=rec)
        if design == "loso" and cond != "none" and (model, kind, k) in none:
            nc = counts(none[(model, kind, k)])
            diff = dist - boot_summary(nc, idx)
            row.update(diff_vs_none=float(point - boot_summary(nc)[0]), diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5))
        if design == "confounded" and (model, kind, k) in mixed:  # 4.1: 같은 시험 행·같은 재표집에서 교락 − 혼합 paired 차이
            mc = counts(mixed[(model, kind, k)])
            diff = dist - boot_summary(mc, idx)
            row.update(diff_vs_mixed=float(point - boot_summary(mc)[0]), diff_mixed_lo=np.percentile(diff, 2.5), diff_mixed_hi=np.percentile(diff, 97.5))
        rows.append(row)
        for i, fold in enumerate(FOLDS):  # 폴드별 기술 보고(판정 아님)
            fb = boot_summary(cd[:, [i]], idx[:, [i]])
            by_fold.append(dict(zip(GROUP, key), fold=fold, macro_f1=boot_summary(cd[:, [i]])[0], ci_lo=np.percentile(fb, 2.5), ci_hi=np.percentile(fb, 97.5)))
    cells = (pred.assign(pred_attack=pred.predicted.eq("attack")).groupby(GROUP + ["server", "impl", "actual", "seed"])
             .agg(rows=("pred_attack", "size"), predicted_attack=("pred_attack", "sum")).reset_index())
    sub = []
    for key, g0 in pred.groupby(GROUP):
        for name, nm, at in (("failonly", "failonly", "fail"), ("none_ep", "none", "none")):
            g = g0[((g0.actual == "normal") & (g0.ep_type == nm)) | ((g0.actual == "attack") & (g0.ep_type == at))]
            for seed, h in g.groupby("seed"):
                sub.append(dict(zip(GROUP, key), subset=name, seed=seed, normal_rows=int((h.actual == "normal").sum()),
                                normal_fp=int(((h.actual == "normal") & (h.predicted == "attack")).sum()), attack_rows=int((h.actual == "attack").sum()),
                                attack_hit=int(((h.actual == "attack") & (h.predicted == "attack")).sum())))
    return pd.DataFrame(rows), pd.DataFrame(by_fold), cells, pd.DataFrame(sub)


def latency(frame):
    """k별·서버별·단계별·클래스별 관측 대기시간(연결 시작→k번째 payload 패킷). 도달한 연결만 대기시간을 갖고, 도달·미도달 수를 함께 낸다.
    실제로 판정하는 행의 분모는 phase == "test"."""
    out = []
    for (sv, ph, cls), g in frame.groupby(["server", "phase", "cls"]):
        for k in (5, 10, 15):
            t = g[f"t_k{k}"].dropna()
            out.append(dict(server=sv, phase=ph, cls=cls, k=k, rows=len(g), reached=len(t), not_reached=len(g) - len(t),
                            median_s=float(t.median()) if len(t) else None, p90_s=float(t.quantile(0.9)) if len(t) else None))
    return pd.DataFrame(out)


def exposure(frame):
    """서버(폴드)×구현×k별 시험 단계 정상 행의 -152 노출률 p_k, 이진 응답 단서 기준선(4.0b), payload가 k개에 도달한 비율."""
    t = frame[frame.phase.eq("test")]
    out = []
    for sv in FOLDS:
        for impl in (None, "curl", "requests"):
            ts = t[(t.server == sv) & (impl is None or t.impl == impl)]
            tn = ts[ts.cls == "normal"]
            seq = tn[[f"seq_{i}" for i in range(30)]].to_numpy()
            for k in KS:
                p = float((seq[:, :k] == -152).any(axis=1).mean())
                out.append(dict(fold=sv, impl=impl or "all", k=k, normal_rows=len(tn), p_k=p, ideal_balanced_acc=0.5 + 0.5 * p,
                                tpr_limit_fpr1=min(1.0, 0.01 / (1 - p)) if p < 1 else 1.0, reach=float((ts.payload_packets >= k).mean())))
    return pd.DataFrame(out)


def main():
    from src.common.config import DATA_DIR

    ap = argparse.ArgumentParser(description="통제 수집 LOSO 평가(개정 5f, --stage 1: 주 대비, 2: 탐색)")
    ap.add_argument("--flows", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out, data_out = root / "outputs/results" / f"track_b_collect_eval_{args.run_id}", DATA_DIR / "predictions" / f"track_b_collect_eval_{args.run_id}"
    if out.exists() or data_out.exists():
        raise FileExistsError("기존 결과를 덮어쓰지 않습니다. 다른 run-id를 지정하세요")
    frame = pd.read_csv(args.flows)
    frame = frame[frame.subset.eq("main")].reset_index(drop=True)
    plan = PLAN if args.stage == 1 else PLAN2
    pred, manifest = evaluate(frame, plan, args.workers)
    idx = np.random.default_rng(BOOT_SEED).integers(0, TEST_ROUNDS, (BOOT_N, len(FOLDS), TEST_ROUNDS))
    summary, by_fold, cells, subsets = summarize(pred, idx)
    n_loso = int(summary["diff_vs_none"].notna().sum()) if "diff_vs_none" in summary else 0
    n_design = int(summary["diff_vs_mixed"].notna().sum()) if "diff_vs_mixed" in summary else 0
    comparisons = dict(stage=args.stage, primary_contrasts=n_loso if args.stage == 1 else 0, exploratory_loso_contrasts=0 if args.stage == 1 else n_loso,
                       exploratory_design_paired=n_design, no_multiplicity_correction=True,
                       not_run=["4.4 단계적 정책: 검증 폴드 설계·누적 FPR 목표 미고정"] + (["2단계 탐색: 별도 run으로 실행"] if args.stage == 1 else []))
    git = lambda *a: subprocess.check_output(["git", *a], cwd=root, text=True, encoding="utf-8").strip()
    meta = dict(date=str(date.today()), machine=os.environ.get("MACHINE", "unknown"), git_commit=git("rev-parse", "HEAD"),
                git_dirty_src_tests=bool(git("status", "--porcelain", "--", "src", "tests")), flows_sha256=sha256_file(args.flows),
                code_sha256={f: sha256_file(root / f) for f in ("src/track_b/collect_eval.py", "src/common/experiment.py", "src/common/models.py",
                                                            "src/track_b/mask3.py", "src/track_b/early_eval.py")},
                comparisons=comparisons, boot_index_sha256=hashlib.sha256(idx.tobytes()).hexdigest(), boot_seed=BOOT_SEED,
                plan=plan, stage=args.stage, design_seed=DESIGN_SEED, seeds=list(SEEDS), unavailable=[m for m in manifest if m.get("status") == "unavailable"], manifest=manifest)
    out.mkdir(parents=True)
    data_out.mkdir(parents=True)
    pred.to_csv(data_out / "predictions.csv.gz", index=False)
    (data_out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    summary.to_csv(out / "summary_by_condition_k.csv", index=False)
    by_fold.to_csv(out / "by_fold.csv", index=False)
    cells.to_csv(out / "cells_impl_class.csv", index=False)
    subsets.to_csv(out / "subsets.csv", index=False)
    exposure(frame).to_csv(out / "exposure_by_fold.csv", index=False)
    latency(frame).to_csv(out / "latency_by_k.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
