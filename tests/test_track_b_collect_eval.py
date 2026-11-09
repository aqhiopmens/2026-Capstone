import numpy as np
import pandas as pd
import pytest

from src.track_b.collect_eval import boot_summary, build_counts, check_complete, select_columns, split_fold

SERVERS = ("S1", "S2", "S3")


def frame():
    """실제 main_flows와 같은 구조: 서버 3 × (학습 30·시험 10 라운드) × 클래스·구현 칸 10개씩."""
    rows, ep = [], 0
    for sv in SERVERS:
        for phase, rounds in (("train", 30), ("test", 10)):
            for rnd in range(1, rounds + 1):
                for i in range(10):
                    for cls in ("normal", "attack"):
                        for impl in ("curl", "requests"):
                            ep += 1
                            rows.append(dict(server=sv, phase=phase, round=rnd, episode=ep, cls=cls, impl=impl,
                                             start_ts=1000.0 * (rnd + 100 * (phase == "test")) + ep % 7 + 0.001 * ep))
    return pd.DataFrame(rows)


def test_none_condition_is_pure_loso_and_test_set_fixed():
    f = frame()
    s = split_fold(f, "S2", "none", n=10, window=1)
    assert len(s["test"]) == 400 and set(f.loc[s["test"], "server"]) == {"S2"} and set(f.loc[s["test"], "phase"]) == {"test"}
    assert len(s["train"]) == 2400 and "S2" not in set(f.loc[s["train"], "server"]) and set(f.loc[s["train"], "phase"]) == {"train"}
    assert len(s["support"]) == 0


@pytest.mark.parametrize("cond", ["label", "select", "both"])
def test_support_is_first_n_per_class_impl_of_window_first_round_and_disjoint_from_test(cond):
    f = frame()
    s = split_fold(f, "S3", cond, n=10, window=2)  # 창 2 = 학습 라운드 10~18, 지원은 라운드 10에서
    sup = f.loc[s["support"]]
    assert len(sup) == 20 and set(sup.server) == {"S3"} and set(sup.phase) == {"train"} and set(sup["round"]) == {10}
    assert sup.groupby(["cls", "impl"]).size().eq(5).all()
    cell = f[(f.server == "S3") & (f.phase == "train") & (f["round"] == 10) & (f.cls == "attack") & (f.impl == "curl")]
    assert set(sup[(sup.cls == "attack") & (sup.impl == "curl")].index) == set(cell.nsmallest(5, "start_ts").index)
    assert not set(s["support"]) & set(s["test"])


def test_support_added_to_train_only_for_label_conditions_and_test_rows_identical():
    f = frame()
    got = {c: split_fold(f, "S1", c, n=6, window=1) for c in ("none", "label", "select", "both")}
    base = got["none"]["train"]
    assert list(got["select"]["train"]) == list(base)                      # 특징 선택만: 학습에 지원을 넣지 않는다
    for c in ("label", "both"):
        assert set(got[c]["train"]) == set(base) | set(got[c]["support"]) and len(got[c]["support"]) == 12
        assert not set(got[c]["support"]) & set(base)
    assert all(list(got[c]["test"]) == list(got["none"]["test"]) for c in got)  # 시험 행·순서가 모든 조건에서 같다
    assert split_fold(f, "S1", "none", n=6, window=1)["train"].tolist() == base.tolist()


def test_window_three_and_bad_inputs():
    f = frame()
    s = split_fold(f, "S1", "label", n=10, window=3)
    assert set(f.loc[s["support"], "round"]) == {19}
    with pytest.raises(ValueError):
        split_fold(f, "S1", "nope", n=10, window=1)
    with pytest.raises(ValueError):
        split_fold(f, "S1", "label", n=10, window=4)
    dup = pd.concat([f, f.iloc[:1]])
    with pytest.raises(ValueError):  # 행 키 (server, phase, episode) 중복 금지
        split_fold(dup.reset_index(drop=True), "S1", "none", n=10, window=1)


def stump_data():
    y = pd.Series(["normal"] * 50 + ["attack"] * 50)
    flip = np.zeros(100, int); flip[[0, 10, 20, 30, 40, 55, 65, 75, 85, 95]] = 1  # b: 학습에서 10% 어긋남
    a = (y == "attack").astype(int).to_numpy()
    src = pd.DataFrame(dict(a=a, b=a ^ flip, const=7, c=a))               # const는 분산 0, c는 a와 같은 열
    sup_y = pd.Series(["normal", "normal", "attack", "attack"])
    sup = pd.DataFrame(dict(a=[1, 1, 0, 0], b=[0, 0, 1, 1], const=7, c=[1, 1, 0, 0]))  # a·c는 지원에서 뒤집혀 0점, b는 만점
    return src, y, sup, sup_y


def test_select_ranks_by_support_score_and_drops_zero_variance_candidates():
    src, y, sup, sup_y = stump_data()
    out = select_columns(src, y, sup, sup_y, r=1)
    assert out["status"] == "ok" and out["selected"] == ["b"] and out["candidates"] == ["a", "b", "c"]  # const 제외, 지원 점수 1순위
    assert out["tie_at_cutoff"] is False


def test_select_tie_breaks_by_source_accuracy_then_column_order_and_keeps_input_order():
    src, y, sup, sup_y = stump_data()
    sup = pd.DataFrame(dict(a=[0, 0, 1, 1], b=[0, 0, 1, 1], const=7, c=[0, 0, 1, 1]))  # 셋 다 지원 만점
    assert select_columns(src, y, sup, sup_y, r=1)["tie_at_cutoff"] is True    # 1위 a와 2위 c는 점수가 같아 열 번호로 갈린다
    assert select_columns(src, y, sup, sup_y, r=2)["tie_at_cutoff"] is False   # 2위 c(1.0)와 3위 b(0.9)는 점수가 다르다
    assert select_columns(src[["a", "c"]], y, sup[["a", "c"]], sup_y, r=1)["tie_at_cutoff"] is True  # a·c는 점수가 모두 같아 열 번호로 갈림
    assert select_columns(src, y, sup, sup_y, r=1)["selected"] == ["a"]       # 동률: 원학습 정확도(a=c=1.0 > b=0.9) → 열 번호(a<c)
    assert select_columns(src, y, sup, sup_y, r=2)["selected"] == ["a", "c"]
    out = select_columns(src[["c", "b", "a"]], y, sup[["c", "b", "a"]], sup_y, r=2)
    assert out["selected"] == ["c", "a"]                                       # 동률(원학습 1.0)은 열 번호 순 = c, a, 반환은 입력 열 순서


def test_select_too_few_or_zero_candidates():
    src, y, sup, sup_y = stump_data()
    assert select_columns(src, y, sup, sup_y, r=9)["selected"] == ["a", "b", "c"]  # 후보 < r: 전부 유지
    none = select_columns(src[["const"]], y, sup[["const"]], sup_y, r=2)
    assert none["status"] == "unavailable" and none["selected"] == [] and none["candidates"] == []  # 대체 없이 평가 불가


# counts 축: (seed, fold, round, actual, predicted). 폴드 1은 [[10,0],[5,5]], 폴드 2는 [[0,10],[0,10]]를 두 라운드로 나눈 것.
FOLD1 = [[[6, 0], [3, 3]], [[4, 0], [2, 2]]]
FOLD2 = [[[0, 6], [0, 6]], [[0, 4], [0, 4]]]
PERFECT = [[[[5, 0], [0, 5]]] * 2] * 2  # (2 폴드, 2 라운드, 2, 2) 완벽 분류


def test_pooled_f1_differs_from_mean_of_fold_f1_and_identity_resample_gives_point_estimate():
    counts = np.array([[FOLD1, FOLD2]])                                   # (1 seed, 2 folds, 2 rounds, 2, 2)
    identity = np.array([[[0, 1], [0, 1]]])                               # (1 반복, 2 폴드, 2 블록)
    pooled = boot_summary(counts, identity)                               # 합산 [[10,10],[5,15]] → macro-F1 0.6190476 (폴드별 평균은 0.5333)
    assert pooled.shape == (1,) and pooled[0] == pytest.approx(0.6190476, abs=1e-6)
    assert boot_summary(counts)[0] == pytest.approx(0.6190476, abs=1e-6)  # idx 없이 부르면 원 점추정


def test_resample_counts_blocks_with_multiplicity_per_fold():
    counts = np.array([[FOLD1, FOLD2]])
    idx = np.array([[[0, 0], [1, 1]]])                                    # 폴드1은 라운드0 두 번, 폴드2는 라운드1 두 번 → 합산 [[12,8],[6,14]]
    assert boot_summary(counts, idx)[0] == pytest.approx(0.649123, abs=1e-6)


def test_seed_mean_is_taken_after_per_seed_f1_and_paired_difference_of_same_condition_is_zero():
    counts = np.array([[FOLD1, FOLD2], PERFECT])                           # 시드 0: 위 값, 시드 1: 완벽
    idx = np.array([[[0, 1], [0, 1]], [[1, 1], [0, 0]]])
    got = boot_summary(counts, idx)
    assert got[0] == pytest.approx((0.6190476 + 1.0) / 2, abs=1e-6)       # 시드별 F1을 먼저 계산한 뒤 평균
    assert np.all(boot_summary(counts, idx) - boot_summary(counts, idx) == 0)


def test_build_counts_maps_test_round_r_to_index_r_minus_1_and_fixed_fold_order():
    rows = [dict(seed=0, fold="S2", round=3, actual="attack", predicted="normal"),
            dict(seed=0, fold="S1", round=1, actual="normal", predicted="normal"),
            dict(seed=1, fold="S1", round=10, actual="attack", predicted="attack")]
    c = build_counts(pd.DataFrame(rows), seeds=(0, 1), folds=("S1", "S2", "S3"), rounds=10)
    assert c.shape == (2, 3, 10, 2, 2) and c.sum() == 3
    assert c[0, 1, 2, 1, 0] == 1 and c[0, 0, 0, 0, 0] == 1 and c[1, 0, 9, 1, 1] == 1   # 클래스 순서 (normal, attack)


def test_split_fails_loudly_when_a_support_cell_has_too_few_rows():
    f = frame()
    drop = f[(f.server == "S1") & (f.phase == "train") & (f["round"] == 1) & (f.cls == "attack") & (f.impl == "curl")].index[2:]
    with pytest.raises(ValueError):  # 지원 칸이 n/2개보다 적으면 조용히 줄이지 않는다
        split_fold(f.drop(drop), "S1", "label", n=10, window=1)


def test_check_complete_rejects_missing_fold_or_seed_rows():
    ok = pd.DataFrame([dict(seed=s, fold=f) for s in (0, 1) for f in ("S1", "S2") for _ in range(4)])
    check_complete(ok, seeds=(0, 1), folds=("S1", "S2"))
    with pytest.raises(ValueError):  # 한 폴드가 빠지면(평가 불가 폴드) 합산에 조용히 섞이지 않는다
        check_complete(ok[ok.fold != "S2"], seeds=(0, 1), folds=("S1", "S2"))
    with pytest.raises(ValueError):  # 행 수가 셀마다 다르면 거절
        check_complete(ok.iloc[:-1], seeds=(0, 1), folds=("S1", "S2"))
