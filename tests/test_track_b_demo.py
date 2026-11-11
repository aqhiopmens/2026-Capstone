import numpy as np
import pandas as pd
import pytest

from src.track_b.demo_model import K, class_probs, compare_keys_and_labels, predict_names
from src.track_b.four_class import CLASSES, SEQ, features


def tiny_model():
    from src.common.models import make_model

    rng = np.random.default_rng(0)
    labels = np.array(["benign", "probing", "bruteforce_xml", "bruteforce_https"] * 10)
    x = pd.DataFrame(rng.integers(-300, 300, (40, 30)), columns=SEQ)
    x.iloc[:, 0] = pd.Series(labels).map({"benign": 10, "probing": 20, "bruteforce_xml": 30, "bruteforce_https": 40}).to_numpy()
    return make_model("rf", seed=0, workers=1).fit(x, labels), x, labels


def test_probability_columns_follow_model_classes_not_display_order():
    model, x, labels = tiny_model()
    p = class_probs(model, x)
    assert list(p.columns) == list(model.classes_) != list(CLASSES)               # sklearn 사전순과 표시 순서(CLASSES)는 다르다
    assert list(model.classes_) == sorted(CLASSES)
    assert (predict_names(model, x) == model.predict(x)).all()                   # argmax 이름 = predict
    assert (p.idxmax(axis=1).to_numpy() == labels).mean() > 0.9                  # 열 이름이 실제 클래스에 연결됨
    assert np.allclose(p.sum(axis=1), 1.0)


def test_demo_input_equals_four_class_features_for_same_rows():
    rng = np.random.default_rng(1)
    flows = pd.DataFrame(rng.integers(-300, 300, (5, 30)), columns=SEQ)
    x = features(flows, K, "seq")
    assert K == 10 and (x.iloc[:, K:] == 0).all().all() and (x.iloc[:, :K].to_numpy() == flows.iloc[:, :K].to_numpy()).all()  # 앞 10칸만, 나머지 0


def test_compare_keys_and_labels_requires_same_keys_without_duplicates():
    keys = pd.DataFrame(dict(pcap=["a", "a", "b"], flow_id=[1, 2, 1]))
    ours = keys.assign(predicted=["x", "y", "z"])
    saved = keys.assign(predicted=["x", "y", "w"])
    out = compare_keys_and_labels(ours, saved)
    assert out == dict(keys_equal=True, rows=3, label_match=2, duplicates=0)
    out = compare_keys_and_labels(ours.iloc[:2], saved)                          # 키 집합이 다르면 라벨 일치 수를 믿지 않는다
    assert out["keys_equal"] is False
    dup = pd.concat([ours, ours.iloc[:1]])
    assert compare_keys_and_labels(dup, saved)["duplicates"] == 1
