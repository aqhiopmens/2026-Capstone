"""시연 모델: 4클래스 평가 v2와 같은 학습 조건(원본 seq, k=10, RF 시드 0, workers=2)으로 학습·저장·로드한다.

확률 열은 `model.classes_`(사전순)를 따른다. `CLASSES`는 화면 표시·혼동 행렬 순서일 뿐이다. 재학습하면 항상 새 평가 ID로 기록하고,
v2 저장 예측과 키·라벨이 같을 때만 기존 점수를 인용한다.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.models import make_model
from src.common.results import sha256_file
from src.track_b.four_class import CLASSES, COLS, SEQ, XML, assign_parts, features

K, KIND, SEED, WORKERS = 10, "seq", 0, 2
V2 = "2026-11-08_v2"
INPUTS = dict(https="processed/track_b_https_2026-11-02_v1/flows.csv", xml="processed/track_b_2026-10-31_v3/flows.csv",
              probing="processed/track_b_https_probing_2026-11-08_v1/flows.csv")


def load_data(data_dir):
    """four_class.main과 같은 입력·필터·행 순서: payload>0, XML 라벨 치환, assign_parts, reset_index."""
    load = lambda name: pd.read_csv(Path(data_dir) / INPUTS[name], usecols=COLS, keep_default_na=False).query("payload_packets > 0")
    https, xml, probing = load("https"), load("xml"), load("probing")
    xml = xml.assign(label=xml.label.replace({"bruteforce_xml": XML}))
    data, boundary = assign_parts(https, xml, probing)
    return data.reset_index(drop=True), boundary


def train_demo_model(data):
    train = data[data.part == "train"]
    return make_model("rf", seed=SEED, workers=WORKERS).fit(features(train, K, KIND), train.label.to_numpy())


def class_probs(model, x):
    return pd.DataFrame(model.predict_proba(x), columns=list(model.classes_), index=x.index)


def predict_names(model, x):
    """확률 argmax의 클래스 이름(동률은 `classes_` 순서). `predict`와 같다."""
    return np.asarray(model.classes_)[model.predict_proba(x).argmax(axis=1)]


def compare_keys_and_labels(ours, saved):
    """(pcap, flow_id) 키 집합이 중복 없이 같을 때만 라벨 일치 수를 센다. 키가 다르면 일치 수는 0."""
    key = ["pcap", "flow_id"]
    dup = int(ours.duplicated(key).sum())
    keys_equal = dup == 0 and not saved.duplicated(key).any() and set(map(tuple, ours[key].to_numpy())) == set(map(tuple, saved[key].to_numpy()))
    m = ours.merge(saved, on=key, suffixes=("", "_saved")) if keys_equal else ours.iloc[:0].assign(predicted_saved=[])
    return dict(keys_equal=bool(keys_equal), rows=len(ours), label_match=int((m.predicted == m.predicted_saved).sum()), duplicates=dup)


def save_model(model, out_dir, run_id, data, boundary, data_dir, comparison):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    import joblib
    import sklearn

    joblib.dump(model, out / "model.joblib")
    train = data[data.part == "train"]
    keys = "\n".join(sorted(f"{p}|{f}" for p, f in zip(train.pcap, train.flow_id)))
    meta = dict(run_id=run_id, source_eval_id=f"track_b_four_class_{V2}", model="rf", seed=SEED, workers=WORKERS, kind=KIND, k=K,
                feature_columns=SEQ, display_classes=list(CLASSES), model_classes=list(map(str, model.classes_)), train_rows=len(train),
                train_keys_sha256=hashlib.sha256(keys.encode()).hexdigest(), probing_boundary_ts=boundary,
                inputs={n: dict(path=p, sha256=sha256_file(Path(data_dir) / p)) for n, p in INPUTS.items()},
                code_sha256={"src/track_b/demo_model.py": sha256_file(Path(__file__)), "src/track_b/four_class.py": sha256_file(Path(__file__).with_name("four_class.py"))},
                v2_comparison=comparison, environment=dict(python=platform.python_version(), platform=platform.platform(), sklearn=sklearn.__version__,
                                                           numpy=np.__version__, pandas=pd.__version__))
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_model(model_dir):
    import joblib
    import sklearn

    meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    if meta["environment"]["sklearn"] != sklearn.__version__:
        print(f"경고: 학습 sklearn {meta['environment']['sklearn']} != 현재 {sklearn.__version__}. 예측이 달라질 수 있다")
    model = joblib.load(Path(model_dir) / "model.joblib")
    assert list(map(str, model.classes_)) == meta["model_classes"], "classes_ 순서가 메타데이터와 다르다"
    return model, meta


def main():
    import argparse

    from src.common.config import DATA_DIR

    ap = argparse.ArgumentParser(description="시연 모델(RF 시드 0, 원본 seq, k=10) 학습·저장, v2 저장 예측과 키·라벨 대조")
    ap.add_argument("--run-id", required=True, help="새 평가 ID(재학습은 항상 새 ID). 예: 2026-11-11_v1")
    args = ap.parse_args()
    out = DATA_DIR / "models" / f"track_b_demo_{args.run_id}"
    data, boundary = load_data(DATA_DIR)
    model = train_demo_model(data)
    test = data[data.part == "test"]
    ours = pd.DataFrame(dict(pcap=test.pcap.to_numpy(), flow_id=test.flow_id.to_numpy(), predicted=predict_names(model, features(test, K, KIND))))
    saved = pd.read_csv(DATA_DIR / "predictions" / f"track_b_four_class_{V2}" / "predictions.csv.gz", keep_default_na=False)
    saved = saved[(saved.model == "rf") & (saved.kind == KIND) & (saved.k == K) & (saved.seed == SEED)][["pcap", "flow_id", "predicted"]]
    cmp = compare_keys_and_labels(ours, saved)
    meta = save_model(model, out, args.run_id, data, boundary, DATA_DIR, cmp)
    print(json.dumps(dict(out=str(out), classes_=meta["model_classes"], train_rows=meta["train_rows"], v2_comparison=cmp), ensure_ascii=False))


if __name__ == "__main__":
    main()
