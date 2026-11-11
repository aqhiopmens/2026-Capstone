"""시연 CLI: 시연 pcap(+대응표)을 저장된 모델로 분석해 연결 표와 채점 요약을 출력한다. Streamlit 화면과 같은 함수를 쓴다.

Run: python -m src.track_b.demo_run --pcap https_0416 --model 2026-11-11_v1
"""

import argparse
import json

import pandas as pd

from src.track_b.demo_infer import analyze_pcap, file_sha256, match_to_original, score_connections, scoring_capacity
from src.track_b.demo_model import load_model


def run_demo(pcap_path, model_dir):
    model, meta = load_model(model_dir)
    conn, timing = analyze_pcap(pcap_path, model)
    sha = file_sha256(pcap_path)
    table = pd.read_csv(str(pcap_path) + ".match.csv", keep_default_na=False)
    conn = match_to_original(conn, table, sha)
    return conn, score_connections(conn), timing, meta, sha


def main():
    from src.common.config import DATA_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcap", required=True, help="DATA_DIR/demo_pcaps 아래 이름(확장자 제외)")
    ap.add_argument("--model", required=True, help="DATA_DIR/models/track_b_demo_<ID>의 ID")
    ap.add_argument("--capacity-only", action="store_true", help="예측을 보기 전에 그룹별 실제 채점 가능 수만 출력")
    args = ap.parse_args()
    conn, score, timing, meta, sha = run_demo(DATA_DIR / "demo_pcaps" / f"{args.pcap}.pcap", DATA_DIR / "models" / f"track_b_demo_{args.model}")
    if args.capacity_only:
        print(scoring_capacity(conn).to_string(index=False))
        print(conn.status.value_counts().to_string())
        return
    out = {k: v for k, v in score.items() if k != "confusion"}
    print(json.dumps(dict(pcap_sha256=sha, model_run_id=meta["run_id"], score=out, timing={k: v for k, v in timing.items() if k != "skipped"}), ensure_ascii=False, indent=1))
    print(score["confusion"].to_string())
    print(conn.status.value_counts().to_string())


if __name__ == "__main__":
    main()
