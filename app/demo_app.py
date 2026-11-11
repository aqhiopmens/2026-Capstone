"""시연 대시보드(Streamlit): 저장된 pcap 분석 · 판정 시점 표시. 실행: streamlit run app/demo_app.py (DATA_DIR 환경변수 필요).

모델은 4클래스 RF(원본 seq, k=10, 시드 0). 화면 순서: 자료 선택 / 연결 목록 / 연결 상세 / 평가 요약.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.common.config import DATA_DIR
from src.track_b.demo_infer import analyze_pcap, file_sha256, match_to_original, score_connections
from src.track_b.demo_model import K, V2, load_model
from src.track_b.four_class import CLASSES

ROOT = Path(__file__).resolve().parents[1]
STATUS_KR = {"reached_k": "10개 관측 후 판정", "short_closed_padded": "10개 미도달·종료 확인 후 0패딩 판정", "short_unconfirmed": "10개 미도달·종료 미확인(판정 보류)",
             "short_late_payload_excluded": "10개 미도달·종료 확인 뒤에도 payload 관측(채점 제외)",
             "midstream_excluded": "중간 시작(SYN 없음)·채점 제외"}
SOURCE_KR = {"https_0416": "HTTPS 04-16 (알려진 피해자 .242.104 + 정상)", "https_0416_new_victim": "HTTPS 04-16 새 피해자 .88.81 (한계 사례)", "xml_0417": "XML 04-17",
             "probing_test": "Probing 시험 구간"}
NAME_KR = {"benign": "정상", "probing": "Probing", "bruteforce_xml": "Bruteforce-XML", "bruteforce_https": "Bruteforce-HTTPS"}

st.set_page_config(page_title="저장된 pcap 분석 · 판정 시점 표시", layout="wide")
st.title("저장된 pcap 분석 · 판정 시점 표시")
st.warning("저장된 pcap을 분석한 결과이며 **실시간 운영 지연이 아닙니다**. 검증된 4클래스(정상·Probing·XML·HTTPS) 범주만 표시하며 **학습에 없던 공격 유형·새 피해자에 대한 탐지를 보증하지 않습니다**. 판정은 연결마다 앞 10개 payload 패킷 길이만 사용합니다.")


@st.cache_resource
def get_model(model_dir):
    return load_model(model_dir)


@st.cache_data(show_spinner="pcap 분석 중…")
def analyze(path, sha, model_dir, with_labels):
    model, _ = get_model(model_dir)
    conn, timing = analyze_pcap(path, model)
    if with_labels:
        conn = match_to_original(conn, pd.read_csv(path + ".match.csv", keep_default_na=False), sha)
    return conn, timing


demo_dir, model_root = DATA_DIR / "demo_pcaps", DATA_DIR / "models"
has_inputs = lambda p: Path(str(p) + ".match.csv").exists() and "orig_seq_9" in pd.read_csv(str(p) + ".match.csv", nrows=0).columns  # 입력 일치 검사가 가능한 대응표만
pcaps = sorted(p.stem for p in demo_dir.glob("*.pcap") if has_inputs(p)) if demo_dir.exists() else []
models = sorted(p.name for p in model_root.glob("track_b_demo_*")) if model_root.exists() else []
t1, t2, t3, t4 = st.tabs(["① 자료 선택", "② 연결 목록", "③ 연결 상세", "④ 평가 요약"])

with t1:
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("모델(저장된 시연 모델)", models) if models else None
    source = c2.radio("자료", ["시연 pcap(정답 대응표 있음)", "다른 pcap 올리기(정답 없음)"], horizontal=True)
    path, with_labels = None, False
    if source.startswith("시연") and pcaps:
        choice = st.selectbox("시연 pcap", pcaps, index=next((i for i, n in enumerate(pcaps) if n.endswith("_64")), 0))
        path, with_labels = str(demo_dir / f"{choice}.pcap"), True
        st.caption("시연 pcap은 시험 캡처의 시험 구간 시작점부터 같은 크기(64 MiB)를 규칙대로 자른 것입니다. N은 모델 예측을 보지 않고 라벨 구성만으로 정했으며, 처음 만든 HTTPS 앞 10 MiB를 먼저 확인한 뒤 고정한 **시연용으로 추가 구성한 사례**입니다. 정답 라벨은 ④평가 요약 채점에만 쓰고 모델 입력에서 제외합니다.")
    elif source.startswith("다른"):
        up = st.file_uploader("classic libpcap(.pcap)", type=["pcap"])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                tmp.write(up.getvalue())
            path = tmp.name
    if model_name:
        meta = load_model(model_root / model_name)[1]
        st.markdown(f"**모델** `{meta['run_id']}` · 원 평가 `{meta['source_eval_id']}` · RF 시드 {meta['seed']} · k={meta['k']} · 입력 원본 `seq`  \n"
                    f"확률 열 순서(`classes_`): {meta['model_classes']} / 화면 표시 순서: {meta['display_classes']}  \n"
                    f"v2 저장 예측 대조: 키 {'일치' if meta['v2_comparison']['keys_equal'] else '불일치'}, 라벨 {meta['v2_comparison']['label_match']:,}/{meta['v2_comparison']['rows']:,}")

if path and model_name:
    sha = file_sha256(path)
    conn, timing = analyze(path, sha, str(model_root / model_name), with_labels)
    if conn.empty:
        for t in (t2, t3, t4):
            t.info("분석할 연결이 없습니다(payload가 있는 443 연결 0개).")
        st.stop()
    show = conn.assign(상태=conn.status.map(STATUS_KR), 판정=conn.predicted.map(NAME_KR)).assign(
        확률=conn[[c for c in conn.columns if c.startswith("p_")]].max(axis=1))
    with t2:
        st.caption(f"전체 연결(payload>0) {len(conn):,}개 · pcap SHA `{sha[:12]}…`")
        pick = st.multiselect("상태 필터", list(STATUS_KR.values()), default=list(STATUS_KR.values()))
        cols = ["client_ip", "client_port", "server_ip", "상태", "판정", "확률", "observation_wait_s", "payload_packets"] + (["actual"] if with_labels else [])
        st.dataframe(show[show.상태.isin(pick)][cols].rename(columns={"observation_wait_s": "관측 대기시간(s)"}), width="stretch", height=420)
    with t3:
        i = st.number_input("연결 번호(연결 목록의 행 번호)", 0, len(conn) - 1, 0)
        r = conn.iloc[int(i)]
        st.write(f"**{r.client_ip}:{r.client_port} → {r.server_ip}:443** · 상태: {STATUS_KR[r.status]}")
        a, b = st.columns(2)
        a.caption("앞 10개 payload 길이(부호=방향, 모델이 본 값)")
        a.bar_chart(pd.Series([r[f"len_{j}"] for j in range(K)], index=range(1, K + 1)))
        probs = pd.Series({NAME_KR[c]: r[f"p_{c}"] for c in CLASSES}) if r.status in ("reached_k", "short_closed_padded") else None
        b.caption("4클래스 확률")
        if probs is not None:
            b.bar_chart(probs)
        else:
            b.info("판정 보류 또는 채점 제외 연결입니다(확률 없음).")
        w = r.observation_wait_s
        st.write(f"관측 대기시간(패킷 시각 기준): {'-' if pd.isna(w) else f'{w:.3f} 초'} · 판정 결과: {NAME_KR.get(r.predicted, '-')}"
                 + (f" · 정답: {NAME_KR.get(r.actual, r.actual)}" if with_labels and isinstance(r.actual, str) else ""))
    with t4:
        left, right = st.columns(2)
        with left:
            st.subheader("업로드한 pcap 채점 결과(시드 0)")
            if with_labels:
                s = score_connections(conn)
                st.write(f"전체 연결 {s['total']} · 정답 매칭 {s['matched']} · **실제 채점 {s['scored']}** (분모는 payload>0 연결 기준)  \n"
                         f"제외: 중간 시작 {s['excluded_midstream']}(그중 대응표에 없어 매칭 실패 {s['excluded_midstream_unmatched']}, 합계에 중복 계산하지 않음) · 종료 미확인 {s['excluded_short_unconfirmed']} · "
                         f"종료 뒤 payload 혼입 {s['excluded_late_payload']} · 그 밖의 매칭 실패 {s['unmatched'] - s['excluded_midstream_unmatched']} · 후보 여러 개 {s['ambiguous']} · 입력 불일치 {s['excluded_input_mismatch']}  \n"
                         f"10개 미도달 {s['short_total']}개({s['short_ratio']:.1%}) 중 종료 확인 후 0패딩 판정 {s['short_closed_padded']}개")
                st.write(f"정상 오탐 {s['benign_false_alarms']} / 정상 채점 {s['benign_scored']}")
                st.dataframe(s["confusion"].rename(index=NAME_KR, columns=NAME_KR))
                if "source" in conn:
                    ok = conn[conn.status.isin(("reached_k", "short_closed_padded")) & (conn.match == "matched") & conn.input_match]
                    by = ok.assign(맞힘=ok.predicted == ok.actual).groupby(["source", "actual"]).agg(채점=("맞힘", "size"), 맞힘=("맞힘", "sum")).reset_index()
                    by["재현율"] = (by.맞힘 / by.채점).round(3)
                    by["그룹"] = by.source.map(SOURCE_KR).fillna(by.source)
                    st.caption("그룹별 채점(HTTPS는 피해자별). 새 피해자 .88.81은 학습에 없던 한계 사례입니다.")
                    st.dataframe(by[["그룹", "actual", "채점", "맞힘", "재현율"]].rename(columns={"actual": "실제"}).assign(실제=lambda d: d.실제.map(NAME_KR)), hide_index=True)
                st.caption("행 = 실제, 열 = 예측. 이 시연 자료의 결과이며 연구 전체 성능이 아닙니다.")
            else:
                st.info("정답 대응표가 없는 pcap이라 채점하지 않습니다.")
        with right:
            st.subheader(f"기존 평가 요약(평가 ID {V2}, 전체 시험 3시드 평균)")
            f = ROOT / "outputs/results" / f"track_b_four_class_{V2}" / "summary.csv"
            if f.exists():
                row = pd.read_csv(f).query("kind == 'seq' and k == 10 and model == 'rf'").iloc[0]
                st.metric("macro-F1(전체 시험)", f"{row.macro_f1:.4f}")
                st.write({NAME_KR[c]: round(float(row[f"recall_{c}"]), 4) for c in CLASSES})
                st.caption("새 피해자 .88.81 공격은 전부 정상으로 판정돼(recall 0) 전체 값이 낮고, 이를 뺀 4클래스는 0.999 이상이나 각 유형이 서로 다른 도구·서버·캡처에서 나와 쉬운 구분입니다. 시연 모델은 시드 0 하나이며 이 표는 3시드 평균입니다.")
            else:
                st.info("평가 요약 파일이 이 브랜치에 없습니다.")
        st.divider()
        st.subheader("처리시간 vs 관측 대기시간(별도)")
        st.write(f"프로그램 처리시간: 파싱·조립 {timing['parse_and_assemble_s']:.3f}s · 특징 {timing['features_s']:.3f}s · 예측 {timing['predict_s']:.3f}s (합 {timing['total_s']:.3f}s, 연결 {timing['connections']:,}개, 이 노트북 측정)  \n"
                 f"관측 대기시간(패킷 시각 기준): 10개 관측 후 판정 연결(10번째 payload 패킷까지) 중앙값 {conn[conn.status == 'reached_k'].observation_wait_s.median():.3f}s · 90% {conn[conn.status == 'reached_k'].observation_wait_s.quantile(0.9):.3f}s"
                 f" / 10개 미도달 종료 확인 후 판정 연결(연결 시작~종료 증거 확인) 중앙값 {conn[conn.status == 'short_closed_padded'].observation_wait_s.median():.3f}s")
else:
    for t in (t2, t3, t4):
        t.info("① 자료 선택에서 모델과 pcap을 고르세요.")
