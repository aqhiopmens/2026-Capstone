# 원본 재전처리 특징군 비교

## 실행 조건

- 입력: 원본 재전처리 `track_a_2026-10-29`의 features·rows·splits. 원본 CSV와 Git의 재전처리 메타데이터도 필요하다
- 원본 SHA-256, 저장 특징과 원본 형식 변환 값의 정확한 일치, 행 ID·라벨·용도·패턴 ID, 6개 외부 배정을 검사한다. 저장 배정이 해당 시드로 재계산한 배정과 다르면 학습 전에 중단한다
- 알려진 13종, 특징군 4개(all/fingerprint/stats/sequence) × 분할 2개(random/pattern_group) × 모델 2개(RF/LightGBM) × 시드 3개(0/1/2) = **48회**
- PR #12 공통 코드를 사용한다. 각 조건에서 외부 train 전체를 학습하며 validation은 0행이다. 무작위 학습 52,667행·시험 13,167행, 그룹 학습 52,828행·시험 13,006행이다. 보류 12,877행은 학습·알려진 시험 지표에서 제외한다
- RF·LightGBM 모두 100개 트리·balanced, CPU worker 기본 2개. 스케일링·특징 선택·극단값/중복 제거·튜닝·문턱 조정은 하지 않는다. 시드가 분할과 모델을 함께 바꾸므로 변동 원인을 분리해 주장하지 않는다
- 모델별 점수·13×13 혼동 행렬을 각각 저장한다. unknown의 최대 확률은 원자료이며 미확인 위협 거절률이 아니다

```sh
# DATA_DIR, MACHINE을 먼저 설정한다. run-id는 매번 새로운 값을 쓴다.
python -m src.track_a.feature_group_experiment \
  --run-id feature_group_2026-10-29_mac --preprocess-run-id 2026-10-29 --workers 2
```

맥에서는 `MACHINE=eunbin-mac`, Colab CPU에서는 `MACHINE=colab-cpu`를 사용한다. 로컬 결과는 예비로 표시한다. Colab도 [workflow](workflow.md)의 2026.07 런타임·버전 확인 후 동일한 원본·가공 파일을 DATA_DIR에 두고 동일 명령을 실행한다. run-id만 새 값(예: `feature_group_2026-10-30_colab`)으로 바꾼다. Colab 결과도 버전·교차 검증을 통과한 뒤 보고서에 사용한다. 스크립트가 machine 문자열만으로 버전 적합성을 보증하지는 않는다.

## 산출물과 실패 처리

| 위치 | 내용 |
|---|---|
| `outputs/results/<batch-id>/` | 48회 metrics.csv, 8,112행 confusion.csv, 16조건 summary.csv, manifest.json |
| `outputs/results/<batch-id>_<분할>_<모델>_<특징군>_s<시드>/` | 공통 API의 개별 metrics·confusion·metadata |
| `$DATA_DIR/predictions/<개별-run-id>/` | 행별 예측·최종 행 배정. Git에 넣지 않는다 |

각 모델 실행 후 집계 파일과 `completed_runs`를 체크포인트로 기록한다. 48회와 16개 집계 조건이 모두 완성되고 입력·코드 해시가 시작 시점과 같을 때만 `manifest.complete=true`로 만든다. 중단된 배치는 불완전 상태로 남으며, 기존 batch·개별 run-id는 덮어쓰지 않는다. 자동 재개는 지원하지 않는다. 실패 원인을 확인하고 새로운 run-id로 재실행한다.

manifest에는 실행 시작 Git 커밋·dirty 여부, 입력과 실행 코드/uv.lock의 SHA-256을 남긴다. 개별 metadata의 dirty가 true여도 새로 생성한 결과 파일 때문일 수 있으므로 코드 수정과 혼동하지 않는다. 실행 중 입력·코드를 수정하지 않는다.

세 시드 평균·표본 표준편차(ddof=1)는 모델별로 분리한다. 그룹 시험 집합은 시드 사이 겹침이 크므로 표준편차를 독립 재표집 신뢰구간으로 해석하지 않는다. [분할·보류 한계](track_a_preprocessing.md), [공통 API](common_experiments.md)를 함께 참고한다.

## 검증

`python -m unittest discover -s tests`로 기존 검사와 입력 파일 손상·시드 배정 변경·중단 배치의 불완전 상태 유지 검사를 수행한다. 실제 48회 결과는 실행 완료 후 이 문서에 별도로 기록한다.
