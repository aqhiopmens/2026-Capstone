# 협업 워크플로우

김강민(윈도우) · 서은빈(맥) 2인이 GitHub + 구글드라이브 + Colab으로 작업하는 방식.

## 원칙

1. **코드는 GitHub 한 벌.** 공유 로직은 반드시 `.py`로 작성한다. 노트북(`.ipynb`)은 개인 탐색용으로만 쓴다.
2. **데이터는 드라이브 공유 폴더 한 곳.** git에는 데이터를 넣지 않는다.
3. **결과는 GitHub로 모은다.** 실험 결과 CSV를 `outputs/results/`에 커밋해 통합 결과표를 만든다.
4. **경로는 코드에 쓰지 않는다.** 데이터 위치는 환경변수 `DATA_DIR` 하나로만 받는다.

## 역할과 기계 배치 (확정 — 하드웨어 기준)

| | 김강민 (윈도우, 32GB) | 서은빈 (맥 애플 실리콘, 8GB) |
|---|---|---|
| 트랙 | B — pcap 추출기, 조기 탐지, 대시보드, 데모 영상 | A — Malicious_TLS 실험, 공통 코드 |
| 로컬에서 | 추출기 개발·실행, 대시보드, 시연 | 코드 작성, 작은 샘플로 확인 |
| Colab에서 | 최종 수치 재실행 | 본 실험 반복 (메모리 부족 대비) |
| 기타 | 원본 데이터 최초 업로드, hwp 문서 작업 | |

- **시연·발표 기계는 김강민 노트북**으로 고정한다. 대시보드는 11월 내내 이 기계에서 테스트한다.
- **공통 코드**(`src/common/` — 모델, 평가 함수, 결과 기록)는 A트랙 담당이 소유하고 B트랙 담당은 가져다 쓴다.

## 로컬 vs Colab

| 작업 | 어디서 | 이유 |
|---|---|---|
| 코드 작성·디버깅 | 로컬 | 빠른 반복 |
| pcap 추출 (약 4GB) | 로컬 (김강민) | 대용량 파일 읽기. Colab의 드라이브 읽기는 느림 |
| A트랙 전처리 (56MB) | 로컬 | 가벼움 |
| 본 실험 반복 (10조합 × 시드 3 = 30회, [plan.md](plan.md) 참조) | Colab Pro | 로컬 기계를 비워둠. 8GB 맥 부담 회피 |
| **보고서에 넣는 최종 수치** | **Colab** | 두 사람 PC와 무관한 중립 환경 → 누가 돌려도 같은 숫자 |
| 대시보드, 시연용 축소 pcap, 데모 영상 | 로컬 (김강민) | 발표 기계에서 인터넷 없이 동작해야 함 |

## 저장소 구조 (계획)

```
2026-Capstone/
├── src/
│   ├── common/      # 설정·모델·평가·결과 기록 (A트랙 담당 소유)
│   ├── track_a/     # Malicious_TLS 실험
│   └── track_b/     # pcap 추출기, 조기 탐지
├── app/             # 시연 대시보드 (Streamlit)
├── outputs/results/ # 실험 결과 CSV (git 추적)
├── docs/
└── notebooks/       # 캡디1 Colab 노트북 (참고용)
```

## 데이터 공유 (구글드라이브)

공유 폴더 이름: `2026-Capstone-data` (김강민이 만들어 서은빈에게 공유)

```
2026-Capstone-data/
├── raw/         # 원본. 수정 금지
│   ├── malicious_tls/
│   └── hikari/
└── processed/   # 추출·가공 결과. 파일명에 날짜 포함 (예: hikari_features_2026-10-20.csv)
```

- **공유 폴더는 `projects/` 동기화 대상 밖에 만든다.** 김강민 PC의 `projects/` 폴더는 드라이브로 단방향 동기화되고 있어서, 그 안에 상대가 파일을 넣으면 15분 안에 지워진다.
- 원본 파일은 덮어쓰지 않는다. 가공 결과는 새 파일로 올린다.

## 첫 세팅

### 공통
1. GitHub 저장소 초대 수락 후 `git clone https://github.com/aqhiopmens/2026-Capstone.git`
2. 드라이브 공유 폴더 `2026-Capstone-data`를 "내 드라이브에 바로가기 추가"

### 맥 (서은빈)
1. Homebrew가 없으면 먼저 설치
2. `brew install uv libomp` — `libomp`는 LightGBM에 필요
3. `~/.zshrc`에 `export DATA_DIR=<데이터를 둘 로컬 경로>` 추가
4. A트랙에 필요한 `raw/malicious_tls/`만 드라이브에서 받아 `DATA_DIR` 아래에 둔다
5. (파이썬 프로젝트 파일이 생긴 뒤) 저장소 폴더에서 `uv sync`

### 윈도우 (김강민)
1. `DATA_DIR` 환경변수를 로컬 데이터 폴더로 설정
2. (파이썬 프로젝트 파일이 생긴 뒤) 저장소 폴더에서 `uv sync`

### Colab
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/aqhiopmens/2026-Capstone.git
%cd 2026-Capstone
!pip install -r requirements.txt
import os
os.environ['DATA_DIR'] = '/content/drive/MyDrive/2026-Capstone-data'
```
`requirements.txt`는 `uv.lock`에서 내보낸 파일이다 (`uv export --no-hashes > requirements.txt`).

## 브랜치와 PR

1. 작업 시작 시 트랙별 브랜치를 만든다: `a/<작업명>`, `b/<작업명>`
2. 작업이 끝나면 푸시 → Pull Request → **상대가 훑어본 뒤** 병합
3. `main`에는 직접 푸시하지 않는다
4. `src/common/`을 고치는 PR은 두 사람 모두 확인한다 (양쪽 트랙이 쓰는 코드)

## 실험 결과 기록

모든 실험은 `outputs/results/`에 같은 형식의 CSV로 남긴다.

| 컬럼 | 예시 |
|---|---|
| date | 2026-10-12 |
| track | A / B |
| experiment | feature_group, ood, time_split, early_detection |
| split | random, pattern_group, time_0412to0417 |
| feature_group | all, fingerprint, stats, sequence |
| model | rf, lgbm, cnn1d, lstm |
| seed | 0, 1, 2 |
| macro_f1 | 0.8712 |
| accuracy | 0.9531 |
| machine | kyle-win, eunbin-mac, colab |
| note | 자유 기재 |

- 시드는 3회 반복하고 평균·편차를 보고한다
- macro-F1은 항상 함께 기록한다 (정확도 단독 보고 금지)
- `machine` 컬럼으로 어디서 나온 숫자인지 구분한다. 보고서에는 `colab` 행만 쓴다

## 주간 리듬

- **주 1회**: 두 트랙의 결과 CSV를 통합표로 합쳐 서로 교차 확인한다
- 이상한 숫자(너무 높거나 두 기계 결과가 다름)는 그 주에 원인을 찾는다

## 일정 기준점

| 시점 | 내용 |
|---|---|
| 9/30 | 과제수행계획서 제출 |
| 11월 2주 | **데모 영상 녹화** (최종 발표 PPT용, 라이브 시연 백업) |
| 11/20 | 발표자료·결과보고서 제출 |
| 11/23~26 | 연구실 평가 (교수님별 1팀 추천) |
| 12/1 | 발표회 (추천 시 라이브 시연) |
