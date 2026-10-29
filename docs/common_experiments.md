# 공통 RF·LightGBM 실험 코드

두 트랙이 공유하는 모델·평가·기록 API다. 정식 48회 실행기는 아직 없으며 이 단계에서는 합성 데이터로만 모델을 학습했다. 딥러닝, 스케일러, 미확인 위협 문턱 조정은 후속 단계다.

## 계약과 기본값

- **[실험 1] 특징군 비교에서는 검증 분리를 생략하고 외부 train 전체를 학습한다.** 현재 고정 RF·LightGBM에는 조기 종료·파라미터 탐색이 없어 검증셋이 필요하지 않다. API는 validation 없는 배정도 지원한다. 미확인 위협 문턱 조정 단계에서는 외부 train 내부에 검증셋을 분리하고, 해당 최종 train으로 별도 학습한 모델의 validation 예측으로만 문턱을 정한다. 시험·보류 데이터로 문턱을 맞추지 않는다
- `models.make_model`: RF·LightGBM 모두 100개 트리, `class_weight="balanced"`, 명시적인 모델 시드, 기본 CPU worker 2개. LightGBM은 deterministic·force_col_wise를 켠다. RF의 나머지 기본값과 라이브러리 버전은 메타데이터에 남긴다
- `splits.split_validation`: 외부 train의 약 20%를 검증으로 분리한다. 무작위는 클래스 층화, 패턴 그룹은 전역 그룹 ID를 시드로 섞고 SGKF 5개 폴드 중 클래스별 최종 학습·검증 30행 이상인 후보를 선택한다. 검증 비율 20%와의 차이, 폴드 번호순으로 고르며 점수는 보지 않는다. 기존 test·unknown·excluded·invalid 배정은 바꾸지 않는다
- `splits.validate_partitions`: 고유 행 ID와 순서, 알려진 클래스 전체 포함, 보류 클래스 격리를 검사한다. 그룹 조건에서는 train·validation·test·unknown 사이 모든 패턴 겹침을 거부한다. 무작위 조건은 패턴 겹침을 허용한다
- `experiment.run_classification`: 선택한 수치 특징 DataFrame, 별도의 라벨·배정 Series를 받는다. 행 ID와 순서가 같아야 한다. fit은 최종 train에만 수행하며 validation을 다시 합쳐 재학습하지 않는다. 스케일링·특징 선택·행 삭제·조기 종료·하이퍼파라미터 탐색은 하지 않는다
- `evaluation.evaluate_classification`: 명시한 고정 클래스 전체로 macro-F1, 정확도, 0칸을 포함한 혼동 행렬을 계산한다. 클래스 누락·미확인 라벨 혼입은 중단한다
- validation·unknown에는 예측 클래스와 최대 확률만 생성한다. **이 단계의 unknown 예측은 미확인 거절 결과가 아니다.** 알려진 시험 지표에 포함하지 않으며, 문턱은 아직 만들지 않는다
- B의 시간 분할에서는 train/validation/test를 시간 규칙으로 직접 배정하고 공통 실행기에 전달한다. `split_validation`으로 임의 무작위 검증을 만들지 않는다. B의 IP·포트·SNI·JA3 격리와 트랙별 클래스·품질 정책은 호출자가 적용해야 한다

## 저장과 재현

`results.save_run`은 `DATA_DIR`·`MACHINE` 환경변수를 읽는다. DATA_DIR은 저장소 밖이어야 하며 기존 run-id는 어느 쪽 디렉터리에도 덮어쓰지 않는다.

| 위치 | 파일 |
|---|---|
| Git의 `outputs/results/<run-id>/` | `metrics.csv`, `confusion.csv`, `metadata.json` |
| `$DATA_DIR/predictions/<run-id>/` | `predictions.csv`(검증·시험·보류 행별 실제/예측/최대 확률), `assignments.csv`(모든 행의 최종 배정) |

결과 CSV는 workflow의 기본 컬럼에 `split_seed`, 행 수·특징 수·학습 시간을 추가한다. `seed`는 모델 시드다. 메타데이터에는 입력 파일 SHA-256, 모델 전체 파라미터, 특징 목록, 클래스 목록, Git 커밋·dirty 여부, 라이브러리 버전을 남긴다. 모델 본체는 저장하지 않는다. 행별 자료나 실제 경로를 Git에 올리지 않는다. 입력 파일 해시는 호출자가 `sha256_file`로 계산해 전달한다.

`git_dirty`는 `git status --porcelain`으로 확인한 작업 폴더 전체 상태다. 미추적 PDF·PPTX나 앞선 실행에서 생성한 결과 파일 때문에도 true가 될 수 있어 코드 수정 여부와 같지 않다. 최종 보고서용 실행은 코드를 커밋하고 작업 폴더 상태를 확인한 뒤 시작하며, true이면 원인을 별도로 확인한다. 표시를 깨끗하게 만들려고 자료를 삭제하거나 숨기지 않는다.

`summarize_runs`는 같은 실험·분할·특징군·모델·기계·버전끼리 정확히 3개의 서로 다른 (모델 시드, 분할 시드) 결과가 모였을 때 평균과 표본 표준편차(ddof=1)를 낸다. 불완전한 조건과 중복 결과는 거부한다. 표준편차는 해당 3회 실행의 기술 통계이며 독립 재표집 신뢰구간이 아니다. 시드 사이 시험 집합 겹침의 한계는 [재전처리 기록](track_a_preprocessing.md)에 있다.

## A 호출 예시 (검토 후 정식 실행 단계에서 사용)

아래는 한 조건의 사용 예시다. 이번 구현 검증에서 이 코드로 실제 모델을 학습하지 않았다. 모든 특징군·모델에 동일한 최종 배정을 재사용하고, 예비 정제본 대신 원본 재전처리 산출물을 쓴다.

```python
import pandas as pd
from src.common.config import DATA_DIR
from src.common.experiment import run_classification
from src.common.results import save_run, sha256_file
from src.common.splits import split_validation
from src.track_a.preprocess import feature_groups, validate_assignment

folder = DATA_DIR / "processed/track_a_2026-10-29"
features = pd.read_csv(folder / "features.csv", float_precision="round_trip").set_index("source_row")
rows = pd.read_csv(folder / "rows.csv").set_index("source_row")
splits = pd.read_csv(folder / "splits.csv").set_index("source_row")
if not features.index.equals(rows.index) or not rows.index.equals(splits.index):
    raise ValueError("입력 파일의 행 ID 또는 순서 불일치")
if not features.Label.equals(rows.Label):
    raise ValueError("특징 파일과 행 대장의 라벨 불일치")
split, split_seed, model_seed = "pattern_group", 0, 0
outer = splits[f"{split}_{split_seed}"]
assignment = rows[["Label", "pattern_id"]].assign(partition=outer).reset_index()
validate_assignment(features, assignment, split)
partitions = outer.copy()  # [실험 1]: 검증 분리 없이 외부 train 전체 사용
# 후속 문턱 조정 단계에서만 아래 배정으로 별도 모델을 학습한다.
# partitions = split_validation(rows.Label, outer, split=split, seed=split_seed,
#                               groups=rows.pattern_id)
known_labels = sorted(rows.loc[rows.role.eq("known"), "Label"].unique())
result = run_classification(features[feature_groups()["all"]], rows.Label, partitions,
                            known_labels=known_labels, model_name="rf", seed=model_seed,
                            split=split, groups=rows.pattern_id)
save_run(result, run_id="feature_group_A_group_all_rf_s0", track="A",
         experiment="feature_group", feature_group="all", split_seed=split_seed,
         source_hashes={name: sha256_file(folder / name)
                        for name in ("features.csv", "rows.csv", "splits.csv")},
         note="원본 재전처리; 최종 train만 fit; 실행 환경은 machine 참조")
```

## 검증 기록 (2026-10-29)

- 합성 데이터에서 RF·LightGBM의 fit 대상이 최종 train뿐인지, unknown을 알려진 클래스 지표에 섞지 않는지, macro-F1·혼동 행렬·저장 위치 분리·덮어쓰기 거부를 검사했다
- 맥에 저장된 행 대장과 외부 분할 6개를 읽어 검증 분할만 계산했다. 실제 데이터의 모델 학습·파일 출력은 하지 않았다. 아래 표는 **검증을 분리하는 경우의 기능 검사**이며, [실험 1]의 실제 학습 배정은 아니다. [실험 1]은 무작위 52,667행·그룹 52,828행을 학습하고 validation은 0행이다

| 조건 | 최종 train | validation | 외부 test | 클래스별 최소 train / validation / test |
|---|---|---|---|---|
| random 시드 0·1·2 | 42,133 | 10,534 | 13,167 | 709 / 177 / 221 |
| pattern_group 시드 0·1 | 42,020 | 10,808 | 13,006 | 709 / 177 / 221 |
| pattern_group 시드 2 | 42,019 | 10,809 | 13,006 | 709 / 177 / 221 |

그룹 조건의 train·validation·test·unknown 간 패턴 겹침은 0건이다. 보류 12,877행·제외 13,314행·invalid 9행도 바뀌지 않았다. 데이터는 Git에 포함하지 않는다.

검사 명령: `uv run python -m unittest discover -s tests`. `src/common/` 변경이므로 A·B 양쪽 담당이 확인하고 별도 승인을 받아 병합한다.

Claude는 김강민 요청으로 테스트 21개와 별도 원본의 검증 분리를 재확인했고, 김강민 확인을 포함해 병합 가능하다고 보고했다. 검증 분리 선택화·B 시간 배정·dirty 해석의 확인 사항을 검토했으며, 서은빈이 문서 보완과 PR #12 병합을 승인했다. 단일 시드 성능 비교로 검증 정책을 고르지 않았다.
