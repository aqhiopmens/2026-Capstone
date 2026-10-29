# Colab CPU 최종 재실행 (2026-10-29)

서은빈 승인으로 특징군 비교 48회와 시퀀스 순서 섞기·해시 조회 18회를 Colab에서 실행한다. 중복 결과 정리는 이 실행 뒤 별도 PR로 진행한다. 실행 모델·분할·정제 정책은 PR #13·#14와 같다.

## 실행 환경과 입력

- Drive 연결 및 Colab 화면에서 서은빈 계정을 확인했다. 기존 내 드라이브의 `2026-Capstone-data` 바로가기를 사용한다
- CPU 및 런타임 버전 **2026.07**을 선택했다. 첫 셀의 실제 버전 출력은 Python 3.12.13, numpy 2.0.2, pandas 2.2.2, sklearn 1.6.1, LightGBM 4.6.0, torch 2.11.0으로 일치했다. 호스팅 Colab 환경 표시와 GPU 미사용 검사도 통과했다. 패키지를 설치·업그레이드하지 않았다
- 원본 CSV는 기존 공유본을 사용한다. 재전처리 산출물 features.csv·rows.csv·splits.csv·invalid_rows.csv 4개는 [공유 processed/track_a_2026-10-29](https://drive.google.com/drive/folders/131CtBLZMCvDTlBVKR8D1664j8Effcask)에 새로 업로드하고 파일명·크기를 다시 확인했다. 원본·맥 결과는 덮어쓰지 않았다
- 모델 코드와 검사는 GitHub `133c802088d5e53707ff7784321bfa67705fd4be`의 파일을 그대로 사용한다. 비공개 저장소 토큰을 노트에 넣지 않고, 실행에 필요한 src·tests·의존성 정보·재전처리 메타데이터만 추린 개인 실행 묶음으로 전달했다
- 실행 묶음은 문서·PDF·PPTX·HWP·데이터 행·원본 Git 이력을 포함하지 않는다. 포함한 원본 파일 20개를 upstream의 `git show`와 SHA-256으로 대조해 일치했다. SOURCE_SNAPSHOT.json에 파일별 해시와 upstream 커밋을 보존했다
- 공통 결과 기록기가 Git 커밋을 요구하므로 최소 묶음에 새 로컬 snapshot Git 저장소를 만들었다. 실제 기록되는 snapshot 커밋은 `8a73b86a0956e2480281f30f6f4390c85982e56f`이며 upstream 커밋과 구분한다. 묶음 SHA-256은 `8b348e750d286ae63e92c3b1b042c9e96f499ba1cd24381c9c73969cbf29ec5e`다
- 개인 런처 노트는 모델 로직을 구현하지 않고 기존 `.py` 모듈을 호출한다. 소스·입력 존재와 Git clean 검사 및 검사 실행(returncode=0)을 통과한 뒤 학습을 시작했다

## 실행 식별자와 보존

| 순서 | 실행 ID | 조건 수 |
|---|---|---|
| 1 | `feature_group_2026-10-29_200024_colab` | 48 |
| 2 | `sequence_controls_2026-10-29_200024_colab` | 18 |
| 3 | 위 시퀀스 대조의 `_audit` | 재학습 없는 추가 점검 |

`MACHINE=colab-cpu`, `DATA_DIR`은 Drive의 공유 데이터 폴더로 설정했다. 두 번째 실행의 baseline-run-id는 첫 번째 Colab 배치 ID다. 맥 결과를 기준선으로 전달하지 않는다. 모델과 분할 시드, CPU worker=2, validation 미분리 등 세부 정책은 기존 [특징군 비교](feature_group_experiment.md)와 [시퀀스 대조](sequence_controls.md)를 따른다.

행별 예측·배정은 `$DATA_DIR/predictions/<run-id>/`에 남긴다. 집계는 Colab 작업 폴더의 `outputs/results/`에 저장하며, 약 45초 간격과 각 단계 종료 시 집계 ZIP을 `$DATA_DIR/colab_exports/2026-10-29_200024/`로 복사한다. 중간 ZIP은 불완전할 수 있는 체크포인트이며 최종 결과로 쓰지 않는다. 두 배치의 완료·건수·machine 검사가 통과한 뒤에만 final ZIP과 run_info의 complete=true를 만든다.

[Drive 실행 내보내기 폴더](https://drive.google.com/drive/folders/1ZOoPs1d_jYWsEISopgK-xmmPE8okQrAI). Git에는 이후 검증한 집계·메타데이터만 반영한다. 개인 런처 노트와 행별 자료는 넣지 않는다.

## 완료 결과와 검증

66/66회와 추가 감사가 완료됐고 두 manifest·run_info의 complete=true를 확인했다. Colab의 실행 셀은 약 25분에 종료됐다. final_028.zip(352,825바이트)의 집계·메타데이터 212개 파일을 가져왔다. 원본·맥 결과를 덮어쓰지 않았고 행별 예측·배정은 Drive에 그대로 남겨 두었다.

전체 특징 평균 macro-F1 ± 표본 표준편차(3시드):

| 모델 | 무작위 | 원본 패턴 그룹 |
|---|---|---|
| RF | 0.9721 ± 0.0011 | 0.9621 ± 0.0049 |
| LightGBM | 0.9783 ± 0.0003 | 0.9761 ± 0.0013 |

시퀀스 원순서→섞기 평균 macro-F1:

| 모델 | 분할 | 원순서 | 섞기 | 차이 |
|---|---|---|---|---|
| RF | 무작위 | 0.9142 | 0.7882 | −0.1260 |
| RF | 그룹 | 0.8798 | 0.7871 | −0.0927 |
| LightGBM | 무작위 | 0.9176 | 0.8696 | −0.0480 |
| LightGBM | 그룹 | 0.8828 | 0.8032 | −0.0796 |

그룹 섞기 편차는 RF 0.0409·LightGBM 0.0619다. 시드별 값은 각 배치 metrics.csv·paired.csv에 보존한다. 해시 조회는 무작위 macro-F1 0.7625, 그룹 미조회 fallback 0.0528이며 맥과 일치했다. 그룹 값은 조회율 0%의 최빈 benign 예측 성능으로 설명한다. 시드 1의 burpsuite 하락과 섞기 뒤 그룹 시험 1행 겹침의 해석 한계도 그대로 적용된다.

맥 대비 macro-F1 최대 절대 차이는 특징군 비교 RF 0.0017978, 섞기 RF 0.0058334였고 LightGBM·해시는 모든 조건에서 일치했다. RF 차이의 원인은 확정하지 않았다. 모델별 점수·평균·편차·정확도와 클래스별 결과를 분리 보존하며 유리한 시드를 다시 선택하지 않았다.

`python -m src.track_a.verify_colab_export`로 다음을 검증했다. 미완료 체크포인트는 최종본으로 거부하는 것도 확인했다.

- upstream과 실행 코드 20개 파일의 SHA-256, 실제 고정 환경, snapshot Git 커밋, 두 배치 완료와 입력 해시
- 66회 개별/통합 점수·혼동 행렬 11,154칸, 원본 배정의 클래스별 시험 분모, 모델 설정·machine·버전
- 혼동 행렬에서 직접 재계산한 macro-F1·정확도, 22조건 평균·ddof=1 표준편차, 12개 짝 비교(절대 오차 1e-12 이내)
- Colab 원순서 배치와 메타데이터를 실제 기준선으로 참조한 해시, 78개 해시 클래스 조회율, 156개 클래스별 짝 점수, 추가 감사의 코드·입력 참조
- ZIP CRC·경로·파일 수 및 행별 원자료 미포함. 변환 후 겹침과 순서 무시 값 집합 조회율은 맥과 일치

이는 Colab의 저장 집계를 검증한 것이며, Drive의 행별 예측 66개를 로컬로 내려받아 전부 재계산한 검증은 아니다. raw 행별 자료는 Git에 넣지 않는다. 상대 AI 교차 검토 후 최종 보고서에 사용한다.

집계 원본은 `outputs/results/feature_group_2026-10-29_200024_colab/`과 `outputs/results/sequence_controls_2026-10-29_200024_colab/`, 추가 감사는 후자의 `_audit/`에 있다. `outputs/results/colab_2026-10-29_200024/`에는 실제 run_info·source_snapshot·verification·export SHA 기록을 보존했다.

별도 정리 브랜치에서 맥·Colab 132개 개별 실행의 메타데이터를 배치별 `run_metadata.json` 4개로 보존했다. 중복 점수·혼동 행렬을 집계와 대조한 뒤 개별 파일 396개의 Git 추적만 해제한다. 원본 배치·감사·provenance 및 Drive ZIP은 그대로 보존한다. 절차와 과거 ZIP 검증 방법은 [workflow.md](workflow.md#완료-배치의-중복-결과-정리-1029)에 기록했다.

정리 검증: 테스트 39개 통과. 보존 메타데이터 132개는 정리 전 Git 원본과 바이트 단위로 일치했다. 새 임시 체크아웃에 개별 폴더가 없음을 확인하고, 맥·Colab 기준선 각각 12조건과 4개 아카이브 전체를 읽었다. 집계·manifest·감사 파일은 변경하지 않았다.
