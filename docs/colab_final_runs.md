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

실행 완료 수치·검증 결과는 종료 후 추가한다. 다음 정리 PR은 실행별 메타데이터를 배치에 통합 보존하고 기준선 로더를 맞춘 뒤 중복 추적을 해제하며, 새 체크아웃에서 로더가 동작하는지 검증한다.
