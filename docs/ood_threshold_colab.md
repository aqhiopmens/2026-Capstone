# 미확인 문턱 평가 — Colab 최종 실행 (10/31)

서은빈이 계획 #28·구현/맥 결과 #30 병합과 Colab 준비를 승인했고 직접 CPU12회를 실행했다. 고정 환경 실행과 저장 예측 검증·로컬 집계 검증을 완료했다. **Claude 교차 검토·강민 최종 판단·결과 PR 병합은 대기**다. [설계](ood_threshold_plan.md), [맥 예비 결과](ood_threshold_experiment.md)를 따른다.

## 실행·출처

- 실행 ID `ood_threshold_2026-10-31_210713_colab`, RF100trees·balanced·workers2, 전체117/시퀀스30 × random/pattern_group × 시드0/1/2 = 신규12fit·36문턱·48정책. 기존 모델/예측 재사용·검증 합쳐 재학습·사후 문턱 조정 없음
- CPU2026.07, release `release-colab-external-images_20260715-060047_RC00`; Python3.12.13/numpy2.0.2/pandas2.2.2/sklearn1.6.1/LightGBM4.6.0/torch2.11.0. 패키지 설치 없음
- upstream `c8bf0c685fa0bc599500e49369afc991a8c5658e`, 최소 runtime snapshot `ad38fb64d5dde6c608128a952ee2526819565e18`. 원래 Git 이력과 구분한다. 시작 dirty·source dirty 모두 false, 코드 SHA와 입력 SHA는 맥 실행 및 병합 소스와 일치
- 소스 ZIP SHA `db38f2aad8dfa6fcd14b3ed5c02258b1e746b7bd987dce02dd3dca33a5c100d7`. 원본 데이터·과거 Git 이력을 제외한 소스37파일의 SHA는 `source_snapshot.json`에 보존
- [Drive 최종 ZIP](https://drive.google.com/file/d/1sesTQFpIvK4N9n2Y1jJzfKSX3QSImPNg/view) `final_007.zip`, 3,203,597바이트, SHA `16be0de02c149ef83d0e3619e885032dbb06196492e26e14c9ad0ede19f34a6b`. CRC·배치 경로·원본17파일을 확인했다
- Git에는 집계17파일과 실행 기록·소스 목록·다운로드 출처·로컬 검증·맥 비교·공유 위치를 남긴다. 행별 CSV는 Git에 넣지 않는다. 공유 DATA_DIR의 신규12폴더·24행파일 등록을 Drive에서 확인했다. `sharing.json`에 파일 ID/크기를 기록하고 원본 SHA·LF SHA는 `run_metadata.json`에 보존한다. 로컬에서 이24파일의 원격 바이트를 다시 다운로드하지는 않았다

## 주 정책 5%의 3시드 평균

| 분할 | 입력 | known 거절률 | unknown 행 거절률 | 보류5종 평균 | 패턴 평균 | 14종 F1 |
|---|---|---|---|---|---|---|
| random | all | 0.048986 | 0.979757 | 0.982524 | 0.981595 | 0.908458 |
| random | sequence | 0.042176 | 0.561104 | 0.596986 | 0.677718 | 0.778622 |
| pattern_group | all | 0.005049 | 0.583832 | 0.613000 | 0.761693 | 0.843951 |
| pattern_group | sequence | 0.115870 | 0.543890 | 0.585681 | 0.671336 | 0.703776 |

비율은 0~1이다. validation의 known 거절5% 이하로 고정한 문턱은 시험 거절5%를 보장하지 않는다. 그룹 시퀀스의 시험 known 거절 평균11.5870%도 그대로 보고한다. 같은 보류 행과 겹치는 known 시험 행을 세 시드가 사용하므로 표준편차를 독립 신뢰구간으로 읽지 않는다. 두 분할을 대조 조건으로 동등하게 보고한다.

## 맥 차이와 한계

48개 정책을 동일 split/입력/seed/policy로 짝 비교했다(`mac_colab_paired.csv`). 모든 정책 중 unknown 행 거절률 차이의 최대 절댓값은 0.243147, known 거절률은 0.083731, 14종 macro-F1은 0.081268이다. 3시드 평균 차이와 개별 정책 최대값을 혼동하지 않는다.

주5% unknown 행 거절률 평균은 맥→Colab에서 random all 0.977376→0.979757, random sequence 0.619917→0.561104, group all 0.519350→0.583832, group sequence 0.578085→0.543890이다. group sequence known 거절 평균은 0.089036→0.115870이다. 문턱 전 closed13 F1의 최대 실행별 절대 차이는 0.009709다. 같은 소스·입력에서도 RF 예측/문턱과 strict 경계에 따른 정책 결과 차이가 관측됐으나 **원인은 이번 집계만으로 확정하지 않는다**. 환경별 문턱·유일값·경계 동률 수를 보존하고 반올림/보간·사후 재조정을 하지 않는다. 재현 안정성의 한계로 보고한다.

보류5종의 고정 구성·같은 시험 집합 재사용·낮은 클래스별 탐지율 한계는 유지한다. unknown 거절을 악성 확정이나 보류 종류 식별로 부르지 않으며, 운영 발생률·범용 미지 공격 탐지·실시간 조기 탐지 보장으로 확대하지 않는다.

## 검증 범위와 다음 단계

Colab 독립 검증기가 원본·배정·439,616행의13종 확률·24파일 SHA에서 문턱36개, 정책48개, open14 행렬9,408칸·closed13 행렬2,028칸, 클래스864행, 패턴75,504행·동일가중 요약240행, AUC12개·점수/배정 감사·요약·짝 비교를 재계산해 통과했다. `verification.json`에 검증기 SHA와 원본 집계 SHA를 보존한다.

로컬은 최종 ZIP/집계 SHA·고정환경·소스/입력 일치·12개 메타데이터 원본 SHA를 확인하고, 저장 혼동 행렬의 F1/정확도·known/unknown/benign 거절/수용 지표, 클래스/패턴 집계,36문턱의 정수 예산,16정책·4closed 요약의 평균/표본편차와36개 무거절 대비 차이를 독립 재계산했다. **로컬 행별 CSV 재계산·학습·추론은 하지 않았다.** AUC·점수 분포·최대 문턱 선택의 행별 독립 재검증은 Colab 검증 범위이며 다음 Claude의 Windows 교차 검증에서 원본·저장 확률로 확인을 요청한다.

다음은 Claude 교차 검토(특히 맥/Colab 차이와 경계·그룹 시퀀스 과거절)와 강민 최종 판단이다. 결과 병합은 은빈님 별도 승인 후 진행한다.
