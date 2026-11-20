# 11월 12일 팀 전달본 실행

목표는 지정된 합성응답자에게 Fixed와 Non-graph로 각각 9문항을 제시하고,
수집한 응답을 팀의 공통 MNL/panel MMNL 추정기로 전달하는 것이다.
최종 비교 지표는 생성 계수에 대한 회복 Bias·RMSE이며, 이번 실행에는 예측 holdout이 없다.

## 확정된 설문 규칙

| Context | 기준 블록 | 공통 초기 문항 순서 | Adaptive 문항은행 |
|---|---:|---|---|
| Transfer | 1 | B1_Q2 → B1_Q4 → B1_Q9 | Block 1·2, 18개 |
| Layover | 3 | B3_Q1 → B3_Q3 → B3_Q6 | Block 3·4, 18개 |
| Stopover | 5 | B5_Q1 → B5_Q3 → B5_Q9 | Block 5·6, 18개 |

Fixed는 `fixed_task_order.csv`의 기준 블록 9문항을 순서대로 제시한다.
Non-graph는 공통 초기 3문항 이후 미응답 후보 15개에서 6문항을 순차 선택한다.
새 답변은 즉시 개인 이력에 추가된다. 학습된 전역 가중치는 설문 중 동결한다.

응답자는 `sampled_respondents_rep01.csv` … `rep10.csv`의 ID·프로필·Context를 그대로 사용한다.
한 회차는 Context별 500명, 총 1,500명이다. 같은 UUID가 다른 회차에 다시 등장할 수 있다.
`reserved_respondent_uuids.csv`가 열 회차 UUID의 합집합인지 확인하고,
이 합집합을 학습과 validation에서 모두 제외한다.

응답은 기존 `synthetic_responses_full.csv`의 `mnl_choice`와 `mmnl_choice`를 사용한다.
선택한 문항의 저장 응답만 실행기가 공개한다. 생성기 조건부 확률과 개인 latent taste는
선택기에 전달하지 않는다. 10회 반복은 고정 응답은행의 응답자 재추출이다.

## 입력과 준비

`--inputs` 디렉터리에 아래 자료를 둔다. 원본·프로필·모형·산출물은 Git에 올리지 않는다.

```text
nemotron_mapped.csv
sp_tasks.json
synthetic_responses_full.csv
sampling_manifest_10rep.json
reserved_respondent_uuids.csv
sampled_respondents_rep01.csv ... sampled_respondents_rep10.csv
```

팀 문항·출력 규격과 제공 검증기는 `configs/team_20261112/`에 있다.
원본 ZIP과 동일한 설계이며, manifest의 원 작성자 로컬 디렉터리 경로는 파일명으로 줄이고
저장소 CSV의 줄바꿈은 LF로 통일했다. 실제 제출 CSV는 UTF-8 BOM·CRLF로 저장한다.
기존 `sp_tasks.json`의 54문항 속성을 수정된 팀 카탈로그의 216개 대안과 대조한다.
원본 표본 파일 해시와 프로필 출처 해시도 확인한다.

```bash
python scripts/run_team_handoff.py prepare \
  --inputs /local/team-inputs --handoff configs/team_20261112 \
  --output outputs/team/prepared --train 1200 --validation 300
```

학습·validation은 평가 합집합 밖에서 공개 프로필만으로 고정 seed로 나눈다.
평가에 나타나는 프로필 범주 조합과 같은 지원을 가진 프로필을 사용한다.
응답은행은 한 번 순회해 필요한 응답만 SQLite로 보관한다. 원본과 선택 데이터의 해시,
분할 UUID, 설계 버전을 비공개 `prepared.json`에 기록한다.

## Non-graph 학습

기존 DeepSets choice learner와 respondent bootstrap ensemble 5개를 사용한다.
입력은 프로필 더미 9개, Context, 공개된 문항 속성과 선택 대안, 후보 문항 속성이다.
UUID와 task ID는 모델 특징으로 사용하지 않는다.

학습 이력 길이는 0~8개에서 표본 추출한다. 길이가 3개 이상이면 지정 초기 3문항을 포함하고,
추가 문항을 무작위로 선택한다. 이력 밖의 선택을 cross-entropy로 예측한다.
Validation NLL로 조기 종료하고 평가 응답은 학습·선택에 사용하지 않는다.
기본 상한은 모델별 120 epochs × 1,024 episodes, patience 30이다.

```bash
python scripts/run_team_handoff.py train --prepared outputs/team/prepared \
  --generator mnl --checkpoint outputs/team/checkpoints/mnl.pt
python scripts/run_team_handoff.py train --prepared outputs/team/prepared \
  --generator panel_mmnl --checkpoint outputs/team/checkpoints/panel_mmnl.pt
```

Weighted JS는 후보 문항의 네 가상 답변으로 다른 남은 후보들의 예측이 얼마나 달라지는지
현재 답변 확률로 가중해 계산한다. GNN 비교에서도 동일한 후보 범위·공개 이력·점수를 사용해야 한다.
이 점수는 예측 의존성의 대리 지표다. 계수 회복을 직접 최적화하지 않으므로,
그 유용성은 수집 응답의 공통 최종 추정 결과로 판단한다.

## 설문 실행과 제출

```bash
# 먼저 Context별 2명의 pilot을 별도 경로에서 확인한다.
python scripts/run_team_handoff.py run --prepared outputs/team/prepared \
  --generator panel_mmnl --method non_graph --replicate 1 \
  --checkpoint outputs/team/checkpoints/panel_mmnl.pt \
  --limit-per-context 2 --output outputs/team/pilot

# 본 실행: generator 두 가지와 replicate 1~10을 각각 실행한다.
python scripts/run_team_handoff.py run --prepared outputs/team/prepared \
  --generator panel_mmnl --method non_graph --replicate 1 \
  --checkpoint outputs/team/checkpoints/panel_mmnl.pt \
  --output outputs/team/delivery
python scripts/run_team_handoff.py run --prepared outputs/team/prepared \
  --generator panel_mmnl --method fixed --replicate 1 \
  --output outputs/team/delivery
```

선택된 응답은 한 단계씩 UTF-8 BOM CSV에 기록한다. 완료 전에는 `.csv.partial`로 보관하고,
제공 검증기를 통과한 파일만 `.csv`로 바꾼다. 실패 시 부분 응답과 실행 상태가 남는다.
기존 결과를 덮어쓰거나 미완료 실행을 성공으로 세지 않는다.

`--accelerated`는 동결 모델의 문항·이벤트 인코딩을 캐시한다. 같은 float32 층을 계산하되
덧셈 순서가 달라 미세한 반올림 차이가 생길 수 있다. 테스트에서 원래 예측·Weighted JS와
대조한다. 점수가 거의 같은 후보 사이의 선택은 수치 정밀도의 영향을 받을 수 있다.

```bash
python scripts/run_team_handoff.py collect --output outputs/team/delivery \
  --handoff configs/team_20261112
```

| 산출물 | 크기와 의미 |
|---|---|
| `responses_rep01_mnl_non_graph.csv` 등 | 1회차 × 생성환경 × 방법, 파일당 13,500행·32열 |
| Non-graph 전체 | 20개 CSV, 270,000행 |
| Fixed + Non-graph 전체 | 40개 CSV, 540,000행 |
| `run_manifest.csv` | 실행별 seed·버전·완료 상태 |
| `*.verification.json` | 제공 검증기 통과 여부, 파일·체크포인트 해시 |
| `*.training.json` | 학습 설정, validation 이력, 선택 epoch |

표본·답변이 방법 간 일치하는지 별도로 교차 확인한다. 각 파일의 단독 검증만으로는
다른 방법과의 응답 일치까지 보장되지 않는다.

```bash
python scripts/audit_team_delivery.py --prepared outputs/team/prepared \
  --delivery outputs/team/delivery --output outputs/team/delivery_audit.json
```

이 검사는 40개 본 실행 전부에 제공 검증기를 적용하고, 원본 회차별 표본의 18개 열,
응답은행의 실제 답, 실행 완료 기록과 파일 해시를 함께 대조한다.
`prepared/`는 필요한 응답 인덱스·프로필·문항·표본·전달본을 자체 포함하므로
그 디렉터리와 체크포인트를 함께 옮기면 같은 실행을 다른 경로에서 재현할 수 있다.

최종 모형 추정은 Junyuk이 진행한다. 초기 3응답을 포함한 9응답 전체를 사용하고,
MNL 환경은 공통 MNL, panel MMNL 환경은 공통 panel MMNL로 추정한다.
표본에서 식별 가능한 계수와 동일 정규화 기준을 맞춘 뒤 생성 계수와 비교해야 한다.
